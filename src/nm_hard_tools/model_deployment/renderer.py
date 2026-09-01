"""Safe caller-YAML validation and pinned llm-manifesto rendering."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from nm_hard_tools.model_deployment.config import DeploymentSettings
from nm_hard_tools.model_deployment.models import ResourceRef

OWNER_ANNOTATION = "nm-hard-tools.neuralmagic.com/deployment-id"
DIGEST_ANNOTATION = "nm-hard-tools.neuralmagic.com/desired-state-digest"
MAX_YAML_NODES = 10_000
MANIFESTO_REVISION = "ebb7932c63967d622cf8f7e8a8e7781c5214a779"
ALLOWED_RESOURCE_KINDS = {
    "ConfigMap",
    "Deployment",
    "DestinationRule",
    "EndpointPickerConfig",
    "Gateway",
    "HTTPRoute",
    "InferencePool",
    "LeaderWorkerSet",
    "PersistentVolumeClaim",
    "Service",
    "ServiceAccount",
    "StatefulSet",
}
FORBIDDEN_KEYS = {
    "accelerator",
    "cluster",
    "cluster_profile",
    "context",
    "credential",
    "credentials",
    "env",
    "fabric_profile",
    "hf_home",
    "identity",
    "image_pull_secrets",
    "kube_context",
    "local_queue",
    "namespace",
    "routing_profile",
    "routing",
    "secret",
    "secret_name",
    "secrets",
    "submission_policy",
    "trace_dir",
    "user",
    "user_root",
    "vars",
    "vllm_env",
    "cache_root",
    "log_root",
}


class ManifestoConfigError(ValueError):
    pass


class OperatorConfigurationError(ValueError):
    pass


class _UniqueSafeLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: _UniqueSafeLoader, node: yaml.MappingNode, deep=False):
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ManifestoConfigError("duplicate YAML mapping key")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def parse_manifesto_config(source: str) -> dict[str, Any]:
    document_count = 0
    node_count = 0
    try:
        for event in yaml.parse(source, Loader=yaml.SafeLoader):
            if isinstance(event, yaml.events.DocumentStartEvent):
                document_count += 1
            if isinstance(event, yaml.events.AliasEvent):
                raise ManifestoConfigError("YAML aliases are not allowed")
            if isinstance(
                event,
                (
                    yaml.events.ScalarEvent,
                    yaml.events.SequenceStartEvent,
                    yaml.events.MappingStartEvent,
                ),
            ):
                node_count += 1
                if node_count > MAX_YAML_NODES:
                    raise ManifestoConfigError("YAML exceeds the 10000-node limit")
                tag = getattr(event, "tag", None)
                if tag is not None and not tag.startswith("tag:yaml.org,2002:"):
                    raise ManifestoConfigError("custom YAML tags are not allowed")
        if document_count != 1:
            raise ManifestoConfigError("exactly one YAML document is required")
        value = yaml.load(source, Loader=_UniqueSafeLoader)
    except ManifestoConfigError:
        raise
    except (RecursionError, TypeError, yaml.YAMLError) as exc:
        raise ManifestoConfigError("manifesto_config is not valid safe YAML") from exc
    if not isinstance(value, dict):
        raise ManifestoConfigError("manifesto_config must decode to an object")
    _reject_forbidden_fields(value)
    if "extends" in value:
        raise ManifestoConfigError("manifesto_config must be self-contained")
    return value


def _reject_forbidden_fields(value: Any) -> None:
    pending: list[Any] = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            for key, nested in item.items():
                if not isinstance(key, str):
                    raise ManifestoConfigError("YAML mapping keys must be strings")
                if key.lower().replace("-", "_") in FORBIDDEN_KEYS:
                    raise ManifestoConfigError(
                        "operator-controlled field is not allowed"
                    )
                pending.append(nested)
        elif isinstance(item, list):
            pending.extend(item)


@dataclass(frozen=True)
class WorkloadExpectation:
    name: str
    expected_pods: int
    pod_selector: dict[str, str]


@dataclass(frozen=True)
class RenderedDeployment:
    manifesto_digest: str
    intent_digest: str
    deployment_id: str
    namespace: str
    endpoint: str
    objects: list[dict[str, Any]]
    resources: list[ResourceRef]
    workloads: list[WorkloadExpectation]

    @property
    def yaml(self) -> str:
        return yaml.safe_dump_all(self.objects, sort_keys=False, explicit_start=True)


class ManifestoRenderer:
    def __init__(self, settings: DeploymentSettings) -> None:
        self.settings = settings

    def render(self, source: str) -> RenderedDeployment:
        parsed = parse_manifesto_config(source)
        manifesto_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        intent_digest = _intent_digest(parsed)
        try:
            renderer_version = importlib.metadata.version("llm-manifesto")
            pinned_renderer = f"{renderer_version}+{MANIFESTO_REVISION}"
            context_id = self.settings.render_context_identity(pinned_renderer)
            cluster = self._load_cluster()
        except Exception as exc:
            raise OperatorConfigurationError(
                "Operator render configuration is invalid"
            ) from exc
        try:
            identity_bytes = (
                f"nm-hard-tools-deploy-v2\0{intent_digest}\0{context_id}".encode()
            )
            deployment_id = "hard-" + hashlib.sha256(identity_bytes).hexdigest()[:24]
            objects, endpoint = self._render_manifesto(parsed, deployment_id, cluster)
        except ManifestoConfigError:
            raise
        except Exception as exc:
            raise ManifestoConfigError(
                "Manifesto rejected the model configuration"
            ) from exc
        if not objects or len(objects) > 64:
            raise ManifestoConfigError(
                "Manifesto must render between 1 and 64 resources"
            )
        resources: list[ResourceRef] = []
        workloads: list[WorkloadExpectation] = []
        for obj in objects:
            if obj.get("kind") not in ALLOWED_RESOURCE_KINDS:
                raise ManifestoConfigError(
                    f"rendered resource kind is not allowed: {obj.get('kind')}"
                )
            metadata = obj.setdefault("metadata", {})
            if (
                metadata.get("namespace", self.settings.namespace)
                != self.settings.namespace
            ):
                raise ManifestoConfigError(
                    "rendered resource escaped the target namespace"
                )
            metadata["namespace"] = self.settings.namespace
            annotations = metadata.setdefault("annotations", {})
            annotations[OWNER_ANNOTATION] = deployment_id
            annotations.pop(DIGEST_ANNOTATION, None)
            _annotate_pod_templates(obj, deployment_id, intent_digest)
            desired = hashlib.sha256(
                json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            annotations[DIGEST_ANNOTATION] = desired
            resources.append(
                ResourceRef(
                    api_version=str(obj.get("apiVersion", "")),
                    kind=str(obj.get("kind", "")),
                    name=str(metadata.get("name", "")),
                )
            )
            workload = _workload_expectation(obj)
            if workload is not None:
                workloads.append(workload)
        if not workloads:
            raise ManifestoConfigError("Manifesto rendered no model workload")
        selectors = {
            tuple(sorted(workload.pod_selector.items())) for workload in workloads
        }
        if len(selectors) != len(workloads):
            raise ManifestoConfigError(
                "Manifesto rendered model workloads with ambiguous pod selectors"
            )
        return RenderedDeployment(
            manifesto_digest=manifesto_digest,
            intent_digest=intent_digest,
            deployment_id=deployment_id,
            namespace=self.settings.namespace,
            endpoint=endpoint,
            objects=objects,
            resources=resources,
            workloads=workloads,
        )

    def _render_manifesto(
        self, parsed: dict[str, Any], deployment_id: str, cluster: Any
    ) -> tuple[list[dict[str, Any]], str]:
        from manifesto.instance import Instance
        from manifesto.render import render
        from manifesto.render.routing import gateway_name, standalone_service_name
        from manifesto.resolve import resolve_role
        from manifesto.spec import RoutingFrontend, RoutingKind, TopologyKind, load_spec

        with tempfile.TemporaryDirectory(prefix="nm-hard-manifesto-") as temp_dir:
            path = Path(temp_dir) / "model.yaml"
            render_input = {**parsed, "release": deployment_id}
            path.write_text(
                yaml.safe_dump(render_input, sort_keys=False), encoding="utf-8"
            )
            spec = load_spec(path, cluster)
        spec.namespace = self.settings.namespace
        spec.release = deployment_id
        spec.runtime.idle_shutdown.enabled = False
        if spec.topology != TopologyKind.AGGREGATED:
            raise ManifestoConfigError(
                "operator routing policy currently supports aggregated topology only"
            )
        spec.routing.kind = RoutingKind.DISABLED
        objects = render(spec, user="nm-hard-tools", cluster=cluster)
        instance = Instance(
            user="nm-hard-tools",
            release=spec.release,
            include_user_in_name=cluster.naming.user_prefix,
        )
        if spec.routing.kind == RoutingKind.DISABLED:
            role = next(
                (item for item in spec.roles if item.name == "decode"), spec.roles[0]
            )
            port = resolve_role(spec, instance, cluster, role).ports.public[0]
            service = instance.name(f"{role.name}-svc")
        elif spec.routing.frontend == RoutingFrontend.STANDALONE:
            service = standalone_service_name(instance)
            port = 80
        else:
            service = gateway_name(instance, cluster) + f"-{cluster.gateway.class_name}"
            port = 80
        endpoint = (
            f"http://{service}.{self.settings.namespace}.svc.cluster.local:{port}"
            "/v1/models"
        )
        return objects, endpoint

    def _load_cluster(self) -> Any:
        from manifesto.cluster import load_cluster

        return load_cluster(self.settings.cluster_profile)


def _workload_expectation(obj: dict[str, Any]) -> WorkloadExpectation | None:
    kind = obj.get("kind")
    spec = obj.get("spec", {})
    metadata = obj.get("metadata", {})
    labels = metadata.get("labels", {})
    component = labels.get("app.kubernetes.io/component", "")
    if component != "model-server":
        return None
    selector: dict[str, str]
    if kind == "Deployment":
        selector = spec.get("selector", {}).get("matchLabels", {})
        expected_pods = int(spec.get("replicas", 1))
    if kind == "LeaderWorkerSet":
        replicas = int(spec.get("replicas", 1))
        size = int(spec.get("leaderWorkerTemplate", {}).get("size", 1))
        expected_pods = replicas * size
        selector = {
            key: str(labels[key])
            for key in ("app.kubernetes.io/instance", "llm-d.ai/role")
            if key in labels
        }
    elif kind != "Deployment":
        return None
    if (
        expected_pods < 1
        or not selector
        or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in selector.items()
        )
    ):
        raise ManifestoConfigError("Manifesto rendered an invalid model workload")
    return WorkloadExpectation(
        name=str(metadata.get("name", "")),
        expected_pods=expected_pods,
        pod_selector=selector,
    )


def _intent_digest(parsed: dict[str, Any]) -> str:
    intent = {key: value for key, value in parsed.items() if key != "release"}
    try:
        canonical = json.dumps(
            intent,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ManifestoConfigError(
            "manifesto_config contains a non-canonical value"
        ) from exc
    return hashlib.sha256(
        b"nm-hard-tools-manifesto-intent-v1\0" + canonical
    ).hexdigest()


def _annotate_pod_templates(
    obj: dict[str, Any], deployment_id: str, manifesto_digest: str
) -> None:
    templates: list[dict[str, Any]] = []
    spec = obj.get("spec", {})
    if obj.get("kind") in {"Deployment", "StatefulSet"}:
        templates.append(spec.get("template", {}))
    elif obj.get("kind") == "LeaderWorkerSet":
        group = spec.get("leaderWorkerTemplate", {})
        templates.extend(
            [group.get("leaderTemplate", {}), group.get("workerTemplate", {})]
        )
    for template in templates:
        metadata = template.setdefault("metadata", {})
        annotations = metadata.setdefault("annotations", {})
        annotations[OWNER_ANNOTATION] = deployment_id
        annotations[DIGEST_ANNOTATION] = manifesto_digest
