"""Deterministic probe planning and Kubernetes Job rendering."""

from __future__ import annotations

import hashlib
import json

from nm_hard_tools.diff_probe import SERVICE_VERSION
from nm_hard_tools.diff_probe.config import ProbeServiceSettings
from nm_hard_tools.diff_probe.models import (
    ProbePlanResponse,
    ProbeRequest,
    canonical_request,
    probe_id_from_plan,
    request_sha256,
)

ANNOTATION_PREFIX = "probe.nm-hard-tools.io/"
ID_LABEL = f"{ANNOTATION_PREFIX}probe-id"
MANAGED_BY = "diff-probe-service"


def plan_probe(
    settings: ProbeServiceSettings, request: ProbeRequest
) -> ProbePlanResponse:
    """Render the complete immutable worker plan without creating resources."""
    targets = {}
    for role, name in (("a", request.target_a), ("b", request.target_b)):
        if name is None:
            continue
        target = settings.targets.get(name)
        if target is None:
            allowed = ", ".join(sorted(settings.targets))
            raise ValueError(f"unknown target {name!r}; allowed targets: {allowed}")
        targets[role] = (name, target)
    request_set = settings.request_sets.get(request.request_set)
    if request_set is None:
        allowed = ", ".join(sorted(settings.request_sets))
        raise ValueError(
            f"unknown request set {request.request_set!r}; "
            f"allowed request sets: {allowed}"
        )
    if request.limit is not None and request.limit > request_set.max_requests:
        raise ValueError(
            f"limit exceeds request-set maximum {request_set.max_requests}"
        )
    max_tokens = request.max_tokens or request_set.default_max_tokens
    chat = request_set.api == "chat"
    endpoint_identities = {
        role: target.endpoint_identity(chat) for role, (_, target) in targets.items()
    }
    digest = request_sha256(request)
    identity_inputs = {
        "request": request.model_dump(mode="json", exclude_none=False),
        "request_set": request_set.model_dump(mode="json"),
        "endpoint_identities": endpoint_identities,
        "targets": {
            role: {
                "logical_name": name,
                "tls_verify": target.tls_verify,
                "max_concurrency": target.max_concurrency,
                "auth_secret_name": target.auth_secret_name,
                "auth_secret_key": target.auth_secret_key,
            }
            for role, (name, target) in targets.items()
        },
        "worker_image": settings.worker_image,
        "result_claim_name": settings.result_claim_name,
        "result_root": settings.result_root,
        "request_set_claim_name": settings.request_set_claim_name,
        "request_set_root": settings.request_set_root,
        "namespace": settings.namespace,
        "worker_service_account": settings.worker_service_account,
        "service_instance": settings.service_instance,
        "image_pull_secrets": settings.image_pull_secrets,
        "node_selector": settings.node_selector,
        "resources": {
            "cpu_request": settings.cpu_request,
            "cpu_limit": settings.cpu_limit,
            "memory_request": settings.memory_request,
            "memory_limit": settings.memory_limit,
            "cache_size_limit": settings.worker_cache_size_limit,
        },
        "service_version": SERVICE_VERSION,
    }
    plan_digest = hashlib.sha256(
        json.dumps(identity_inputs, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    probe = probe_id_from_plan(plan_digest)
    effective = {
        **request.model_dump(mode="json", exclude_none=False),
        "probe_id": probe,
        "request_sha256": digest,
        "plan_sha256": plan_digest,
        "resolved_max_tokens": max_tokens,
        "api_mode": "chat_completions" if chat else "completions",
        "request_set_configuration": request_set.model_dump(mode="json"),
        "endpoint_identities": endpoint_identities,
        "sampling": {"temperature": 0.0, "top_p": 1.0, "max_tokens": max_tokens},
        "service_version": SERVICE_VERSION,
        "worker_image": settings.worker_image,
    }
    worker_config = {
        "probe_id": probe,
        "request": request.model_dump(mode="json", exclude_none=False),
        "request_set": request_set.model_dump(mode="json"),
        "endpoints": {
            role: {
                "logical_name": name,
                "url": target.endpoint(chat),
                "identity": endpoint_identities[role],
                "tls_verify": target.tls_verify,
                "api_key_env": f"OPENAI_API_KEY_{role.upper()}",
            }
            for role, (name, target) in targets.items()
        },
        "max_tokens": max_tokens,
        "effective_configuration": effective,
    }
    annotations = {
        f"{ANNOTATION_PREFIX}request-sha256": digest,
        f"{ANNOTATION_PREFIX}plan-sha256": plan_digest,
        f"{ANNOTATION_PREFIX}request": canonical_request(request),
        f"{ANNOTATION_PREFIX}endpoint-identities": json.dumps(
            endpoint_identities, sort_keys=True, separators=(",", ":")
        ),
        f"{ANNOTATION_PREFIX}effective-configuration": json.dumps(
            effective, sort_keys=True, separators=(",", ":")
        ),
    }
    env: list[dict] = [
        {
            "name": "DIFF_PROBE_CONFIG",
            "value": json.dumps(worker_config, sort_keys=True, separators=(",", ":")),
        },
        {"name": "DIFF_PROBE_RESULT_ROOT", "value": settings.result_root},
        {"name": "DIFF_PROBE_REQUEST_SET_ROOT", "value": settings.request_set_root},
        {"name": "HOME", "value": "/tmp"},
    ]
    for role, (_, target) in targets.items():
        variable = f"OPENAI_API_KEY_{role.upper()}"
        if target.auth_secret_name:
            env.append(
                {
                    "name": variable,
                    "valueFrom": {
                        "secretKeyRef": {
                            "name": target.auth_secret_name,
                            "key": target.auth_secret_key,
                        }
                    },
                }
            )
        else:
            env.append({"name": variable, "value": ""})
    labels = {
        "app.kubernetes.io/name": "diff-probe",
        "app.kubernetes.io/managed-by": MANAGED_BY,
        "app.kubernetes.io/instance": settings.service_instance,
        ID_LABEL: probe,
        f"{ANNOTATION_PREFIX}request-set": request.request_set,
    }
    volume_mounts = [
        {"name": "results", "mountPath": settings.result_root},
        {"name": "tmp", "mountPath": "/tmp"},
    ]
    volumes = [
        {
            "name": "results",
            "persistentVolumeClaim": {"claimName": settings.result_claim_name},
        },
        {"name": "tmp", "emptyDir": {"sizeLimit": settings.worker_cache_size_limit}},
    ]
    if settings.request_set_claim_name:
        volume_mounts.append(
            {
                "name": "request-sets",
                "mountPath": settings.request_set_root,
                "readOnly": True,
            }
        )
        volumes.append(
            {
                "name": "request-sets",
                "persistentVolumeClaim": {
                    "claimName": settings.request_set_claim_name,
                    "readOnly": True,
                },
            }
        )
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": probe,
            "namespace": settings.namespace,
            "labels": dict(labels),
            "annotations": annotations,
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": request.timeout_seconds + 60,
            "template": {
                "metadata": {"labels": dict(labels)},
                "spec": {
                    "serviceAccountName": settings.worker_service_account,
                    "nodeSelector": dict(settings.node_selector),
                    "imagePullSecrets": [
                        {"name": name} for name in settings.image_pull_secrets
                    ],
                    "automountServiceAccountToken": False,
                    "restartPolicy": "Never",
                    "securityContext": {
                        "runAsNonRoot": True,
                        "runAsUser": 65532,
                        "runAsGroup": 65532,
                        "fsGroup": 65532,
                        "fsGroupChangePolicy": "OnRootMismatch",
                        "seccompProfile": {"type": "RuntimeDefault"},
                    },
                    "containers": [
                        {
                            "name": "prober",
                            "image": settings.worker_image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": [
                                "python",
                                "-m",
                                "nm_hard_tools.diff_probe.worker",
                            ],
                            "env": env,
                            "resources": {
                                "requests": {
                                    "cpu": settings.cpu_request,
                                    "memory": settings.memory_request,
                                },
                                "limits": {
                                    "cpu": settings.cpu_limit,
                                    "memory": settings.memory_limit,
                                },
                            },
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "capabilities": {"drop": ["ALL"]},
                                "readOnlyRootFilesystem": True,
                            },
                            "volumeMounts": volume_mounts,
                        }
                    ],
                    "volumes": volumes,
                },
            },
        },
    }
    return ProbePlanResponse(
        probe_id=probe,
        request_sha256=digest,
        plan_sha256=plan_digest,
        endpoint_identities=endpoint_identities,
        effective_configuration=effective,
        resources=[job],
    )
