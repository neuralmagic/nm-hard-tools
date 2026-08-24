"""Deterministic lm-eval configuration and Kubernetes Job rendering."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.resources
import json
from pathlib import Path

from inference_hard_tools.evaluation_service import SERVICE_VERSION
from inference_hard_tools.evaluation_service.config import ServiceSettings, TaskConfig
from inference_hard_tools.evaluation_service.models import (
    EvaluationRequest,
    PlanResponse,
    canonical_request,
    evaluation_id_from_plan,
    request_sha256,
)

ANNOTATION_PREFIX = "evaluation.lm-eval.io/"


def _ensure_tasks_installed(profile_name: str, tasks: list[TaskConfig]) -> None:
    root = importlib.resources.files("lm_eval")
    for task in tasks:
        config_path = task.config_path
        relative = Path(config_path).relative_to("lm_eval")
        if not root.joinpath(*relative.parts).is_file():
            raise ValueError(
                f"profile {profile_name!r} references missing installed task "
                f"configuration {config_path!r}"
            )


def _resolve_request(
    settings: ServiceSettings, request: EvaluationRequest
) -> tuple[int, int]:
    profile = settings.profiles.get(request.profile)
    if profile is None:
        allowed = ", ".join(sorted(settings.profiles))
        raise ValueError(
            f"unknown profile {request.profile!r}; allowed profiles: {allowed}"
        )
    num_fewshot = (
        request.num_fewshot
        if request.num_fewshot is not None
        else profile.default_num_fewshot
    )
    generation_limit = (
        request.generation_limit
        if request.generation_limit is not None
        else profile.default_generation_limit
    )
    if num_fewshot > profile.max_num_fewshot:
        raise ValueError(
            f"num_fewshot exceeds profile maximum {profile.max_num_fewshot}"
        )
    if generation_limit > profile.max_generation_limit:
        raise ValueError(
            f"generation_limit exceeds profile maximum {profile.max_generation_limit}"
        )
    if request.limit is not None and request.limit > profile.max_samples:
        raise ValueError(f"limit exceeds profile maximum {profile.max_samples}")
    return num_fewshot, generation_limit


def plan_evaluation(
    settings: ServiceSettings,
    request: EvaluationRequest,
    lm_eval_revision: str,
) -> PlanResponse:
    """Render the complete immutable worker plan without creating resources."""
    target = settings.targets.get(request.target)
    if target is None:
        allowed = ", ".join(sorted(settings.targets))
        raise ValueError(
            f"unknown target {request.target!r}; allowed targets: {allowed}"
        )
    num_fewshot, generation_limit = _resolve_request(settings, request)
    profile = settings.profiles[request.profile]
    dataset_token_secret_name = (
        profile.dataset_token_secret_name or settings.hf_token_secret_name
    )
    dataset_token_secret_key = (
        profile.dataset_token_secret_key or settings.hf_token_secret_key
        if dataset_token_secret_name
        else None
    )
    _ensure_tasks_installed(request.profile, list(profile.tasks))
    chat = request.chat_template == "default"
    digest = request_sha256(request)
    endpoint_identity = target.endpoint_identity(chat)
    profile_configuration = profile.model_dump(mode="json", exclude_none=False)
    identity_inputs = {
        "request": request.model_dump(mode="json", exclude_none=False),
        "profile": profile_configuration,
        "endpoint_identity": endpoint_identity,
        "worker_image": settings.worker_image,
        "lm_eval_commit": lm_eval_revision,
        "tls_verify": target.tls_verify,
        "max_concurrency": target.max_concurrency,
        "auth_secret_name": target.auth_secret_name,
        "auth_secret_key": target.auth_secret_key,
        "result_claim_name": settings.result_claim_name,
        "result_root": settings.result_root,
        "dataset_cache_claim_name": settings.dataset_cache_claim_name,
        "dataset_cache_root": settings.dataset_cache_root,
        "dataset_token_secret_name": dataset_token_secret_name,
        "dataset_token_secret_key": dataset_token_secret_key,
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
    evaluation = evaluation_id_from_plan(plan_digest)
    invocation = {
        "model_backend": "local-chat-completions" if chat else "local-completions",
        "model_args": {
            "model": request.model,
            "base_url": target.endpoint(chat),
            "tokenizer_backend": "none",
            "tokenized_requests": False,
            "num_concurrent": target.max_concurrency,
            "max_retries": 2,
            "max_gen_toks": generation_limit,
            "timeout": min(request.timeout_seconds, 300),
            "verify_certificate": target.tls_verify,
            "seed": request.seeds.python,
        },
        "tasks": [task.model_dump(mode="json") for task in profile.tasks],
        "primary_metric": profile.primary_metric.model_dump(mode="json"),
        "num_fewshot": num_fewshot,
        "batch_size": 1,
        "limit": request.limit,
        "bootstrap_iters": 0,
        "log_samples": True,
        "apply_chat_template": chat,
        "fewshot_as_multiturn": chat,
        "generation_kwargs": {
            "max_gen_toks": generation_limit,
            "do_sample": False,
            "temperature": 0.0,
        },
        "seeds": request.seeds.model_dump(mode="json"),
        "confirm_run_unsafe_code": False,
        "cache_requests": False,
        "rewrite_requests_cache": False,
        "delete_requests_cache": False,
        "check_integrity": False,
        "write_out": False,
        "predict_only": False,
        "system_instruction": None,
        "metadata": {},
    }
    effective = {
        **request.model_dump(mode="json", exclude_none=False),
        "resolved_num_fewshot": num_fewshot,
        "resolved_generation_limit": generation_limit,
        "evaluation_id": evaluation,
        "request_sha256": digest,
        "plan_sha256": plan_digest,
        "resolved_target": endpoint_identity,
        "api_mode": "chat_completions" if chat else "completions",
        "profile_configuration": profile_configuration,
        "target_configuration": {
            "logical_name": request.target,
            "endpoint_identity": endpoint_identity,
            "tls_verify": target.tls_verify,
            "max_concurrency": target.max_concurrency,
            "auth_secret_ref": (
                {"name": target.auth_secret_name, "key": target.auth_secret_key}
                if target.auth_secret_name
                else None
            ),
        },
        "job_configuration": {
            "namespace": settings.namespace,
            "service_account": settings.worker_service_account,
            "service_instance": settings.service_instance,
            "image_pull_secrets": settings.image_pull_secrets,
            "node_selector": settings.node_selector,
            "backoff_limit": 0,
            "active_deadline_seconds": request.timeout_seconds + 60,
            "cpu_request": settings.cpu_request,
            "cpu_limit": settings.cpu_limit,
            "memory_request": settings.memory_request,
            "memory_limit": settings.memory_limit,
            "cache_size_limit": settings.worker_cache_size_limit,
        },
        "dataset_cache": {
            "mode": (
                "shared-read-only"
                if settings.dataset_cache_claim_name
                else "per-job-ephemeral"
            ),
            "claim_name": settings.dataset_cache_claim_name,
            "root": (
                settings.dataset_cache_root
                if settings.dataset_cache_claim_name
                else "/tmp/huggingface"
            ),
        },
        "dataset_auth_secret_ref": (
            {
                "name": dataset_token_secret_name,
                "key": dataset_token_secret_key,
            }
            if dataset_token_secret_name
            else None
        ),
        "lm_eval_version": importlib.metadata.version("lm_eval"),
        "lm_eval_commit": lm_eval_revision,
        "lm_eval_invocation": invocation,
        "service_version": SERVICE_VERSION,
        "worker_image": settings.worker_image,
        "result_backend": {
            "type": "persistent_volume_claim",
            "claim_name": settings.result_claim_name,
            "root": settings.result_root,
        },
    }
    worker_config = {
        "evaluation_id": evaluation,
        "request": request.model_dump(mode="json", exclude_none=False),
        "endpoint_identity": endpoint_identity,
        "effective_configuration": effective,
        "invocation": invocation,
    }
    annotations = {
        f"{ANNOTATION_PREFIX}request-sha256": digest,
        f"{ANNOTATION_PREFIX}plan-sha256": plan_digest,
        f"{ANNOTATION_PREFIX}endpoint-identity": endpoint_identity,
        f"{ANNOTATION_PREFIX}request": canonical_request(request),
        f"{ANNOTATION_PREFIX}effective-configuration": json.dumps(
            effective, sort_keys=True, separators=(",", ":")
        ),
    }
    cache_root = (
        settings.dataset_cache_root
        if settings.dataset_cache_claim_name
        else "/tmp/huggingface"
    )
    env = [
        {
            "name": "LM_EVAL_EVALUATION_CONFIG",
            "value": json.dumps(worker_config, sort_keys=True, separators=(",", ":")),
        },
        {"name": "LM_EVAL_RESULT_ROOT", "value": settings.result_root},
        {"name": "HOME", "value": "/tmp"},
        {"name": "XDG_CACHE_HOME", "value": "/tmp/cache"},
        {"name": "HF_HOME", "value": cache_root},
        {"name": "HF_DATASETS_CACHE", "value": f"{cache_root}/datasets"},
    ]
    if settings.dataset_cache_claim_name:
        env.extend(
            [
                {"name": "HF_HUB_OFFLINE", "value": "1"},
                {"name": "HF_DATASETS_OFFLINE", "value": "1"},
            ]
        )
    if dataset_token_secret_name:
        env.append(
            {
                "name": "HF_TOKEN",
                "valueFrom": {
                    "secretKeyRef": {
                        "name": dataset_token_secret_name,
                        "key": dataset_token_secret_key,
                    }
                },
            }
        )
    if target.auth_secret_name:
        env.append(
            {
                "name": "OPENAI_API_KEY",
                "valueFrom": {
                    "secretKeyRef": {
                        "name": target.auth_secret_name,
                        "key": target.auth_secret_key,
                    }
                },
            }
        )
    else:
        env.append({"name": "OPENAI_API_KEY", "value": ""})
    labels = {
        "app.kubernetes.io/name": "lm-evaluation",
        "app.kubernetes.io/managed-by": "lm-eval-service",
        "app.kubernetes.io/instance": settings.service_instance,
        "evaluation.lm-eval.io/evaluation-id": evaluation,
        "evaluation.lm-eval.io/profile": request.profile,
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
        {
            "name": "tmp",
            "emptyDir": {"sizeLimit": settings.worker_cache_size_limit},
        },
    ]
    if settings.dataset_cache_claim_name:
        volume_mounts.append(
            {
                "name": "dataset-cache",
                "mountPath": settings.dataset_cache_root,
                "readOnly": True,
            }
        )
        volumes.append(
            {
                "name": "dataset-cache",
                "persistentVolumeClaim": {
                    "claimName": settings.dataset_cache_claim_name,
                    "readOnly": True,
                },
            }
        )
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": evaluation,
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
                            "name": "evaluator",
                            "image": settings.worker_image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": [
                                "python",
                                "-m",
                                "inference_hard_tools.evaluation_service.worker",
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
    return PlanResponse(
        evaluation_id=evaluation,
        request_sha256=digest,
        plan_sha256=plan_digest,
        endpoint_identity=endpoint_identity,
        effective_configuration=effective,
        resources=[job],
    )
