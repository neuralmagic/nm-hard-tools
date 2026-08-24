{{- define "lmeval.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- define "lmeval.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "lmeval.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "lmeval.workerServiceAccount" -}}
{{- printf "%s-worker" (include "lmeval.fullname" . | trunc 56 | trimSuffix "-") -}}
{{- end -}}

{{- define "lmeval.controllerServiceAccount" -}}
{{- printf "%s-controller" (include "lmeval.fullname" . | trunc 52 | trimSuffix "-") -}}
{{- end -}}

{{- define "lmeval.controllerNetworkPolicy" -}}
{{- printf "%s-controller" (include "lmeval.fullname" . | trunc 52 | trimSuffix "-") -}}
{{- end -}}

{{- define "lmeval.workerNetworkPolicy" -}}
{{- printf "%s-workers" (include "lmeval.fullname" . | trunc 55 | trimSuffix "-") -}}
{{- end -}}

{{- define "lmeval.image" -}}
{{- $repository := required "image.repository is required" .Values.image.repository -}}
{{- $digest := required "image.digest is required and must be sha256:<64 hex>" .Values.image.digest -}}
{{- if not (regexMatch "^sha256:[a-f0-9]{64}$" $digest) -}}
{{- fail "image.digest must be sha256:<64 hex>" -}}
{{- end -}}
{{- printf "%s@%s" $repository $digest -}}
{{- end -}}

{{- define "lmeval.resultClaim" -}}
{{- default (printf "%s-results" (include "lmeval.fullname" . | trunc 55 | trimSuffix "-")) .Values.persistence.existingClaim -}}
{{- end -}}
