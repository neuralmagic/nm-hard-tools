{{- define "diffprobe.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- define "diffprobe.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "diffprobe.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "diffprobe.workerServiceAccount" -}}
{{- printf "%s-worker" (include "diffprobe.fullname" . | trunc 56 | trimSuffix "-") -}}
{{- end -}}

{{- define "diffprobe.controllerServiceAccount" -}}
{{- printf "%s-controller" (include "diffprobe.fullname" . | trunc 52 | trimSuffix "-") -}}
{{- end -}}

{{- define "diffprobe.controllerNetworkPolicy" -}}
{{- printf "%s-controller" (include "diffprobe.fullname" . | trunc 52 | trimSuffix "-") -}}
{{- end -}}

{{- define "diffprobe.workerNetworkPolicy" -}}
{{- printf "%s-workers" (include "diffprobe.fullname" . | trunc 55 | trimSuffix "-") -}}
{{- end -}}

{{- define "diffprobe.image" -}}
{{- $repository := required "image.repository is required" .Values.image.repository -}}
{{- $digest := required "image.digest is required and must be sha256:<64 hex>" .Values.image.digest -}}
{{- if not (regexMatch "^sha256:[a-f0-9]{64}$" $digest) -}}
{{- fail "image.digest must be sha256:<64 hex>" -}}
{{- end -}}
{{- printf "%s@%s" $repository $digest -}}
{{- end -}}

{{- define "diffprobe.resultClaim" -}}
{{- default (printf "%s-results" (include "diffprobe.fullname" . | trunc 55 | trimSuffix "-")) .Values.persistence.existingClaim -}}
{{- end -}}
