{{- define "model-deployment.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "model-deployment.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "model-deployment.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "model-deployment.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "model-deployment.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- required "serviceAccount.name is required when serviceAccount.create=false" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}
