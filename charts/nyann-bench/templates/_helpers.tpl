{{- define "nyann.digestImage" -}}
{{- $field := index . 0 -}}
{{- $image := required (printf "%s is required" $field) (index . 1) -}}
{{- if not (regexMatch "^[^@[:space:]]+@sha256:[0-9a-f]{64}$" $image) -}}
{{- fail (printf "%s must be digest pinned as IMAGE@sha256:<64 lowercase hex characters>" $field) -}}
{{- end -}}
{{- $image -}}
{{- end -}}
