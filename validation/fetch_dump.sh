#!/usr/bin/env bash
# Streams the public InferenceX database dump (~25 GB, 13 parts) and keeps only the tables
# needed for validation. Nothing but the extracted tables is written to disk.
# Needs zstd and pg_restore >= 17 (the dump uses archive format 1.16).
# Usage: TAG=2026-09-14 OUT=./tables.copy ./fetch_dump.sh
set -euo pipefail
TAG=${TAG:-2026-09-14}
OUT=${OUT:-./tables.copy}
PG_RESTORE=${PG_RESTORE:-pg_restore}
U=https://github.com/SemiAnalysisAI/InferenceX-app/releases/download/db-dump/$TAG
PARTS=$(curl -sSfL "$U/SHA256SUMS" | awk '{print $2}' | grep '\.dump\.zst\.part' | sort)
for p in $PARTS; do curl -sSfL --retry 5 "$U/$p"; echo "downloaded $p" >&2; done \
  | zstd -dc \
  | "$PG_RESTORE" --data-only --no-owner --no-privileges \
      -t configs -t benchmark_results -t workflow_runs -f "$OUT"
