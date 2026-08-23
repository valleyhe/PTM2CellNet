#!/usr/bin/env bash
# Robust wheel downloader for the isolated PerturbGen env (unstable network).
#
# pip's own resume logic loses progress when the connection breaks with a raw
# ProtocolError (observed twice at ~99% of large nvidia wheels). This script
# instead downloads every resolved wheel with `curl -C -` into a persistent
# directory and retries each file until its sha256 matches PyPI's record.
#
# Usage: bash scripts/download_perturbgen_wheels.sh [report.json] [outdir]
set -uo pipefail

REPORT="${1:-/tmp/pip_report.json}"
OUTDIR="${2:-$HOME/perturbgen_wheels}"
PY="${PY:-/home/scu/anaconda3/envs/perturbgen/bin/python}"

mkdir -p "$OUTDIR"

# Emit url|sha256|filename lines from the pip dry-run report.
"$PY" - "$REPORT" <<'EOF' > "$OUTDIR/_manifest.tsv"
import json, sys
report = json.load(open(sys.argv[1]))
for item in report["install"]:
    info = item["download_info"]
    url = info["url"]
    # pip report uses "sha256=<hex>"; normalize to bare hex.
    sha = info.get("archive_info", {}).get("hash", "").split("=", 1)[-1]
    print(f"{url}\t{sha}\t{url.rsplit('/', 1)[-1]}")
EOF

total=$(wc -l < "$OUTDIR/_manifest.tsv")
i=0
fail=0
while IFS=$'\t' read -r url sha name; do
    i=$((i + 1))
    dest="$OUTDIR/$name"
    if [ -f "$dest" ] && echo "$sha  $dest" | sha256sum -c --quiet >/dev/null 2>&1; then
        echo "[$i/$total] $name OK (cached)"
        continue
    fi
    echo "[$i/$total] $name ..."
    # curl byte-resume loop: keep retrying until the file is complete, then
    # verify sha256; loop forever on mismatch-free but incomplete files.
    for attempt in $(seq 1 500); do
        curl -sSfL -C - --retry 20 --retry-all-errors --connect-timeout 30 \
             --speed-limit 1024 --speed-time 60 -o "$dest" "$url" && break
        sleep 3
    done
    if echo "$sha  $dest" | sha256sum -c --quiet >/dev/null 2>&1; then
        echo "[$i/$total] $name verified"
    else
        echo "[$i/$total] $name FAILED verification"
        fail=$((fail + 1))
    fi
done < "$OUTDIR/_manifest.tsv"

echo "done: $((total - fail))/$total ok, failures=$fail"
exit "$([ "$fail" -eq 0 ] && echo 0 || echo 1)"
