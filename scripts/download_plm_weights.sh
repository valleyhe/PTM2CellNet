#!/usr/bin/env bash
# Download the pinned Ankh Base, ESM-2 150M and ProtT5 encoder checkpoints.
#
# Usage:
#   bash scripts/download_plm_weights.sh
#
# Optional overrides:
#   PROXY_URL=http://192.168.100.216:7897
#   MODEL_ROOT=/some/path/data/weights/plm
#   DOWNLOAD_METADATA=0    # weights only; default is 1

set -Eeuo pipefail
IFS=$'\n\t'

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '1,14p' "$0"
    exit 0
fi

readonly PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PROXY_URL="${PROXY_URL:-http://192.168.100.216:7897}"
readonly MODEL_ROOT="${MODEL_ROOT:-$PROJECT_ROOT/data/weights/plm}"
readonly DOWNLOAD_METADATA="${DOWNLOAD_METADATA:-1}"
readonly CURL_RETRIES="${CURL_RETRIES:-20}"
readonly DOWNLOAD_ATTEMPTS="${DOWNLOAD_ATTEMPTS:-10}"
readonly CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-60}"
readonly SPEED_TIME="${SPEED_TIME:-120}"
readonly SPEED_LIMIT="${SPEED_LIMIT:-1024}"

require_command() {
    command -v "$1" >/dev/null 2>&1 || {
        printf 'ERROR: required command not found: %s\n' "$1" >&2
        exit 2
    }
}

require_command curl
require_command sha256sum
require_command stat
require_command flock

mkdir -p "$MODEL_ROOT"
readonly LOCK_PATH="$MODEL_ROOT/.download.lock"
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
    printf 'ERROR: another download script already holds %s\n' "$LOCK_PATH" >&2
    exit 3
fi

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

# curl 7.71+ supports --retry-all-errors.  Keep compatibility with older
# curl versions while retaining retries for the installed curl on this host.
CURL_RETRY_FLAGS=(--retry "$CURL_RETRIES" --retry-delay 5)
if curl --help all 2>/dev/null | grep -q -- '--retry-all-errors'; then
    CURL_RETRY_FLAGS+=(--retry-all-errors)
fi

curl_download() {
    local url="$1"
    local target="$2"

    curl \
        --proxy "$PROXY_URL" \
        --location \
        --fail \
        --silent \
        --show-error \
        "${CURL_RETRY_FLAGS[@]}" \
        --connect-timeout "$CONNECT_TIMEOUT" \
        --speed-time "$SPEED_TIME" \
        --speed-limit "$SPEED_LIMIT" \
        --continue-at - \
        --output "$target" \
        "$url"
}

download_weight() {
    local name="$1"
    local repo_id="$2"
    local revision="$3"
    local remote_file="$4"
    local target="$5"
    local expected_size="$6"
    local expected_sha256="$7"
    local url="https://huggingface.co/${repo_id}/resolve/${revision}/${remote_file}?download=true"
    local partial="${target}.part"
    local attempt current_size actual_sha256

    mkdir -p "$(dirname -- "$target")"

    if [[ -e "$target" ]]; then
        current_size="$(stat -c '%s' "$target")"
        if [[ "$current_size" == "$expected_size" ]]; then
            actual_sha256="$(sha256sum "$target" | awk '{print $1}')"
            if [[ "$actual_sha256" == "$expected_sha256" ]]; then
                log "$name: already verified; skipping"
                return 0
            fi
        fi
        printf 'ERROR: final file exists but failed verification: %s\n' "$target" >&2
        printf 'Move it aside manually before retrying; it was not overwritten.\n' >&2
        return 1
    fi

    for ((attempt = 1; attempt <= DOWNLOAD_ATTEMPTS; attempt++)); do
        current_size=0
        [[ -e "$partial" ]] && current_size="$(stat -c '%s' "$partial")"
        log "$name: attempt ${attempt}/${DOWNLOAD_ATTEMPTS}, resume_bytes=${current_size}"

        if curl_download "$url" "$partial"; then
            current_size="$(stat -c '%s' "$partial")"
            if [[ "$current_size" != "$expected_size" ]]; then
                log "$name: incomplete size=${current_size}, expected=${expected_size}; retrying"
                sleep 5
                continue
            fi

            actual_sha256="$(sha256sum "$partial" | awk '{print $1}')"
            if [[ "$actual_sha256" != "$expected_sha256" ]]; then
                printf 'ERROR: SHA-256 mismatch for %s\n' "$name" >&2
                printf '  expected: %s\n  actual:   %s\n' "$expected_sha256" "$actual_sha256" >&2
                printf '  partial file retained: %s\n' "$partial" >&2
                return 1
            fi

            mv --no-clobber -- "$partial" "$target"
            log "$name: complete and verified (${expected_size} bytes)"
            return 0
        fi

        current_size=0
        [[ -e "$partial" ]] && current_size="$(stat -c '%s' "$partial")"
        log "$name: curl failed; retained ${current_size} bytes for resume"
        sleep 5
    done

    printf 'ERROR: retry limit exhausted for %s; partial file retained: %s\n' "$name" "$partial" >&2
    return 1
}

download_metadata() {
    local repo_id="$1"
    local revision="$2"
    local remote_file="$3"
    local target="$4"
    local url="https://huggingface.co/${repo_id}/resolve/${revision}/${remote_file}?download=true"
    local partial="${target}.part"

    [[ -e "$target" ]] && return 0
    mkdir -p "$(dirname -- "$target")"
    log "metadata: ${repo_id}/${remote_file}"
    curl_download "$url" "$partial"
    [[ -s "$partial" ]] || {
        printf 'ERROR: empty metadata file: %s\n' "$partial" >&2
        return 1
    }
    mv --no-clobber -- "$partial" "$target"
}

log "using proxy: ${PROXY_URL}"
log "model root: ${MODEL_ROOT}"

download_weight \
    'Ankh Base' \
    'ElnaggarLab/ankh-base' \
    'd99cb6b966530dfc2ae96bc69d9255c2a07308b' \
    'pytorch_model.bin' \
    "$MODEL_ROOT/ankh-base/pytorch_model.bin" \
    '2946066741' \
    '9b2a886374f0ff4a893f4e7a989deed76bb2458c8998bd5202ea8e97d92ddcc3'

download_weight \
    'ESM-2 150M' \
    'facebook/esm2_t30_150M_UR50D' \
    'a695f6045e2e32885fa60af20c13cb35398ce30c' \
    'model.safetensors' \
    "$MODEL_ROOT/esm2_t30_150M_UR50D/model.safetensors" \
    '595257706' \
    'c3f1da8aea53bddd32c246c86168c23b9fd72341fb9db9a94436f855f5053566'

download_weight \
    'ProtT5 XL half encoder' \
    'Rostlab/prot_t5_xl_half_uniref50-enc' \
    '94a6abc029ae13029317b140b7424e012bf8dfbf' \
    'pytorch_model.bin' \
    "$MODEL_ROOT/prot_t5_xl_half_uniref50-enc/pytorch_model.bin" \
    '2416373051' \
    '7f51ba885541c7dc569d46b796af57cc7a2ba7945107dced4f19d1b5ec091157'

if [[ "$DOWNLOAD_METADATA" == '1' ]]; then
    log 'downloading tokenizer/config files'

    for file in config.json special_tokens_map.json tokenizer.json tokenizer_config.json; do
        download_metadata \
            'ElnaggarLab/ankh-base' \
            'd99cb6b966530dfc2ae96bc69d9255c2a07308b' \
            "$file" \
            "$MODEL_ROOT/ankh-base/$file"
    done

    for file in config.json special_tokens_map.json tokenizer_config.json vocab.txt; do
        download_metadata \
            'facebook/esm2_t30_150M_UR50D' \
            'a695f6045e2e32885fa60af20c13cb35398ce30c' \
            "$file" \
            "$MODEL_ROOT/esm2_t30_150M_UR50D/$file"
    done

    for file in config.json special_tokens_map.json spiece.model tokenizer_config.json; do
        download_metadata \
            'Rostlab/prot_t5_xl_half_uniref50-enc' \
            '94a6abc029ae13029317b140b7424e012bf8dfbf' \
            "$file" \
            "$MODEL_ROOT/prot_t5_xl_half_uniref50-enc/$file"
    done
fi

log 'all requested model files completed and verified'
