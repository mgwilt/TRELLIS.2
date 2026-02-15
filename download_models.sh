#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PRIMARY_REPO="microsoft/TRELLIS.2-4B"
REVISION=""
TOKEN="${HF_TOKEN:-}"
CACHE_DIR=""
LOCAL_DIR="$ROOT_DIR/models"
DRY_RUN=false
QUIET=false
CORE_ONLY=false

usage() {
    cat <<'EOF'
Usage: ./download_models.sh [options]

Downloads TRELLIS model repos from Hugging Face using uvx + hf CLI.
Defaults to downloading all inference dependencies discovered from:
  - microsoft/TRELLIS.2-4B/pipeline.json
  - microsoft/TRELLIS.2-4B/texturing_pipeline.json

Options:
  --repo <repo_id>         Primary repo to inspect/download (default: microsoft/TRELLIS.2-4B)
  --revision <rev>         Revision for primary repo (branch/tag/commit)
  --token <hf_token>       Hugging Face token (or set HF_TOKEN)
  --cache-dir <path>       Custom HF cache directory
  --local-dir <path>       Download into local folder(s): <path>/<repo_id>/ (default: ./models)
  --core-only              Download only --repo (skip extra dependency repos)
  --dry-run                Show what would be downloaded
  --quiet                  Pass --quiet to hf download
  -h, --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)
            PRIMARY_REPO="$2"
            shift 2
            ;;
        --revision)
            REVISION="$2"
            shift 2
            ;;
        --token)
            TOKEN="$2"
            shift 2
            ;;
        --cache-dir)
            CACHE_DIR="$2"
            shift 2
            ;;
        --local-dir)
            LOCAL_DIR="$2"
            shift 2
            ;;
        --core-only)
            CORE_ONLY=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --quiet)
            QUIET=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if ! command -v uvx >/dev/null 2>&1; then
    echo "Error: uvx is required but not found in PATH." >&2
    exit 1
fi

export HF_HUB_CACHE="${HF_HUB_CACHE:-$ROOT_DIR/.hf/hub}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        echo "Error: Python is required but not found in PATH." >&2
        exit 1
    fi
fi

HF_CMD=(uvx hf)
if ! "${HF_CMD[@]}" --help >/dev/null 2>&1; then
    HF_CMD=(uvx --from huggingface_hub hf)
fi

hf_download() {
    local repo_id="$1"
    local revision="${2:-}"

    local -a cmd=("${HF_CMD[@]}" download "$repo_id" --repo-type model)
    if [[ -n "$revision" ]]; then
        cmd+=(--revision "$revision")
    fi
    if [[ -n "$TOKEN" ]]; then
        cmd+=(--token "$TOKEN")
    fi
    if [[ -n "$CACHE_DIR" ]]; then
        cmd+=(--cache-dir "$CACHE_DIR")
    fi
    if [[ -n "$LOCAL_DIR" ]]; then
        local target_dir="$LOCAL_DIR/$repo_id"
        mkdir -p "$target_dir"
        cmd+=(--local-dir "$target_dir")
    fi
    if [[ "$DRY_RUN" == true ]]; then
        cmd+=(--dry-run)
    fi
    if [[ "$QUIET" == true ]]; then
        cmd+=(--quiet)
    fi

    echo "+ ${cmd[*]}"
    if ! "${cmd[@]}"; then
        cat >&2 <<EOF
Error: download failed for $repo_id
If this repo is gated/private:
1) run: uvx hf auth login
2) open the model page and request/accept access terms
3) rerun this script (or pass --token)
EOF
        return 1
    fi
}

get_config_path() {
    local filename="$1"
    local -a cmd=("${HF_CMD[@]}" download "$PRIMARY_REPO" "$filename" --repo-type model --quiet)
    if [[ -n "$REVISION" ]]; then
        cmd+=(--revision "$REVISION")
    fi
    if [[ -n "$TOKEN" ]]; then
        cmd+=(--token "$TOKEN")
    fi
    if [[ -n "$CACHE_DIR" ]]; then
        cmd+=(--cache-dir "$CACHE_DIR")
    fi
    if [[ -n "$LOCAL_DIR" ]]; then
        local target_dir="$LOCAL_DIR/$PRIMARY_REPO"
        mkdir -p "$target_dir"
        cmd+=(--local-dir "$target_dir")
    fi
    "${cmd[@]}" | tail -n1
}

declare -a repos=("$PRIMARY_REPO")
if [[ "$CORE_ONLY" != true ]]; then
    pipeline_cfg="$(get_config_path "pipeline.json")"
    texturing_cfg="$(get_config_path "texturing_pipeline.json")"

    while IFS= read -r repo; do
        if [[ -n "$repo" ]]; then
            repos+=("$repo")
        fi
    done < <(
        "$PYTHON_BIN" - "$PRIMARY_REPO" "$pipeline_cfg" "$texturing_cfg" <<'PY'
import json
import sys

primary_repo = sys.argv[1]
config_paths = sys.argv[2:]

repos = {primary_repo}

for path in config_paths:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        continue

    args = data.get("args", {})
    for value in (args.get("models") or {}).values():
        if isinstance(value, str):
            parts = value.split("/")
            if len(parts) >= 3:
                repos.add(f"{parts[0]}/{parts[1]}")

    for key in ("image_cond_model", "rembg_model"):
        model_name = ((args.get(key) or {}).get("args") or {}).get("model_name")
        if isinstance(model_name, str):
            parts = model_name.split("/")
            if len(parts) >= 2:
                repos.add(f"{parts[0]}/{parts[1]}")

for repo in sorted(repos):
    print(repo)
PY
    )
fi

declare -A seen=()
declare -a unique_repos=()
for repo in "${repos[@]}"; do
    if [[ -z "${seen[$repo]+x}" ]]; then
        seen[$repo]=1
        unique_repos+=("$repo")
    fi
done

echo "Repos to download:"
for repo in "${unique_repos[@]}"; do
    echo "  - $repo"
done

for repo in "${unique_repos[@]}"; do
    if [[ "$repo" == "$PRIMARY_REPO" ]]; then
        hf_download "$repo" "$REVISION"
    else
        hf_download "$repo"
    fi
done

echo "Model download step completed."
