#!/usr/bin/env bash
# Source this file from the server checkout; no packages or assets are downloaded.
_chronogeom_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ ! -f "$HOME/.venvs/chronogeom/bin/activate" ]]; then
  printf 'Missing environment: %s\n' "$HOME/.venvs/chronogeom" >&2
  return 1 2>/dev/null || exit 1
fi
source "$HOME/.venvs/chronogeom/bin/activate"
export CHRONOGEOM_ROOT="$_chronogeom_root"
export CHRONOGEOM_DATA="$HOME/data/ChronoGeom"
export CHRONOGEOM_MODELS="$HOME/models/ChronoGeom"
export CHRONOGEOM_RUNS="$HOME/runs/ChronoGeom"
export HF_HOME="$CHRONOGEOM_MODELS/huggingface"
export MPLCONFIGDIR="$CHRONOGEOM_RUNS/matplotlib"
export PYTHONUNBUFFERED=1
cd "$CHRONOGEOM_ROOT" || return 1
unset _chronogeom_root
printf 'ChronoGeom server workspace: %s\nPython: %s\n' "$CHRONOGEOM_ROOT" "$(command -v python)"
