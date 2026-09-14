#!/usr/bin/env bash
# Sync the bulk pipeline state that is deliberately NOT in git (GitHub's 100 MB
# blob limit; see CLAUDE.md "Live state"): the embedding cache pair, dispatch
# display metadata, and the batch archives. Everything else in state/ stays in git.
#
# Usage:
#   ./sync_state.sh pull user@host [remote_repo_path]   # bring remote state here
#   ./sync_state.sh push user@host [remote_repo_path]   # send local state there
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:?usage: sync_state.sh pull|push user@host [remote_repo_path]}"
REMOTE="${2:?usage: sync_state.sh pull|push user@host [remote_repo_path]}"
RPATH="${3:-techjays/ClimatePros-Data-Processing-Pipeline}"
FILES=(state/embeddings.npy state/embeddings_index.json state/dispatch_meta.json)
mkdir -p state/batches
case "$MODE" in
  pull)
    for f in "${FILES[@]}"; do rsync -av "$REMOTE:$RPATH/$f" "$f"; done
    rsync -av "$REMOTE:$RPATH/state/batches/" state/batches/
    ;;
  push)
    for f in "${FILES[@]}"; do rsync -av "$f" "$REMOTE:$RPATH/$f"; done
    rsync -av state/batches/ "$REMOTE:$RPATH/state/batches/"
    ;;
  *) echo "unknown mode: $MODE (use pull|push)" >&2; exit 1 ;;
esac
echo "sync $MODE complete."
