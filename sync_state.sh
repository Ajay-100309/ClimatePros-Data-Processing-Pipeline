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
# batch_current.json is the staged work order: fetch on one machine, sync, then
# process on another. Missing files are skipped (e.g. no batch in flight).
FILES=(state/embeddings.npy state/embeddings_index.json state/dispatch_meta.json
       state/notes_class.json state/batch_current.json)
mkdir -p state/batches
case "$MODE" in
  pull)
    for f in "${FILES[@]}"; do
      rsync -av "$REMOTE:$RPATH/$f" "$f" \
        || { [ $? -eq 23 ] && echo "  (skipped $f — not on remote)" || exit 1; }
    done
    rsync -av "$REMOTE:$RPATH/state/batches/" state/batches/
    ;;
  push)
    for f in "${FILES[@]}"; do
      if [ -f "$f" ]; then
        rsync -av "$f" "$REMOTE:$RPATH/$f"   # a failure here aborts loudly (set -e)
      else
        echo "  (skipped $f — not present locally)"
      fi
    done
    rsync -av state/batches/ "$REMOTE:$RPATH/state/batches/"
    ;;
  *) echo "unknown mode: $MODE (use pull|push)" >&2; exit 1 ;;
esac
echo "sync $MODE complete."
