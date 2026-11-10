#!/bin/bash
# Autonomous fetch loop (Mac side — the only machine with VPN access to
# FieldJetXStg). Whenever no batch is in flight, stages the next parts-only
# chunk, commits + pushes it, then waits for the server to process and push
# results back before staging the next one. Exits cleanly once supply runs out.
#
# Must run on THIS Mac, connected to VPN, for the whole duration — that is the
# one constraint that can't be automated away (fetching is Mac-only). Run it
# directly in a Terminal (not inside Claude Code, so it survives independently
# of any session), ideally guarded against sleep:
#
#   caffeinate -s ./mac_loop.sh 2>&1 | tee mac_loop.log
set -u
cd "$(dirname "$0")"
CHUNK="${1:-25000}"
POLL_SECONDS=300

batch_file() {
  [ -f state/batch_current.json.gz ] && echo state/batch_current.json.gz && return
  [ -f state/batch_current.json ] && echo state/batch_current.json && return
  echo ""
}

dispatch_count() {
  venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from pipelib import stage_fetch
b = stage_fetch.load_batch()
print(len(b['dispatches']) if b else 0)
"
}

while true; do
  git pull --no-rebase -q || echo "$(date -u +%FT%TZ) git pull failed, will retry"

  if [ -n "$(batch_file)" ]; then
    echo "$(date -u +%FT%TZ) batch still in flight on the server, waiting ${POLL_SECONDS}s..."
    sleep "$POLL_SECONDS"
    continue
  fi

  echo "$(date -u +%FT%TZ) staging next parts-only chunk (up to $CHUNK)..."
  venv/bin/python fetch.py --count "$CHUNK" --parts-only

  N=$(dispatch_count)
  if [ "$N" -eq 0 ]; then
    echo "$(date -u +%FT%TZ) nothing left to fetch — parts-only supply exhausted. Done."
    rm -f "$(batch_file)"
    break
  fi

  echo "$(date -u +%FT%TZ) staged $N dispatches — committing + pushing"
  git add -A
  git commit -q -m "stage next parts-only chunk ($N dispatches, $(date -u +%Y%m%dT%H%M%SZ))"
  git push
  sleep "$POLL_SECONDS"
done
