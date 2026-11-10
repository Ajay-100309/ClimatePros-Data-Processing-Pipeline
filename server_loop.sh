#!/bin/bash
# Autonomous processing loop: pull, process whatever work order is staged,
# push the results, repeat forever. Run THIS (not a one-shot process.py) under
# pm2, so a finished batch doesn't leave the pipeline idle waiting for a human
# to notice and restart it.
#
#   pm2 start server_loop.sh --name pipeline --interpreter bash
#
# Every iteration is a plain `git commit` of whatever changed — safe even if
# the batch was empty (nothing to commit, git exits non-zero, loop continues).
set -u
cd "$(dirname "$0")"
POLL_SECONDS=60

while true; do
  git pull --no-rebase -q || echo "$(date -u +%FT%TZ) git pull failed, will retry"

  if [ -f state/batch_current.json.gz ] || [ -f state/batch_current.json ]; then
    echo "$(date -u +%FT%TZ) staged batch found — processing..."
    venv/bin/python process.py
    git add -A
    if git commit -q -m "processed batch $(date -u +%Y%m%dT%H%M%SZ)"; then
      git push && echo "$(date -u +%FT%TZ) pushed results"
    else
      echo "$(date -u +%FT%TZ) nothing to commit after processing (unexpected)"
    fi
  else
    echo "$(date -u +%FT%TZ) no batch staged, waiting ${POLL_SECONDS}s..."
  fi
  sleep "$POLL_SECONDS"
done
