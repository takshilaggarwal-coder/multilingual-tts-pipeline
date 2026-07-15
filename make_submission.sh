#!/bin/zsh
# Package the Track-A submission: code + docs + generated audio + results,
# EXCLUDING venvs, model caches, and the listening-kit answer key.
# Usage: zsh make_submission.sh   ->  ../infinia_voice_submission.zip
set -e
cd "$(dirname "$0")"
NAME="infinia_voice_submission"
ZIP="../${NAME}.zip"
rm -f "$ZIP"

zip -r "$ZIP" . \
  -x 'envs/*' \
  -x '*/__pycache__/*' -x '*.pyc' \
  -x '.git/*' -x '.claude/*' \
  -x 'hf_cache/*' \
  -x 'eval/listening_test/kit/key.json' \
  -x 'results/*.log' \
  -x '.DS_Store' -x '*/.DS_Store' \
  > /dev/null

echo "built $ZIP"
du -h "$ZIP"
echo "top-level contents:"
unzip -l "$ZIP" | awk '{print $4}' | grep -E '^[^/]+/?$' | sort -u | head -40
echo
echo "sanity: no venv / cache / key leaked ->"
unzip -l "$ZIP" | grep -E 'envs/|hf_cache/|key.json|\.pyc' && echo "  LEAK DETECTED" || echo "  clean"
