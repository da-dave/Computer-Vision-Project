#!/usr/bin/env bash
# Dataset and evaluation tools. First run creates the virtualenv and installs dependencies.
#   ./run.sh capture --person P01 --site forearm --light dim
#   ./run.sh label --labeler A
#   ./run.sh evaluate
#   ./run.sh stream-fps
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
command="${1:-}"
[ $# -gt 0 ] && shift
case "$command" in
  capture)    exec .venv/bin/python capture.py "$@" ;;
  label)      exec .venv/bin/python label.py "$@" ;;
  evaluate)   exec .venv/bin/python evaluate.py "$@" ;;
  stream-fps) exec .venv/bin/python measure_stream.py "$@" ;;
  test)       exec .venv/bin/python -m unittest discover -s tests -v ;;
  *) echo "Usage: ./run.sh capture|label|evaluate|stream-fps|test [options]  (add --help for a command's options)"; exit 1 ;;
esac
