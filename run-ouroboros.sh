#!/usr/bin/env bash
# Starts the Ouroboros web UI and opens it in your browser (Linux counterpart of "Run Ouroboros.bat").
# Keep the terminal window open while you use it; close the window (or Ctrl+C) to quit.
cd "$(dirname "$0")"
# Double-clicked in the file manager: there's no terminal, so open one to run in.
if [ ! -t 1 ] && [ -z "$OUROBOROS_IN_TERM" ]; then
  export OUROBOROS_IN_TERM=1
  for term in x-terminal-emulator cosmic-term gnome-terminal konsole xterm; do
    command -v "$term" >/dev/null && exec "$term" -e "$PWD/run-ouroboros.sh" "$@"
  done
fi
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
"$PY" -m ouroboros serve "$@"
status=$?
if [ $status -ne 0 ] && [ -n "$OUROBOROS_IN_TERM" ]; then
  read -rp "Ouroboros exited with code $status. Press Enter to close."
fi
exit $status
