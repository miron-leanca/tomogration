#!/bin/bash
# tomogration.sh — launcher for Tomogration (cryo-ET pipeline controller).
#
# Self-locating AND self-healing: run it from anywhere, or double-click the
# desktop icon. If the bundled virtual environment (.venv) or PySide6 is missing
# on THIS machine, it runs the one-time setup automatically, so a fresh VM "just
# works" without having to remember install.sh.
#   bash tomogration.sh
#   PYTHON=python3.11 bash tomogration.sh    # override the interpreter
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

pick_python() {
    if [ -x "$VENV/bin/python" ]; then printf '%s' "$VENV/bin/python";
    else printf '%s' "${PYTHON:-python3}"; fi
}

# No-root xcb fix: if fetch_xcb_libs.sh unpacked libs into ./libs, expose them
# so Qt's xcb plugin can find libxcb-cursor without a system install. This MUST
# run AFTER the self-heal below — on a first run install.sh fetches ./libs as
# part of setup, so setting LD_LIBRARY_PATH before that would miss it (and the
# xcb plugin then fails to load: "libxcb-cursor0 is needed").
add_local_libs() {
    if [ -d "$SCRIPT_DIR/libs" ]; then
        for d in "$SCRIPT_DIR"/libs/usr/lib/*/ "$SCRIPT_DIR"/libs/usr/lib/; do
            [ -d "$d" ] && LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH}"
        done
        export LD_LIBRARY_PATH
    fi
}

PYTHON="$(pick_python)"

# Self-heal: no working PySide6 on this machine -> run the one-time setup, which
# (re)builds the venv, installs PySide6, and fetches ./libs, then re-pick the
# interpreter.
if ! "$PYTHON" -c "import PySide6" >/dev/null 2>&1; then
    echo "Tomogration: first-time setup on this machine (building the Python"
    echo "environment; the first run downloads PySide6, ~100 MB)…"
    if [ -f "$SCRIPT_DIR/install.sh" ]; then
        bash "$SCRIPT_DIR/install.sh" || {
            echo "ERROR: setup failed — see the messages above." >&2; exit 1; }
    fi
    PYTHON="$(pick_python)"
fi

if ! "$PYTHON" -c "import PySide6" >/dev/null 2>&1; then
    echo "ERROR: PySide6 is still not available for '$PYTHON'." >&2
    echo "Run the setup by hand:   bash \"$SCRIPT_DIR/install.sh\"" >&2
    exit 1
fi

add_local_libs      # after setup, so a first run picks up the freshly-fetched ./libs
exec "$PYTHON" "$SCRIPT_DIR/tomogration_app.py" "$@"
