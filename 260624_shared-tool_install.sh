#!/bin/bash
# install.sh — one-time setup for Tomogration on this Linux machine.
#
# Run ONCE on the VM, from inside this folder:
#   bash install.sh
#
# It (1) builds a self-contained virtual environment with PySide6 — needed
# because Debian/Ubuntu block system pip (PEP 668) — and (2) registers the
# desktop launcher in your menu and on your Desktop.
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
VENV="$DIR/.venv"

# ---------------------------------------------------------------------------
# 1. Python environment (isolated venv; never touches system site-packages)
# ---------------------------------------------------------------------------
# The venv lives next to this folder, which on a cluster is usually shared
# storage (e.g. ceph). A venv built on one machine references that machine's
# system python by absolute path, so on a DIFFERENT VM it can be stale/broken —
# validate it actually runs here, and rebuild if not.
if [ -x "$VENV/bin/python" ] && ! "$VENV/bin/python" -c "import sys" >/dev/null 2>&1; then
    echo "Existing .venv does not run on this machine; rebuilding it…"
    rm -rf "$VENV"
fi

if [ ! -x "$VENV/bin/python" ]; then
    echo "Creating virtual environment -> $VENV"
    if ! "$PYTHON" -m venv "$VENV" 2>/tmp/tomogration_venv.err; then
        echo >&2
        echo "ERROR: could not create the venv. The venv module is probably missing." >&2
        echo "Install it, then re-run this script:" >&2
        echo "    sudo apt install python3-venv python3-full" >&2
        echo "--- details ---" >&2
        cat /tmp/tomogration_venv.err >&2
        exit 1
    fi
fi

# Only download PySide6 if it isn't already importable (instant on a reused venv).
if "$VENV/bin/python" -c "import PySide6" >/dev/null 2>&1; then
    echo "PySide6 already present in the venv."
else
    echo "Installing PySide6 into the venv (first run downloads ~100 MB)…"
    "$VENV/bin/python" -m pip install --upgrade pip >/dev/null
    "$VENV/bin/python" -m pip install PySide6
fi

# ---------------------------------------------------------------------------
# 2. Executables
# ---------------------------------------------------------------------------
chmod +x "$DIR/tomogration.sh" 2>/dev/null || true
chmod +x "$DIR"/ml_*.sh "$DIR"/fetch_xcb_libs.sh 2>/dev/null || true

# ---------------------------------------------------------------------------
# 2b. Qt xcb dependency (libxcb-cursor). Needed since Qt 6.5 and NOT bundled in
#     the PySide6 wheel. If it's absent system-wide, fetch a local copy without
#     root (tomogration.sh adds ./libs to LD_LIBRARY_PATH automatically).
# ---------------------------------------------------------------------------
if ! ldconfig -p 2>/dev/null | grep -qi xcb-cursor && [ ! -d "$DIR/libs" ]; then
    echo "System libxcb-cursor not found; fetching a local copy (no root)…"
    bash "$DIR/fetch_xcb_libs.sh" || \
        echo "WARN: could not fetch libxcb-cursor automatically — see README." >&2
fi

# ---------------------------------------------------------------------------
# 3. Desktop launcher (absolute paths baked to THIS folder)
# ---------------------------------------------------------------------------
DESKTOP_FILE="$HOME/.local/share/applications/tomogration.desktop"
mkdir -p "$(dirname "$DESKTOP_FILE")"
cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Tomogration
GenericName=Cryo-ET Pipeline Controller
Comment=WarpTools / AreTomo2 cryo-ET processing pipeline GUI
Exec=bash "$DIR/tomogration.sh"
Icon=$DIR/Tomogration-icon.png
Terminal=false
Categories=Science;Education;
StartupNotify=true
EOF
chmod +x "$DESKTOP_FILE"

# Drop a copy on the Desktop too, if one exists.
DESK="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
if [ -d "$DESK" ]; then
    rm -f "$DESK/tomogration.desktop"   # clear any stale/dangling symlink first
    cp "$DESKTOP_FILE" "$DESK/tomogration.desktop"
    chmod +x "$DESK/tomogration.desktop"
    gio set "$DESK/tomogration.desktop" metadata::trusted true 2>/dev/null || true
    echo "Placed launcher on Desktop: $DESK"
fi

update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 4. Self-test: confirm the venv can import PySide6 on THIS machine.
# ---------------------------------------------------------------------------
if "$VENV/bin/python" -c "import PySide6" >/dev/null 2>&1; then
    SELFTEST="OK"
else
    SELFTEST="FAILED — PySide6 not importable; re-run this script."
fi

echo
echo "Setup complete (PySide6 self-test: $SELFTEST)."
echo "  • venv:     $VENV"
echo "  • launcher: $DESKTOP_FILE  (Applications menu -> Science -> Tomogration)"
[ -d "$DESK" ] && echo "  • desktop:  $DESK/tomogration.desktop"
echo
echo "Launch it any of these ways:"
echo "  • Applications menu  ->  Science  ->  Tomogration   (most reliable)"
echo "  • Terminal:          bash \"$DIR/tomogration.sh\""
echo "  • The Desktop icon. If a double-click does nothing the FIRST time, XFCE"
echo "    needs you to trust it once: right-click the icon -> \"Allow This File"
echo "    to Run\" (or Properties -> Permissions -> tick \"Allow executing\")."
echo
echo "NOTE: home (~) is local to each VM, so run this install.sh once per machine."
echo "The .venv on shared storage is reused, so it only takes a moment after the"
echo "first machine."
