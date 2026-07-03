#!/bin/bash
# fetch_xcb_libs.sh — NO-ROOT fix for the Qt "xcb" platform plugin error
# ("libxcb-cursor0 is needed ...").
#
# Since Qt 6.5 the xcb (X11) plugin needs libxcb-cursor, a SYSTEM library the
# PySide6 wheel does not bundle. With no sudo, we download the .deb and unpack
# it into ./libs (no root needed). tomogration.sh adds ./libs to LD_LIBRARY_PATH
# automatically, so the app — and the desktop icon — then start normally.
#
#   bash fetch_xcb_libs.sh
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBS="$DIR/libs"
mkdir -p "$LIBS"

# libxcb-cursor0 is the one that's missing; its companions are almost always
# already present on a desktop machine, but we list common ones harmlessly.
PKGS="libxcb-cursor0"

if ! command -v apt-get >/dev/null 2>&1; then
    echo "apt-get not found. Download libxcb-cursor0_*.deb manually from" >&2
    echo "https://packages.ubuntu.com (or your distro), then run:" >&2
    echo "    dpkg -x libxcb-cursor0_*.deb \"$LIBS\"" >&2
    exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cd "$TMP"

echo "Downloading (no root): $PKGS"
if ! apt-get download $PKGS 2>/tmp/tomogration_aptdl.err; then
    echo "ERROR: 'apt-get download $PKGS' failed." >&2
    echo "Your apt index may be stale or offline. Details:" >&2
    cat /tmp/tomogration_aptdl.err >&2
    echo >&2
    echo "Manual fallback: grab libxcb-cursor0_*.deb from https://packages.ubuntu.com" >&2
    echo "and run:  dpkg -x libxcb-cursor0_*.deb \"$LIBS\"" >&2
    exit 1
fi

for deb in *.deb; do
    echo "Unpacking $deb"
    dpkg -x "$deb" "$LIBS"
done

echo
echo "Done. Local libraries unpacked into: $LIBS"
echo "Launch with:  bash \"$DIR/tomogration.sh\""
