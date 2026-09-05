"""Canvas zoom: the maths, without Qt.

The canvas had exactly two zoom levels — Fit and 1:1 — and the wheel was
entirely consumed by horizontal panning, so a 72-job graph could only be seen
whole or not at all. These drive the REAL _CanvasView.zoom_to/zoom_by against a
fake transform, so the clamping, the anchor swap and the readout are checked
rather than assumed. The Qt stub cannot render, but it can record calls.

    python3 tests/test_canvas_zoom.py
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE / "stub"))
sys.path.insert(0, str(REPO))

passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok  {name}")
    else:
        failed += 1
        print(f"FAIL  {name}")


spec = importlib.util.spec_from_file_location("tomapp", REPO / "tomogration_app.py")
tomapp = importlib.util.module_from_spec(spec)
sys.modules["tomapp"] = tomapp
spec.loader.exec_module(tomapp)

V = tomapp._CanvasView


class _Anchors:
    """Real, distinguishable anchor constants.

    The Qt stub hands back a fresh permissive object on every attribute access,
    so the anchor the code picked could not be observed at all — the test would
    have passed whichever branch ran. Substituting named sentinels means the
    branch is genuinely checked."""
    AnchorUnderMouse = "AnchorUnderMouse"
    AnchorViewCenter = "AnchorViewCenter"


tomapp.QGraphicsView.ViewportAnchor = _Anchors


class FakeView:
    """Just enough QGraphicsView to run the real zoom methods."""

    def __init__(self, level=1.0):
        self.level = level
        self.anchors = []
        self.reported = []
        self._zoom_handler = self.reported.append

    # --- the bits zoom_to touches ---
    def transform(self):
        return type("T", (), {"m11": lambda _s, v=self.level: v})()

    def scale(self, sx, _sy):
        self.level *= sx

    def transformationAnchor(self):
        return "prev"

    def setTransformationAnchor(self, a):
        self.anchors.append(a)

    # the real implementations under test
    MIN_ZOOM = V.MIN_ZOOM
    MAX_ZOOM = V.MAX_ZOOM
    zoom_level = V.zoom_level
    zoom_to = V.zoom_to
    zoom_by = V.zoom_by


def main():
    v = FakeView()
    check("zoom starts at 1:1", abs(v.zoom_level() - 1.0) < 1e-9)

    v.zoom_by(1.25)
    check(f"zooming in scales up ({v.level:.2f})", abs(v.level - 1.25) < 1e-9)
    v.zoom_by(1 / 1.25)
    check("and back out returns exactly to 1:1", abs(v.level - 1.0) < 1e-9)

    # Clamping matters in BOTH directions: unbounded zoom-out silently reaches
    # a scale where the whole graph is one pixel and looks like an empty
    # canvas, which reads as a crash rather than a zoom.
    v = FakeView()
    for _ in range(60):
        v.zoom_by(1 / 1.25)
    check(f"cannot zoom out past the floor ({v.level:.3f})",
          abs(v.level - V.MIN_ZOOM) < 1e-9)
    for _ in range(60):
        v.zoom_by(1.25)
    check(f"nor in past the ceiling ({v.level:.2f})",
          abs(v.level - V.MAX_ZOOM) < 1e-9)
    check("and the floor is low enough for a 72-job graph", V.MIN_ZOOM <= 0.2)

    v = FakeView(0.5)
    check("zoom_to sets an ABSOLUTE level, not a relative one",
          abs(v.zoom_to(2.0) - 2.0) < 1e-9 and abs(v.level - 2.0) < 1e-9)
    check("it returns the level it actually reached, clamped",
          abs(FakeView().zoom_to(99.0) - V.MAX_ZOOM) < 1e-9)

    # Zooming with the wheel must keep the point under the cursor still;
    # anchoring to the viewport centre instead makes the graph swim away from
    # whatever you were pointing at.
    v = FakeView()
    v.zoom_by(1.25, anchor_mouse=True)
    check("wheel zoom anchors under the mouse",
          v.anchors[0] == "AnchorUnderMouse")
    check("and puts the previous anchor back afterwards",
          v.anchors[-1] == "prev")
    v = FakeView()
    v.zoom_by(1.25)
    check("button zoom anchors to the view centre instead",
          v.anchors[0] == "AnchorViewCenter")

    v = FakeView()
    v.zoom_by(1.25)
    v.zoom_by(1.25)
    check("every change is reported for the readout", len(v.reported) == 2)
    check("and the readout renders as a percentage",
          f"{v.reported[-1] * 100:.0f}%" == "156%")

    # A degenerate transform must not divide by zero. zoom_level() reads it
    # as 1:1, so zoom_to treats the view as unscaled rather than crashing.
    check("a degenerate transform is read as 1:1, not divided by",
          abs(FakeView(0.0).zoom_to(2.0) - 2.0) < 1e-9)

    src = (REPO / "tomogration_app.py").read_text()
    check("Ctrl+wheel is handled BEFORE the wheel-to-pan rules",
          src.index("ControlModifier | Qt.MetaModifier")
          < src.index("No horizontal delta and no Shift"))
    check("Ctrl+0 resets to 1:1", "Qt.Key_0" in src and "zoom_to(1.0)" in src)
    check("Ctrl+= zooms in, not just Ctrl+plus",
          "Qt.Key_Plus, Qt.Key_Equal" in src)
    check("Fit still reports the level it landed on",
          "self._show_zoom(self.view.zoom_level())" in src)
    check("and the hint tells the user zoom exists",
          "Ctrl+wheel zooms" in src)

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
