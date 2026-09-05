"""New cards land in their stage's row, rightmost, and the view snaps to them.

Adopting a RELION selection (or building any job) put the card at the far right
of its stage's row — routinely off-screen on a graph this wide. The card
appeared, just not anywhere the user was looking.

    python3 tests/test_canvas_reveal.py
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


spec = importlib.util.spec_from_file_location("tomjobs", REPO / "tomogration_jobs.py")
J = importlib.util.module_from_spec(spec)
sys.modules["tomjobs"] = J
spec.loader.exec_module(J)

aspec = importlib.util.spec_from_file_location("tomapp", REPO / "tomogration_app.py")
A = importlib.util.module_from_spec(aspec)
aspec.loader.exec_module(A)


class RealRect:
    """A real rectangle for QRectF.

    The Qt stub returns a permissive object whose .x is not a number, so the
    visibility comparison could not run at all — the test would have exercised
    nothing. Substituting a real one means the geometry is genuinely checked."""

    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h


A.QRectF = RealRect


def node(i, **kw):
    n = {"id": i, "x": 0, "y": 0, "w": 240, "h": 90}
    n.update(kw)
    return n


class FakeView:
    """Enough QGraphicsView to run the real JobCanvas.reveal."""

    def __init__(self, seen=(0, 0, 800, 600)):
        self._seen = seen
        self.centred = []

    def mapToScene(self, _rect):
        x, y, w, h = self._seen
        return type("P", (), {"boundingRect": lambda _s: FakeRect(x, y, w, h)})()

    def viewport(self):
        return type("V", (), {"rect": lambda _s: None})()

    def centerOn(self, x, y):
        self.centred.append((x, y))


class FakeRect:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    def contains(self, r):
        return (r.x >= self.x and r.y >= self.y
                and r.x + r.w <= self.x + self.w
                and r.y + r.h <= self.y + self.h)


class FakeCanvas:
    _node_index = None
    reveal = A.JobCanvas.reveal

    def __init__(self, view):
        self.view = view


def main():
    # ---- which card is new --------------------------------------------------
    check("the first paint snaps to nothing — every card is new then",
          J.newly_added(None, {"J1": node("J1")}) is None)
    check("a repaint that added nothing snaps to nothing",
          J.newly_added({"J1": node("J1")}, {"J1": node("J1")}) is None)
    check("a newly created job is the one to show",
          J.newly_added({"J1": node("J1")},
                        {"J1": node("J1"), "J2": node("J2")}) == "J2")
    check("with several at once, the newest by job number wins",
          J.newly_added({"J1": node("J1")},
                        {"J1": node("J1"), "J9": node("J9"),
                         "J10": node("J10"), "J2": node("J2")}) == "J10")

    # Ghosts come and go as stages gain their first job; orphans are found on
    # disk by a TIMER, so snapping to one would yank the view mid-work.
    check("a ghost appearing does not move the view",
          J.newly_added({}, {"g": node("g", is_ghost=True)}) is None)
    check("nor does a stage template",
          J.newly_added({}, {"t": node("t", is_template=True)}) is None)
    check("nor an orphan discovered on disk",
          J.newly_added({}, {"disk:relion4_result:job009":
                             node("disk:relion4_result:job009",
                                  is_orphan=True)}) is None)
    check("but ADOPTING one, which creates a real job, does",
          J.newly_added({"disk:x": node("disk:x", is_orphan=True)},
                        {"disk:x": node("disk:x", is_orphan=True),
                         "J42": node("J42")}) == "J42")

    # ---- and the scroll itself ---------------------------------------------
    v = FakeView(seen=(0, 0, 800, 600))
    c = FakeCanvas(v)
    c._node_index = {"J1": node("J1", x=100, y=100)}
    check("a card already on screen does not scroll the canvas",
          c.reveal("J1") is False and not v.centred)

    c._node_index = {"J9": node("J9", x=5000, y=300)}
    check("one off to the right does", c.reveal("J9") is True)
    check("and it is centred on the card, not its corner",
          v.centred[-1] == (5000 + 240 / 2, 300 + 90 / 2))
    c2 = FakeCanvas(FakeView())
    c2._node_index = {"J1": node("J1", x=100, y=100)}
    check("force= scrolls even to a card already in view",
          c2.reveal("J1", force=True) is True and c2.view.centred)
    check("revealing a card that is not there is harmless",
          c.reveal("J404") is False)

    # ---- the row/column layout the request depends on ----------------------
    store = {"jobs": {}}
    for i, sid in ((1, "mb_segment"), (2, "mb_segment"), (10, "mb_segment"),
                   (3, "mb_components")):
        store["jobs"][f"J{i}"] = {"id": f"J{i}", "stage_id": sid,
                                  "status": "completed", "params": {},
                                  "inputs": {}, "label": sid}
    nodes, _edges = J.canvas_layout(store)
    seg = [n for n in nodes if n.get("stage_id") == "mb_segment"
           and not n.get("is_ghost")]
    check("every job of a stage shares one row",
          len({n["row"] for n in seg}) == 1)
    check("a different stage gets a different row",
          {n["row"] for n in seg} != {n["row"] for n in nodes
                                      if n.get("stage_id") == "mb_components"
                                      and not n.get("is_ghost")})
    order = [n["id"] for n in sorted(seg, key=lambda n: n["col"])]
    check(f"and they run oldest to newest left to right {order}",
          order == ["J1", "J2", "J10"])
    check("so the newest job is the rightmost card in its row",
          max(seg, key=lambda n: n["x"])["id"] == "J10")

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
