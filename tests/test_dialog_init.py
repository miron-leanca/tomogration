"""Dialogs must not connect a signal before the widgets its handler reads exist.

WHY THIS IS A STATIC TEST. Connecting a handler and THEN creating the widget it
updates works right up until the first setCheckState/addItem in the constructor
fires that handler — and then every row raises AttributeError. It has now
happened twice: VariantsDialog ('no attribute build_btn', which made Queue
variants unopenable for every stage) and ViewerPickDialog ('no attribute
count', forty tracebacks per open).

The obvious test — construct the dialog and see if it throws — CANNOT catch it:
the Qt stub's connect() is a no-op, so no signal fires and construction always
succeeds. It passes on code that is broken in the real app, which is worse than
no test. So this reads the source instead: any attribute assigned AFTER the
first .connect() in __init__, and read by another method of the same class, is
the bug.

    python3 tests/test_dialog_init.py
"""
import ast
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok  {name}")
    else:
        failed += 1
        print(f"FAIL  {name}")


def self_attrs_assigned(node):
    """{name: lineno} for `self.x = ...` inside a function."""
    out = {}
    for n in ast.walk(node):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                        and t.value.id == "self"):
                    out.setdefault(t.attr, n.lineno)
    return out


def self_attrs_read(node):
    """Attributes READ off self anywhere in a function (not assigned)."""
    assigned = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) \
                        and t.value.id == "self":
                    assigned.add(t.attr)
    out = set()
    for n in ast.walk(node):
        if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                and n.value.id == "self" and isinstance(n.ctx, ast.Load)):
            out.add(n.attr)
    return out - assigned


# Signals that fire when the CODE changes a widget (populating a list, ticking
# a box). A button's clicked/triggered cannot fire while __init__ runs, so
# connecting those early is fine and must not be flagged.
PROGRAMMATIC = {
    "itemChanged", "currentRowChanged", "currentItemChanged", "textChanged",
    "stateChanged", "valueChanged", "currentIndexChanged", "toggled",
    "currentTextChanged", "itemSelectionChanged", "itemExpanded",
}


def methods_of(cls):
    return {f.name: f for f in cls.body if isinstance(f, ast.FunctionDef)}


def called_self_methods(node):
    """[(method, lineno)] for self.foo(...) calls, plus handlers connected to a
    signal that populating a widget would fire."""
    out = []
    for n in ast.walk(node):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                and f.value.id == "self"):
            out.append((f.attr, n.lineno))
        elif isinstance(f, ast.Attribute) and f.attr == "connect":
            sig = f.value.attr if isinstance(f.value, ast.Attribute) else ""
            if sig in PROGRAMMATIC:
                for a in n.args:
                    if (isinstance(a, ast.Attribute)
                            and isinstance(a.value, ast.Name)
                            and a.value.id == "self"):
                        out.append((a.attr, n.lineno))
    return out


def reachable_reads(cls_methods, name, seen=None):
    """Attributes read by a method and everything it calls, transitively."""
    seen = seen if seen is not None else set()
    if name in seen or name not in cls_methods:
        return set()
    seen.add(name)
    fn = cls_methods[name]
    reads = self_attrs_read(fn)
    for callee, _ln in called_self_methods(fn):
        reads |= reachable_reads(cls_methods, callee, seen)
    return reads


def audit(path):
    """[(class, attr, used_line, assigned_line)] — an attribute READ by code
    reachable while __init__ is still running, but assigned only later."""
    tree = ast.parse(Path(path).read_text())
    problems = []
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        ms = methods_of(cls)
        init = ms.get("__init__")
        if init is None:
            continue
        assigned = self_attrs_assigned(init)
        for callee, line in called_self_methods(init):
            for attr in reachable_reads(ms, callee):
                at = assigned.get(attr)
                if at is not None and at > line:
                    problems.append((cls.name, attr, line, at))
    return sorted(set(problems))


# Pre-existing sites, none of which currently FIRE — the connect happens before
# the attribute exists, but nothing between the two changes the widget, so no
# handler runs during construction. They predate this test and belong to code
# this session did not write; ratcheting rather than churning them means a NEW
# one still fails the build, which is the point. Shrink this list when touching
# those dialogs for other reasons.
BASELINE = {
    ("AreTomoVersions", "text"),
    ("GroupsDialog", "member_header"),
    ("GroupsDialog", "series_list"),
    ("JobCanvas", "palette"),
    ("PositionsInspector", "del_extras"),
    ("PositionsInspector", "del_sel"),
    ("PositionsInspector", "detail_header"),
    ("PositionsInspector", "open_frames"),
    ("PositionsInspector", "orphan_label"),
    ("PositionsInspector", "tree"),
    ("Tomogration", "canvas"),
    ("_JobPalette", "tree"),
}


def main():
    app = REPO / "tomogration_app.py"
    problems = audit(app)
    fresh = [p for p in problems if (p[0], p[1]) not in BASELINE]
    detail = "; ".join(f"{c}.{a} reachable at line {ul}, assigned line {al}"
                       for c, a, ul, al in fresh)
    check(f"no NEW dialog reads a widget its __init__ has not made yet "
          f"[{detail or 'clean'}]", not fresh)
    stale = BASELINE - {(p[0], p[1]) for p in problems}
    check(f"the baseline has no stale entries [{sorted(stale) or 'clean'}]",
          not stale)

    # The test must actually be able to FAIL — the constructor version of this
    # could not, because the stub never fires a signal.
    broken = '''
class Fake:
    def __init__(self):
        self.tree = Tree()
        self.tree.itemChanged.connect(self._cascade)
        self.count = Label()

    def _cascade(self):
        self.count.setText("x")
'''
    tmp = HERE / "_broken_dialog_probe.py"
    tmp.write_text(broken)
    try:
        found = audit(tmp)
        check("and it CATCHES the real pattern when reintroduced",
              found and found[0][0] == "Fake" and found[0][1] == "count")
    finally:
        tmp.unlink()

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
