"""Every global name the app uses must actually resolve.

WHY THIS EXISTS: splitting the single-file app into modules moved definitions out
from under their users. `import *` does not re-export underscore names, and one
constant (_FAILED_FILE_RE) ended up defined in a sibling but referenced only in
tomogration_app._log. Nothing failed at import time — the NameError fired at
RUNTIME, on the first log line, inside a Qt slot. Qt swallows slot exceptions to
stderr, so the visible symptom was "I press Run and nothing happens", with an
empty log, and no traceback anywhere the user could see.

py_compile cannot catch this (it is a runtime lookup). So this suite walks the AST
of every module, collects the global names each one READS, and checks each resolves
in that module's namespace — catching a whole class of split/refactor damage before
it reaches the VM.

    python3 tests/test_symbols.py
"""
import ast
import builtins
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE / "stub"))
sys.path.insert(0, str(REPO))

MODULES = ["tomogration_core", "tomogration_stages", "tomogration_project",
           "tomogration_jobs", "tomogration_app"]

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok  {name}")
    else:
        failed += 1
        print(f"FAIL  {name}{(' — ' + detail) if detail else ''}")


class GlobalReads(ast.NodeVisitor):
    """Names loaded at module scope or inside functions that aren't locals."""

    def __init__(self):
        self.reads = set()
        self.bound = set()

    def visit_FunctionDef(self, node):
        self.bound.add(node.name)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.bound.add(node.name)
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.reads.add(node.id)
        else:
            self.bound.add(node.id)

    def visit_arg(self, node):
        self.bound.add(node.arg)

    def visit_alias(self, node):
        self.bound.add((node.asname or node.name).split(".")[0])

    def visit_ExceptHandler(self, node):
        if node.name:
            self.bound.add(node.name)
        self.generic_visit(node)

    def visit_Global(self, node):
        self.bound.update(node.names)


def load(mod):
    path = REPO / f"{mod}.py"
    spec = importlib.util.spec_from_file_location(mod, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod] = m
    spec.loader.exec_module(m)
    return m


def main():
    builtin_names = set(dir(builtins))
    for mod in MODULES:
        path = REPO / f"{mod}.py"
        if not path.is_file():
            check(f"{mod} exists", False)
            continue
        try:
            m = load(mod)
        except Exception as e:                       # noqa: BLE001
            check(f"{mod} imports", False, repr(e))
            continue
        check(f"{mod} imports", True)

        tree = ast.parse(path.read_text())
        g = GlobalReads()
        g.visit(tree)
        ns = set(dir(m))
        missing = sorted(
            n for n in g.reads
            if n not in ns and n not in builtin_names and n not in g.bound
            and not n.startswith("__"))
        check(f"{mod}: all global names resolve",
              not missing, f"unresolved: {missing[:8]}")

    # the specific regression: the app must own every name its logger touches
    app = sys.modules["tomogration_app"]
    for n in ("_FAILED_FILE_RE", "_PROGRESS_RE"):
        check(f"app exposes {n}", hasattr(app, n))

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
