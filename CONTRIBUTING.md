# Contributing to Tomogration

Thanks for helping improve Tomogration. It's a lab tool that grows by use — bug
reports and small, focused pull requests are very welcome.

## Ground rules that keep the app stable

Tomogration is **developed on macOS but only ever runs on a Linux GPU
workstation**, so a few conventions exist to keep it working without a display or
GPU on the dev machine:

- **Syntax-check, don't launch, on non-Linux.**
  `python3 -m py_compile tomogration_*.py ml_*.py` must pass. Do not
  `pip install PySide6` or run the GUI on a Mac — the deploy target is Linux
  (XFCE/X11).
- **`bash -n script.sh`** must pass for every shell script you touch. The `ml_*`
  scripts use GNU userland (`sed -i` GNU syntax, etc.) — they run on Linux.
- **Keep companion scripts next to `tomogration_app.py`.** The app resolves them by
  absolute path relative to `__file__` (`_pkg_script()`), so they must ship together.
- **Never commit data.** No `.eer/.mrc/.mdoc/.tomostar/.settings`, no
  `warp_tiltseries/`, `frames/`, `aretomo_output*/`, etc. (`.gitignore` guards these).

## Architecture in one minute

The app is split across sibling modules with a strict import DAG
(`core → stages → jobs`, with `project` depending only on `core`);
`tomogration_app.py` imports all of them and holds the Qt window. All the
modules must ship to the VM together.

- **`tomogration_stages.py`** — **`STAGES`**, a data-driven list of pipeline
  steps. Each is a dict: `id`, `label`, `group`, `base` command, and a `params`
  list. Adding a step is usually just adding a dict — no new UI code. Also here:
  **`build_command(spec, values, warp_cmd, group_inputs)`**, a *pure* function
  (no Qt) that assembles the shell command from a step + its parameter values —
  the single source of truth the editable command box is seeded from.
- **Parameter kinds** (`param["kind"]`): `text`, `env`, `env_int` (→ spin box),
  `check`, `choice`, `module`. `env`/`env_int` become a `VAR=value` prefix;
  `flag=None` params are positional; a step's `fixed_env` bakes in constants
  (e.g. `MA_MODE=infer`).
- **`tomogration_project.py`** — `ProjectState`, all filesystem logic (mdoc
  repair, versioned AreTomo folders, exclusions, history). Framework-agnostic.
- **`tomogration_jobs.py`** — the job model: the `.tomogration_jobs.json` store,
  the canvas DAG layout, and discovery of work done outside the app. The queue
  is just jobs with `status="queued"`.
- **`tomogration_app.py`** — the PySide6 view layer over all of the above.
- **Companion `ml_*` scripts** — the heavy lifting (rename, remake mdocs, AreTomo
  farm, miss-alignment, the RELION/M helpers). The GUI just builds their command
  line.

## Testing without a GPU or display

There is a real test suite in `tests/` — each `test_*.py` is a standalone script
that stubs PySide6 (`tests/stub`), loads the app by path, and exercises the pure
logic. It runs anywhere Python does:

```bash
python3 tests/run_all.py     # all suites; exit code = number of failing suites
```

A change should keep the suite green, and add cases for logic it touches (the
existing suites show the `check("name", cond)` style). At minimum:

```bash
python3 -m py_compile tomogration_*.py ml_*.py
bash -n ml_*.sh install.sh tomogration.sh
python3 tests/run_all.py
```

## Submitting changes

1. Fork, branch (`fix/short-description`), make the change.
2. Keep new code in the style of the surrounding code (comment density, naming).
3. If you add a pipeline step, add its docs to `tomogration_docs.json` too.
4. Open a pull request describing *what* broke / improved and *how* you verified it
   (paste the `py_compile` / `bash -n` output, or a screenshot if it's UI).

## Reporting bugs

Open an issue with: what you ran (the exact command from the command box helps),
what happened (paste the terminal/log output), and the tool versions
(`WarpTools --version`, AreTomo build, miss-alignment commit). Cryo-ET failures are
often environment-specific (CUDA driver vs runtime, setgid binaries, module stubs) —
the more of that context, the faster it's fixed.
