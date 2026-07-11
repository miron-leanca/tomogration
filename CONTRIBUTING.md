# Contributing to Tomogration

Thanks for helping improve Tomogration. It's a lab tool that grows by use — bug
reports and small, focused pull requests are very welcome.

## Ground rules that keep the app stable

Tomogration is **developed on macOS but only ever runs on a Linux GPU
workstation**, so a few conventions exist to keep it working without a display or
GPU on the dev machine:

- **Syntax-check, don't launch, on non-Linux.** `python3 -m py_compile tomogration_app.py`
  must pass. Do not `pip install PySide6` or run the GUI on a Mac — the deploy
  target is Linux (XFCE/X11).
- **`bash -n script.sh`** must pass for every shell script you touch. The `ml_*`
  scripts use GNU userland (`sed -i` GNU syntax, etc.) — they run on Linux.
- **Keep companion scripts next to `tomogration_app.py`.** The app resolves them by
  absolute path relative to `__file__` (`_pkg_script()`), so they must ship together.
- **Never commit data.** No `.eer/.mrc/.mdoc/.tomostar/.settings`, no
  `warp_tiltseries/`, `frames/`, `aretomo_output*/`, etc. (`.gitignore` guards these).

## Architecture in one minute

- **`STAGES`** — a data-driven list of pipeline steps. Each is a dict: `id`, `label`,
  `group`, `base` command, and a `params` list. Adding a step is usually just adding
  a dict — no new UI code.
- **`build_command(spec, values, warp_cmd, group_inputs)`** — a *pure* function
  (no Qt) that assembles the shell command from a step + its parameter values. It is
  the single source of truth the editable command box is seeded from. It's the
  easiest thing to unit-test.
- **Parameter kinds** (`param["kind"]`): `text`, `env`, `env_int` (→ spin box),
  `check`, `choice`. `env`/`env_int` become a `VAR=value` prefix; `flag=None` params
  are positional; a step's `fixed_env` bakes in constants (e.g. `MA_MODE=infer`).
- **`ProjectState`** — all filesystem logic (mdoc repair, versioned AreTomo folders,
  exclusions, history). Framework-agnostic; also easy to test.
- **Companion `ml_*` scripts** — the heavy lifting (rename, remake mdocs, AreTomo
  farm, miss-alignment). The GUI just builds their command line.

## Testing without a GPU or display

There's a stubbed-Qt harness pattern used throughout development: fake the three
`PySide6.Qt*` modules with a permissive stub, then `import tomogration_app`,
construct `Tomogration(tmpdir)`, and exercise `build_command`, `ProjectState`,
and the stage forms. See the commit history / `scratchpad` examples. At minimum,
a change should pass:

```bash
python3 -m py_compile tomogration_app.py
bash -n ml_*.sh install.sh tomogration.sh
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
