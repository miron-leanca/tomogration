"""Oversampled surface picking (A4) — the geometry, and the column that cannot
be recovered later.

THE POINT of these tests:

  * THETA, the angle between each normal and the wedge axis, is written per
    site and is correct. Detection efficiency varies with it, so a spatial
    statistic computed without it is biased by an amount nobody can reconstruct
    after the fact.
  * The Euler angles really do put the reference Z along the normal, and PSI is
    declared undetermined rather than quietly asserted.
  * Spacing means what it says: sites scale with surface AREA, so a virion
    twice the radius gets ~4x the sites, not 2x.

    python3 tests/test_picking.py
"""
import importlib.util
import sys
import subprocess
import struct
import tempfile
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


try:
    import numpy as np
except ImportError:                      # pragma: no cover
    print("numpy not installed — skipping A4 picking tests")
    print("\n0 passed, 0 failed")
    sys.exit(0)


def load(mod):
    spec = importlib.util.spec_from_file_location(mod, REPO / f"{mod}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod] = m
    spec.loader.exec_module(m)
    return m


P = load("ml_pick_surfaces")


def main():
    # ---- theta: the column that cannot be rebuilt later --------------------
    check("a normal along the wedge axis is theta 0 (worst detected)",
          abs(P.theta_to_wedge([0, 0, 1]) - 0.0) < 1e-6)
    check("a normal across it is theta 90 (best detected)",
          abs(P.theta_to_wedge([1, 0, 0]) - 90.0) < 1e-6)
    check("the opposite pole is 180, not folded to 0 — the caller decides",
          abs(P.theta_to_wedge([0, 0, -1]) - 180.0) < 1e-6)
    check("theta is scale-invariant (normals need not arrive normalised)",
          abs(P.theta_to_wedge([0, 0, 7]) - 0.0) < 1e-6)
    check("45 degrees comes out as 45",
          abs(P.theta_to_wedge([0, 1, 1]) - 45.0) < 1e-6)

    # ---- Euler angles ------------------------------------------------------
    # Rotating the reference Z by (rot, tilt) must land on the normal. Check by
    # reconstructing the direction from the angles.
    for n in ([0, 0, 1], [1, 0, 0], [0, 1, 0], [0.3, -0.5, 0.81]):
        v = np.array(n, dtype=float)
        v /= np.linalg.norm(v)
        rot, tilt, psi = P.euler_from_normal(v)
        t, r = np.deg2rad(tilt), np.deg2rad(rot)
        back = np.array([np.sin(t) * np.cos(r), np.sin(t) * np.sin(r), np.cos(t)])
        check(f"euler round-trips the normal {np.round(v, 2)}",
              np.allclose(back, v, atol=1e-6))
    check("psi is 0 — a surface normal does not determine it",
          P.euler_from_normal([0.3, -0.5, 0.81])[2] == 0.0)

    # ---- spacing scales with AREA -----------------------------------------
    n1 = P.n_sites(400.0, 50.0)
    n2 = P.n_sites(800.0, 50.0)
    check(f"twice the radius gives ~4x the sites ({n1} -> {n2})",
          3.8 < n2 / n1 < 4.2)
    check("halving the spacing also gives ~4x",
          3.8 < P.n_sites(400.0, 25.0) / n1 < 4.2)
    check("a tiny sphere still gets an even lattice, not 3 clustered points",
          P.n_sites(10.0, 50.0) >= 12)
    check("a degenerate radius asks for nothing", P.n_sites(0, 50.0) == 0)

    # ---- shells ------------------------------------------------------------
    sites = P.sample_virion([100.0, 100.0, 100.0], radius_px=40.0, angpix=10.0,
                            spacing_A=50.0, shells_A=(0.0, 50.0, -50.0),
                            virion_id=7)
    shells = sorted({s["shell"] for s in sites})
    check("every requested shell is sampled", shells == [-50.0, 0.0, 50.0])
    check("each site records which virion it belongs to",
          all(s["virion"] == 7 for s in sites))
    r = {}
    for s in sites:
        d = np.linalg.norm(np.array([s["z"], s["y"], s["x"]]) - 100.0)
        r.setdefault(s["shell"], []).append(d * 10.0)     # px -> A
    check("the 0 shell sits at the fitted radius (400 A)",
          abs(np.mean(r[0.0]) - 400.0) < 1.0)
    check("the +50 shell sits 50 A outside it",
          abs(np.mean(r[50.0]) - 450.0) < 1.0)
    check("the -50 shell sits 50 A inside it",
          abs(np.mean(r[-50.0]) - 350.0) < 1.0)
    check("the outer shell carries more sites than the inner one (more area)",
          len(r[50.0]) > len(r[-50.0]))

    # Normals point OUTWARD: dot(normal, position - centre) > 0 everywhere.
    outward = all(
        np.dot([s["nz"], s["ny"], s["nx"]],
               np.array([s["z"], s["y"], s["x"]]) - 100.0) > 0 for s in sites)
    check("every normal points outward from the centre", outward)

    # Sites outside the volume are dropped, not clamped onto the face.
    edge = P.sample_virion([10.0, 10.0, 10.0], radius_px=40.0, angpix=10.0,
                           shells_A=(0.0,), shape=(200, 200, 200))
    check("sites outside the tomogram are dropped rather than clamped",
          all(0 <= s["z"] < 200 and 0 <= s["y"] < 200 and 0 <= s["x"] < 200
              for s in edge) and len(edge) > 0)

    # ---- the star ----------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "picks.star"
        n = P.write_star(p, "Position003", sites)
        body = p.read_text()
        check("star writes one row per site", n == len(sites))
        check("rlnTomoName is the tomostar stem, exactly",
              "\nPosition003\t" in body)
        check("theta rides along in its own column",
              "_tomogrationTheta" in body)
        check("and so does the shell, for duplicate removal after refinement",
              "_tomogrationShell" in body)
        check("tilt/psi priors are written for RELION",
              "_rlnAngleTiltPrior" in body and "_rlnAnglePsiPrior" in body)
        check("the loop header numbers every column",
              all(f"#{i}" in body for i in range(1, 14)))
        rows = [l for l in body.splitlines() if l.startswith("Position003\t")]
        check("every row has the full column count",
              all(len(r.split("\t")) == 13 for r in rows))

        # Randomised psi: undetermined must not become a spurious consensus.
        p2 = Path(td) / "picks_rand.star"
        P.write_star(p2, "Position003", sites, random_psi=True, seed=1)
        psis = [float(l.split("\t")[6]) for l in p2.read_text().splitlines()
                if l.startswith("Position003\t")]
        check("random psi actually varies", len(set(psis)) > len(psis) // 2)
        psis0 = [float(l.split("\t")[6]) for l in rows]
        check("the default is a declared 0, not a random guess",
              set(psis0) == {0.0})


    # ---- crYOLO picks made on a FILTERED tomogram --------------------------
    # crYOLO names its output after the volume it picked on, so picking on
    # IsoNet tomograms gives Position003_12.56Apx_isonet2.coords. The converter
    # globbed '*_<apx>Apx.coords' — anchored at the END — so every file had to
    # be bulk renamed by hand before it would run at all. 2026-08-28.
    with tempfile.TemporaryDirectory() as _td:
        _r = Path(_td)
        (_r / "COORDS").mkdir()
        (_r / "recon").mkdir()
        (_r / "out").mkdir()

        def _mrc(path, nx=512, ny=512, nz=386):
            h = bytearray(1024)
            struct.pack_into("<3i", h, 0, nx, ny, nz)
            struct.pack_into("<i", h, 12, 2)
            path.write_bytes(bytes(h))

        for _pos in ("Position003", "Position045"):
            # Only the PLAIN reconstructions exist, as in the real project.
            _mrc(_r / f"recon/{_pos}_12.56Apx.mrc")
            (_r / f"COORDS/{_pos}_12.56Apx_isonet2.coords").write_text(
                "100 200 150\n101 201 151\n102 202 152\n")

        _res = subprocess.run(
            [sys.executable, str(REPO / "ml_cryolo_to_warp_picks_auto.py"),
             str(_r / "COORDS"), str(_r / "recon"), "--out_dir", str(_r / "out"),
             "--apx", "12.56", "--suffix", "cryolo_isonetmodel", "--execute"],
            capture_output=True, text=True)
        _made = sorted(f.name for f in (_r / "out").glob("*.star"))
        check("picks named after a FILTERED tomogram convert without renaming",
              _res.returncode == 0 and len(_made) == 2)
        check("and the star is named for the SERIES, not the variant",
              _made == ["Position003_12.56Apx_cryolo_isonetmodel.star",
                        "Position045_12.56Apx_cryolo_isonetmodel.star"])
        _rows = [l for l in (_r / "out" / _made[0]).read_text().splitlines()
                 if l.strip() and not l.strip().startswith(("data_", "loop_", "_"))]
        check("every pick is written, not just the paired ones", len(_rows) == 3)
        check("coordinates are normalised 0-1 fractions",
              all(0.0 <= float(t) <= 1.0 for t in _rows[-1].split()[:3]))

    # ---- and thresholding those picks is meaningless -----------------------
    # .coords carry NO score, so the converter writes a CONSTANT figure of
    # merit. threshold_picks can then only keep everything or nothing, while
    # looking like it did something.
    _st = load("tomogration_stages")
    _thr = next(x for x in _st.STAGES if x["id"] == "threshold_picks")
    check("a crYOLO pick set is called out before it is thresholded",
          "cryolo" in _thr["validate"](
              {"in_suffix": "12.56Apx_cryolo_isonet"}).lower())
    check("a template-match suffix is not",
          _thr["validate"]({"in_suffix": "12.56Apx_tmpl"}) == "")
    _h = next(q for q in _thr["params"] if q["name"] == "minimum")["help"]
    check("the help says it is not a fraction, which is what it looked like",
          "NOT a fraction" in _h)
    check("and names the column the number actually cuts on",
          "_rlnAutopickFigureOfMerit" in _h)
    check("the title no longer claims sigma for every input",
          "\u03c3" not in next(q for q in _thr["params"]
                               if q["name"] == "minimum")["title"])


    # ---- every args.X a tool reads must be a flag it declares --------------
    # Removing MODE A and B from ml_relion4_select_picks took the --fom
    # argument with them while mode_c still wrote args.fom into every row, so
    # MODE C died with AttributeError at the first particle — 100% of the time,
    # after printing a full, healthy-looking report. 2026-08-31.
    import ast as _ast                                        # noqa: PLC0415
    for _tool in ("ml_relion4_select_picks", "ml_cryolo_to_warp_picks_auto",
                  "ml_verify_reextract", "ml_star_coords"):
        _src = (REPO / f"{_tool}.py").read_text()
        _t = _ast.parse(_src)
        _used = {n.attr for n in _ast.walk(_t)
                 if isinstance(n, _ast.Attribute)
                 and isinstance(n.value, _ast.Name)
                 and n.value.id in ("args", "a")}
        _declared = set()
        for n in _ast.walk(_t):
            if not (isinstance(n, _ast.Call)
                    and getattr(n.func, "attr", "") == "add_argument"):
                continue
            for kw in n.keywords:
                if kw.arg == "dest" and isinstance(kw.value, _ast.Constant):
                    _declared.add(kw.value.value)
            for arg in n.args:
                if isinstance(arg, _ast.Constant) and isinstance(arg.value, str):
                    _declared.add(arg.value.lstrip("-").replace("-", "_"))
        _missing = sorted(x for x in _used - _declared if not x.startswith("_"))
        check(f"{_tool}: every args.X is a declared flag "
              + (f"(missing: {_missing})" if _missing else ""), not _missing)


    # ---- angles in a pick star do NOT rotate the subtomograms --------------
    # Verified in WarpTools source (2026-09-02): ts_export_particles never
    # pre-rotates (PrerotateParticles is a GUI-only option, default off); input
    # angle columns are copied verbatim into the output star, where
    # relion_reconstruct --3d_rot uses them for an ORIENTED average. The
    # retired converter defaults to zero angles regardless; the export card's
    # RELION-star route carries the refined angles through.
    _src = (REPO / "ml_relion4_select_picks.py").read_text()
    check("the retired converter still parses its old flags",
          '"--no-keep-angles", dest="keep_angles", action="store_false"' in _src
          and '"--no-recenter", dest="recenter", action="store_false"' in _src)
    check("and says at the top that it is retired",
          "RETIRED 2026-09-02" in _src.split("import argparse")[0])

    _st2 = load("tomogration_stages")
    _r2w = next(x for x in _st2.STAGES if x["id"] == "relion4_to_warp")
    check("the pick-star converter card is retired", _r2w.get("legacy") is True)
    check("and carries no mode / recentre / angle switches",
          not any(q["name"] in ("mode", "no_recenter", "keep_angles",
                                "relion_coords", "picks_dir", "recon_dir")
                  for q in _r2w["params"]))
    check("its validator says RETIRED and points at the export card",
          "RETIRED" in _r2w["validate"]({})
          and "ts_export_particles" in _r2w["validate"]({}))
    check("select-good-class is gone",
          not any(x["id"] == "relion4_select_picks" for x in _st2.STAGES))


    # ---- the angles must survive the round trip to the EXPORTED star -------
    with tempfile.TemporaryDirectory() as _td:
        _r = Path(_td)
        _sh = ("data_optics\nloop_\n_rlnOpticsGroup #1\n_rlnImagePixelSize #2\n"
               "1 6.28\n\ndata_particles\nloop_\n_rlnCoordinateX #1\n"
               "_rlnCoordinateY #2\n_rlnCoordinateZ #3\n_rlnTomoName #4\n"
               "_rlnAngleRot #5\n_rlnAngleTilt #6\n_rlnAnglePsi #7\n")
        _nh = ("data_optics\nloop_\n_rlnOpticsGroup #1\n_rlnImagePixelSize #2\n"
               "1 3.14\n\ndata_particles\nloop_\n_rlnCoordinateX #1\n"
               "_rlnCoordinateY #2\n_rlnCoordinateZ #3\n_rlnTomoName #4\n"
               "_rlnImageName #5\n")
        _sr, _nr = [], []
        for _i in range(120):
            _x, _y, _z = 100 + _i * 2, 150 + _i * 3, 300 + (_i % 90)
            _t = f"Position{3 + (_i % 4) * 21:03d}.tomostar"
            _sr.append(f"{_x}.000 {_y}.000 {_z}.000 {_t} "
                       f"{_i % 360}.000 {_i % 180}.000 {(_i * 7) % 360}.000")
            _nr.append(f"{_x * 2}.000 {_y * 2}.000 {_z * 2}.000 {_t} "
                       f"subtomo/x/p{_i:07d}.mrc")
        (_r / "source.star").write_text(_sh + "\n".join(_sr) + "\n")
        (_r / "export.star").write_text(_nh + "\n".join(_nr) + "\n")

        _res = subprocess.run(
            [sys.executable, str(REPO / "ml_merge_angles.py"),
             str(_r / "source.star"), str(_r / "export.star"),
             "--out", str(_r / "merged.star")], capture_output=True, text=True)
        check("merge pairs every particle across the pixel-size change",
              _res.returncode == 0 and "matched 120 / 120" in _res.stdout)

        _m = load("ml_relion4_select_picks")
        _mb = _m.read_star_blocks(_r / "merged.star")
        _mp = _m.particles_block(_mb, need=("rlnCoordinateX",))
        _c = _mp["cols"]
        check("the exported star gains all three angle columns",
              all(f"rlnAngle{n}" in _c for n in ("Rot", "Tilt", "Psi")))
        _row = _mp["rows"][5]
        check("and the angles are the SOURCE's refined values",
              (float(_row[_c["rlnAngleRot"]]), float(_row[_c["rlnAngleTilt"]]),
               float(_row[_c["rlnAnglePsi"]])) == (5.0, 5.0, 35.0))
        check("coordinates are untouched by the merge",
              float(_row[_c["rlnCoordinateX"]]) == (100 + 5 * 2) * 2)
        check("the subtomogram path survives",
              "p0000005.mrc" in _row[_c["rlnImageName"]])

        # Two stars from different re-extractions must not silently half-merge.
        (_r / "other.star").write_text(_nh + "\n".join(
            f"{9000 + _i}.000 {9000 + _i}.000 {9000 + _i}.000 PositionZZZ.tomostar "
            f"subtomo/x/p{_i:07d}.mrc" for _i in range(10)) + "\n")
        _bad = subprocess.run(
            [sys.executable, str(REPO / "ml_merge_angles.py"),
             str(_r / "source.star"), str(_r / "other.star"),
             "--out", str(_r / "bad.star")], capture_output=True, text=True)
        check("mismatched stars are refused, not half-merged",
              _bad.returncode != 0 and "nothing paired" in _bad.stdout + _bad.stderr)


    # ---- recentring: Warp does it, on the RELION-star route -----------------
    # Verified in WarpTools source (2026-09-02, present since the first
    # release): ts_export_particles subtracts rlnOriginX/Y/ZAngst ÷ the star's
    # own pixel size from every coordinate, on both input routes. So the export
    # card hands Warp the RELION star and the SAME pixel size, and nothing
    # recentres by hand any more. EML46's bin1 re-extraction done this way
    # (2026-09-02) gave a clean reference and a working classification.
    _exp = next(x for x in _st.STAGES if x["id"] == "ts_export_particles")
    _capx = next(q for q in _exp["params"] if q["name"] == "coords_angpix")
    check("coords_angpix is sent on the RELION-star route", "skip_if" not in _capx)
    check("and its help names the value: the star's rlnImagePixelSize",
          "rlnImagePixelSize" in _capx["help"])
    _norm = next(q for q in _exp["params"] if q["name"] == "normalized_coords")
    check("'0-1 fractions' is dropped on the RELION-star route",
          _norm["skip_if"]({"input_star": "s.star"}) is True
          and _norm["skip_if"]({"input_star": ""}) is False)
    _istar = next(q for q in _exp["params"] if q["name"] == "input_star")
    check("the star field says Warp applies the shifts itself",
          "SUBTRACTS" in _istar["help"] or "subtracts" in _istar["help"])
    _ver = next(x for x in _st.STAGES if x["id"] == "relion4_verify_reextract")
    _vnr = next(q for q in _ver["params"] if q["name"] == "no_recenter")
    check("the verifier expects the shifts to have been applied",
          _vnr["default"] is False)


    # ---- did RELION already move the coordinates? --------------------------
    # The open question after recentring was ruled the culprit: is the origin a
    # shift still to be applied, or one RELION has ALREADY applied to the
    # coordinates? Ranges cannot answer it (a subset changes them for trivial
    # reasons) but rlnImageName can, because a RELION selection still names the
    # subtomogram file Warp wrote. These two cases must not be confusable.
    import subprocess as _sp2                                 # noqa: PLC0415
    _HDR = ("data_optics\nloop_\n_rlnOpticsGroup #1\n_rlnImagePixelSize #2\n"
            "1 6.28\n\ndata_particles\nloop_\n_rlnCoordinateX #1\n"
            "_rlnCoordinateY #2\n_rlnCoordinateZ #3\n_rlnOriginXAngst #4\n"
            "_rlnOriginYAngst #5\n_rlnOriginZAngst #6\n_rlnImageName #7\n")

    def _star(path, mode):
        rows = []
        for i in range(50):
            x, y, z = 100 + i * 3, 200 + i * 2, 300 + i
            ox, oy, oz = ((i % 13) - 6) * 3.1, ((i % 7) - 3) * 4.2, ((i % 5) - 2) * 5.3
            if mode == "in":
                ox = oy = oz = 0.0
            elif mode == "recentred":
                x, y, z = x - ox / 6.28, y - oy / 6.28, z - oz / 6.28
            rows.append(f"{x:.5f} {y:.5f} {z:.5f} {ox:.5f} {oy:.5f} {oz:.5f} "
                        f"sub/p{i:05d}_6.28A.mrc")
        Path(path).write_text(_HDR + "\n".join(rows) + "\n")

    with tempfile.TemporaryDirectory() as _td:
        _d = Path(_td)
        _star(_d / "in.star", "in")
        _star(_d / "kept.star", "kept")
        _star(_d / "recentred.star", "recentred")
        _tool = str(REPO / "ml_star_compare.py")

        _a = _sp2.run([sys.executable, _tool, str(_d / "in.star"),
                       str(_d / "kept.star")], capture_output=True, text=True).stdout
        check("coordinates RELION left alone are reported unchanged",
              "COORDINATES UNCHANGED" in _a)
        check("and that is not called a double-count",
              "double-count" not in _a.split("->")[-1] or "NOT a double-count" in _a)

        _b = _sp2.run([sys.executable, _tool, str(_d / "in.star"),
                       str(_d / "recentred.star")], capture_output=True, text=True).stdout
        check("coordinates RELION already moved are caught",
              "ALREADY RECENTRED" in _b and "delta == -origin" in _b)
        check("with the count, so a partial match cannot pass as total",
              "50 match delta == -origin" in _b)

        # Matching is by IMAGE NAME; stars about different particles must not
        # be silently 'compared' to zero differences.
        _star(_d / "other.star", "kept")
        _txt = (_d / "other.star").read_text().replace("sub/p", "other/q")
        (_d / "other.star").write_text(_txt)
        _c = _sp2.run([sys.executable, _tool, str(_d / "in.star"),
                       str(_d / "other.star")], capture_output=True, text=True).stdout
        check("unrelated stars report NO MATCH rather than a false verdict",
              "NO PARTICLES MATCHED" in _c)


    # ---- the two remaining explanations, as switches ----------------------
    # RELION does NOT move coordinates (proved 2026-09-02: 414,990 and 28,850
    # particles matched by image name, max delta 0.000 A through Class3D AND
    # Subset selection). So mode_c's subtraction is the first application, and
    # what remains is either an inverted SIGN or an origin expressed in the
    # particle's rotated FRAME. Both displace a particle similarly, so only an
    # experiment separates them -- hence switches rather than a guess.
    import math as _math                                      # noqa: PLC0415
    _SP = load("ml_relion4_select_picks")
    check("no rotation leaves a vector untouched",
          all(abs(a - b) < 1e-9 for a, b in
              zip(_SP.rotate_zyz(10, 20, 30, 0, 0, 0), (10, 20, 30))))
    check("rot=90 about z sends x to y",
          all(abs(a - b) < 1e-9 for a, b in
              zip(_SP.rotate_zyz(1, 0, 0, 90, 0, 0), (0, 1, 0))))
    _lenok = True
    for _i in range(200):
        _v = [((_i * 37) % 91) - 45, ((_i * 53) % 71) - 35, ((_i * 17) % 61) - 30]
        _a = [((_i * 29) % 360) - 180, ((_i * 11) % 360) - 180,
              ((_i * 43) % 360) - 180]
        _w = _SP.rotate_zyz(*_v, *_a)
        if abs(_math.dist(_v, (0, 0, 0)) - _math.dist(_w, (0, 0, 0))) > 1e-9:
            _lenok = False
    check("every rotation preserves length, so it cannot rescale a shift",
          _lenok)
    _srcp = (REPO / "ml_relion4_select_picks.py").read_text()
    check("sign and frame are switches, both defaulting to RELION's convention",
          '"--recenter-sign", type=float, default=1.0' in _srcp
          and 'default="tomogram"' in _srcp)
    check("and the run report states which was used, so a result is traceable",
          "sign {args.recenter_sign:+g}, frame" in _srcp)


    # ---- the comparison must look at ORIGINS, not only coordinates --------
    # The first version compared coordinates and printed origins without
    # comparing them, so it proved "RELION does not move coordinates" and left
    # the actual question open. RELION's Select has a "Re-center the class
    # averages" option; if it compensates by adjusting each particle's origin,
    # the origins in a selection are measured against a MOVED reference. That
    # adjustment is a per-class centre-of-mass shift ROTATED into each
    # particle's frame -- near-zero mean per class, large spread.
    with tempfile.TemporaryDirectory() as _td:
        _d = Path(_td)
        _H = ("data_optics\nloop_\n_rlnOpticsGroup #1\n_rlnImagePixelSize #2\n"
              "1 6.28\n\ndata_particles\nloop_\n_rlnCoordinateX #1\n"
              "_rlnCoordinateY #2\n_rlnCoordinateZ #3\n_rlnOriginXAngst #4\n"
              "_rlnOriginYAngst #5\n_rlnOriginZAngst #6\n_rlnAngleRot #7\n"
              "_rlnAngleTilt #8\n_rlnAnglePsi #9\n_rlnClassNumber #10\n"
              "_rlnImageName #11\n")

        def _pair(recentre):
            rows = []
            for _i in range(120):
                _rot = ((_i * 31) % 360) - 180
                _tilt = (_i * 17) % 180
                _psi = ((_i * 41) % 360) - 180
                _ox, _oy, _oz = ((_i % 11) - 5) * 3.0, ((_i % 7) - 3) * 4.0, \
                    ((_i % 5) - 2) * 5.0
                _cl = 1 if _i % 2 else 8
                if recentre:
                    _com = (7.0, -4.0, 3.0) if _cl == 1 else (-6.0, 2.0, -5.0)
                    _dx, _dy, _dz = _SP.rotate_zyz(*_com, _rot, _tilt, _psi)
                    _ox, _oy, _oz = _ox + _dx, _oy + _dy, _oz + _dz
                rows.append(f"{100+_i}.0 {200+_i}.0 {300+_i}.0 {_ox:.5f} "
                            f"{_oy:.5f} {_oz:.5f} {_rot:.5f} {_tilt:.5f} "
                            f"{_psi:.5f} {_cl} sub/p{_i:05d}_6.28A.mrc")
            return _H + "\n".join(rows) + "\n"

        (_d / "cls.star").write_text(_pair(False))
        (_d / "sel.star").write_text(_pair(True))
        _o = subprocess.run(
            [sys.executable, str(REPO / "ml_star_compare.py"),
             str(_d / "cls.star"), str(_d / "sel.star"), "-n", "0"],
            capture_output=True, text=True).stdout
        check("coordinates are still reported unchanged",
              "coordinates moved in 0/120" in _o)
        check("but a rewritten ORIGIN is now caught",
              "origins changed in 120/120" in _o)
        check("and it names the selection's recentring as the cause",
              "'Re-center the class averages'" in _o
              and "MOVED reference" in _o)
        check("with the per-class breakdown that shows it was rotated",
              "class   1:" in _o and "spread" in _o)
        check("angles, which did not change, are reported as unchanged",
              "angles changed in 0/120" in _o)

        # And when NOTHING changed, it must not invent a finding.
        (_d / "same.star").write_text(_pair(False))
        _o2 = subprocess.run(
            [sys.executable, str(REPO / "ml_star_compare.py"),
             str(_d / "cls.star"), str(_d / "same.star"), "-n", "0"],
            capture_output=True, text=True).stdout
        check("identical stars report no origin change at all",
              "origins changed in 0/120" in _o2
              and "MOVED reference" not in _o2)


    # ---- one tomogram at a time -------------------------------------------
    # Every bin1 SUCCESS used a single tomogram; every bin1 FAILURE spanned 71.
    # That variable stayed confounded for six rounds because bin4 worked across
    # all of them, so multi-tomogram was assumed innocent. Filtering an
    # existing export's star by tomogram tests it for free.
    with tempfile.TemporaryDirectory() as _td:
        _d = Path(_td)
        _rows = "\n".join(
            f"{100+_i} {200+_i} {300+_i} subtomo/Position{_p:03d}/"
            f"Position{_p:03d}_{_i:07d}_1.57A.mrc"
            for _i in range(60)
            for _p in [(3, 45, 116)[_i % 3]][:1])
        (_d / "m.star").write_text(
            "data_optics\nloop_\n_rlnOpticsGroup #1\n_rlnImagePixelSize #2\n"
            "1 1.57\n\ndata_particles\nloop_\n_rlnCoordinateX #1\n"
            "_rlnCoordinateY #2\n_rlnCoordinateZ #3\n_rlnImageName #4\n"
            + _rows + "\n")
        _r = subprocess.run(
            [sys.executable, str(REPO / "ml_star_edit.py"), str(_d / "m.star"),
             "--out", str(_d / "one.star"), "--tomo", "Position003"],
            capture_output=True, text=True)
        _txt = (_d / "one.star").read_text()
        _kept = [l for l in _txt.splitlines() if "Position003" in l]
        check(f"a star can be filtered to one tomogram ({len(_kept)} rows)",
              _r.returncode == 0 and len(_kept) == 20)
        check("and no other tomogram leaks in",
              "Position045" not in _txt and "Position116" not in _txt)
        check("the pixel size survives the filter, or the reference is wrong",
              "1.57" in _txt.split("data_particles")[0])
        _bad = subprocess.run(
            [sys.executable, str(REPO / "ml_star_edit.py"), str(_d / "m.star"),
             "--out", str(_d / "z.star"), "--tomo", "Position999"],
            capture_output=True, text=True)
        check("asking for a tomogram that is not there fails loudly",
              _bad.returncode != 0 and "no particles matched" in
              (_bad.stdout + _bad.stderr))


    # ---- the direct RELION-star route -------------------------------------
    # THE re-extraction route. Warp reads the RELION star's coordinates,
    # subtracts the refined origins itself and scales by coords_angpix, which
    # must be the star's own pixel size. Two earlier direct attempts (EML45
    # 2026-07-24, EML46 J74 2026-09-02) failed only because they passed
    # --normalized_coords, which multiplies pixel coordinates by the tomogram
    # width; the run that worked (EML46, 2026-09-02) was
    #   --input_star Select/job029/particles.star --coords_angpix 6.28
    #   --output_angpix 1.57 --box 192 --diameter 240 --3d
    _exp = next(x for x in _st.STAGES if x["id"] == "ts_export_particles")
    check("the export card can take a RELION star directly",
          any(q["name"] == "input_star" for q in _exp["params"]))
    _v = {q["name"]: q.get("default") for q in _exp["params"]}
    check("with both filled in, the card says the folder is ignored",
          "IGNORED" in _exp["validate"](dict(_v, input_star="a.star",
                                             coords_angpix="6.28")))
    check("setting neither is refused",
          "NO INPUT" in _exp["validate"](dict(_v, input_directory="")))
    check("a blank coords_angpix with a RELION star is refused",
          "REQUIRED" in _exp["validate"](
              dict(_v, input_star="a.star", input_directory="",
                   coords_angpix="")))
    _ok = dict(_v, input_star="Select/job029/particles.star",
               input_directory="", input_pattern="", coords_angpix="6.28",
               output_angpix="1.57", box="192", diameter="240")
    check("a clean direct-star setup raises nothing",
          _exp["validate"](dict(_ok, normalized_coords=False)) == "")
    check("but leaving '0-1 fractions' ticked is called out, not ignored "
          "silently",
          "IGNORED with a RELION star" in _exp["validate"](
              dict(_ok, normalized_coords=True)))
    _cmd = _st.build_command(_exp, _ok)
    check("and it emits --input_star with no --input_directory",
          "--input_star Select/job029/particles.star" in _cmd
          and "--input_directory" not in _cmd)
    check("with --coords_angpix, which Warp requires on every route",
          "--coords_angpix 6.28" in _cmd)

    # The card's DEFAULTS fill in the pick-star folder and pattern, so warning
    # about "both routes" was not enough: the flags must be DROPPED, not warned
    # about. coords_angpix is NOT dropped: an earlier version dropped it and Warp
    # refused the run ("either input pixel size or coordinates are normalized
    # must be set").
    _both = dict(_v, input_star="Select/job029/particles.star",
                 output_angpix="3.14", box="128", diameter="240",
                 coords_angpix="6.28")
    _c2 = _st.build_command(_exp, _both)
    check("pick-star flags are dropped even when their defaults are populated",
          "--input_directory" not in _c2 and "--input_pattern" not in _c2)
    check("and coords_angpix stays, being required",
          "--coords_angpix 6.28" in _c2)
    check("while the RELION star is still passed",
          "--input_star Select/job029/particles.star" in _c2)
    check("the pick-star route is untouched when no star is given",
          all(f in _st.build_command(_exp, dict(_v, input_star=""))
              for f in ("--input_directory", "--input_pattern")))

    # --normalized_coords on the direct route is the mistake that produced J74
    # (2026-09-02): a RELION star holds PIXEL coordinates, so declaring them
    # 0-1 fractions made Warp multiply every one by the tomogram width. It
    # wrote coordinates of 79,354 .. 4,114,010 into a 4096 px volume -- 1004x
    # too large, every particle cut from outside the tomogram.
    check("'0-1 fractions' is never sent with a RELION star, whatever the box "
          "is ticked to",
          "--normalized_coords" not in _st.build_command(
              _exp, dict(_v, input_star="s.star", normalized_coords=True)))
    check("and the card says so rather than silently dropping it",
          "IGNORED with a RELION star" in _exp["validate"](
              dict(_v, input_star="s.star", input_directory="",
                   normalized_coords=True)))
    check("the pick-star route still declares fractions when asked",
          "--normalized_coords" in _st.build_command(
              _exp, dict(_v, input_star="", normalized_coords=True)))


    # ---- the retired converter: readable, inert, no modes -------------------
    # Its three modes wrote three coordinate conventions, and choosing wrong was
    # silent. The stage now exists only so old cards keep their row and their
    # recorded command; it has no mode switch and refuses to run.
    _t2w2 = next(x for x in _st.STAGES if x["id"] == "relion4_to_warp")
    check("the retired card has no mode switch",
          not any(q["name"] == "mode" for q in _t2w2["params"]))
    _b = {q["name"]: q.get("default") for q in _t2w2["params"]}
    _b.update(particles_star="Select/job029/particles.star", out_dir="out",
              suffix="v1")
    _c = _st.build_command(_t2w2, _b)
    check("it never emits --mode or --relion-coords",
          "--mode" not in _c and "--relion-coords" not in _c)
    check("and refuses with a pointer to the export card",
          "RETIRED" in _t2w2["validate"](_b)
          and "ts_export_particles" in _t2w2["validate"](_b))
    check("it is kept out of the palette and the rail",
          _t2w2.get("legacy") is True
          and all(n["stage_id"] != "relion4_to_warp"
                  for n in load("tomogration_jobs").canvas_layout(
                      {"seq": 0, "jobs": {}})[0]))

    # Rows a skip_if switches off are HIDDEN, not merely disabled: a disabled
    # row is not visibly different in this theme. Driven by the SAME skip_if
    # the command uses, so what you can edit and what gets sent cannot disagree
    # (the export card's pick-star fields vanish once a RELION star is set).
    _app_src = (REPO / "tomogration_app.py").read_text()
    check("irrelevant rows are hidden, not just disabled",
          "w.setVisible(not off)" in _app_src
          and "w.setEnabled(not off)" not in _app_src)
    _exp = next(x for x in _st.STAGES if x["id"] == "ts_export_particles")
    _hidden = {q["name"] for q in _exp["params"]
               if q.get("skip_if") and q["skip_if"]({"input_star": "s.star"})}
    check("a RELION star hides the pick-star fields and the fractions box",
          {"input_directory", "input_pattern", "normalized_coords"} <= _hidden
          and "coords_angpix" not in _hidden)

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
