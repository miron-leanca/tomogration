"""Tomogram variant registry (cards spec draft 2, Part B) — the two traps.

THE POINT of these tests:

  * Trap 1: polarity is DETECTED from two medians, never declared. A wrong
    declaration inverts the density-support gate silently — every real virion
    rejected, phantoms in ice accepted — so the rule that decides it is pinned
    here, including the degenerate case where the medians coincide.
  * Trap 2: a size is physical. '1000 voxels' is a different object at every
    binning (8x between bin4 and bin8), so a bare voxel count is REFUSED and
    every size resolves through a pixel size.
  * §5 hard rule: only the original reconstruction is extraction-legal, and a
    folder's own PROVENANCE.json beats the path it was found under.

    python3 tests/test_variants.py
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE / "stub"))
sys.path.insert(0, str(REPO))


def load(mod):
    spec = importlib.util.spec_from_file_location(mod, REPO / f"{mod}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod] = m
    spec.loader.exec_module(m)
    return m


V = load("tomogration_variants")
passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok  {name}")
    else:
        failed += 1
        print(f"FAIL  {name}")


def main():
    # ---- Trap 1: polarity from medians ------------------------------------
    check("polarity: membranes below the global median are dark",
          V.polarity_from_medians(-1.4, 0.0) == V.DARK)
    check("polarity: membranes above it are bright",
          V.polarity_from_medians(1.4, 0.0) == V.BRIGHT)
    check("polarity: an offset volume is judged by the DIFFERENCE, not the sign",
          V.polarity_from_medians(120.0, 140.0) == V.DARK
          and V.polarity_from_medians(-5.0, -9.0) == V.BRIGHT)
    # A segmentation covering nearly everything makes the two medians the same
    # volume; answering anyway would be a coin flip that silently sets a gate.
    check("polarity: medians within tolerance are UNKNOWN, not a guess",
          V.polarity_from_medians(0.5, 0.4, tol=0.5) == V.UNKNOWN)
    check("polarity: a missing median is UNKNOWN",
          V.polarity_from_medians(None, 0.0) == V.UNKNOWN)

    entry = {"name": "isonet2", "polarity": V.DARK}
    check("polarity: a contradicting re-detection warns",
          "isonet2" in V.polarity_disagreement(entry, V.BRIGHT))
    check("polarity: agreeing is silent",
          V.polarity_disagreement(entry, V.DARK) == "")
    check("polarity: first detection of an unknown is silent",
          V.polarity_disagreement({"name": "x", "polarity": V.UNKNOWN},
                                  V.BRIGHT) == "")

    # ---- Trap 2: sizes are physical ---------------------------------------
    # The whole trap in one pair of numbers: the same physical virion is 8x
    # more voxels at bin4 than at bin8.
    v8 = V.resolve_size("1000@12.56", 12.56)
    v4 = V.resolve_size("1000@12.56", 6.28)
    check(f"size: the same cutoff is 8x more voxels at bin4 ({v8} -> {v4})",
          v8 == 1000 and v4 == 8000)
    check("size: nm³ resolves the same way",
          V.resolve_size("1981.3nm3", 12.56) == 1000)
    check("size: a round trip is stable",
          V.nm3_to_voxels(V.voxels_to_nm3(1000, 12.56), 12.56) == 1000)
    for bare in ("1000", "  250 ", "1e3"):
        try:
            V.parse_size(bare)
            ok = False
        except ValueError as e:
            ok = "8x" in str(e) or "8x different" in str(e)
        check(f"size: a bare voxel count {bare!r} is REFUSED with the reason", ok)
    check("size: the description carries both halves",
          "nm³" in V.describe_size("1000@12.56", 6.28)
          and "8000 voxels" in V.describe_size("1000@12.56", 6.28))

    # ---- pixel size and binning -------------------------------------------
    check("angpix: parsed out of Warp's filename",
          V.angpix_from_name("Position003_12.56Apx.mrc") == 12.56
          and V.angpix_from_name("Position003_6.28Apx_scores.mrc") == 6.28)
    check("angpix: absent when the name doesn't carry it",
          V.angpix_from_name("tomo_A.mrc") is None)
    check("binning: labelled off the native pixel size",
          V.binning_label(12.56) == "bin8" and V.binning_label(6.28) == "bin4")

    # ---- discovery, provenance and the §5 hard rule ------------------------
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        recon = root / "warp_tiltseries" / "reconstruction"
        recon.mkdir(parents=True)
        (recon / "Position003_12.56Apx.mrc").write_text("")
        corr = root / "membrane" / "isonet2_corrected" / "corrected"
        corr.mkdir(parents=True)
        (corr / "Position003_6.28Apx.mrc").write_text("")
        (corr.parent / "PROVENANCE.json").write_text(json.dumps(
            {"variant": "wedge_restored", "extraction_allowed": False}))
        (corr / "PROVENANCE.json").write_text(json.dumps(
            {"variant": "wedge_restored", "extraction_allowed": False}))
        empty = root / "membrane" / "deconv" / "s1.0_f1.0"
        empty.mkdir(parents=True)

        found = V.discover(root)
        names = {e["name"]: e for e in found}
        check(f"discover: finds the raw and the corrected variant ({sorted(names)})",
              len(found) == 2)
        check("discover: a folder with no tomograms is not a variant",
              not any("deconv" in n for n in names))
        raw = names.get("raw")
        check("discover: raw carries its pixel size and binning",
              raw and raw["angpix"] == 12.56 and raw["binning"] == "bin8")
        check("discover: raw is extraction-legal", raw and raw["extraction_legal"])
        iso = names.get("isonet2")
        check("discover: the corrected variant is bin4",
              iso and iso["angpix"] == 6.28 and iso["binning"] == "bin4")
        check("discover: PROVENANCE.json beats the path pattern",
              iso and iso["provenance"] == "wedge_restored")
        check("discover: and it is NOT extraction-legal",
              iso and not iso["extraction_legal"])
        check("refusal: names the variant and says coordinates are still fine",
              "EXTRACTION" in V.extraction_refusal(iso)
              and "Coordinates may come from it" in V.extraction_refusal(iso))
        check("refusal: the original reconstruction is allowed through",
              V.extraction_refusal(raw) == "")

        # A detected polarity costs a pass over the voxels; rediscovery must
        # not throw it away.
        V.refresh(root)
        warn = V.record_polarity(root, iso["path"], V.BRIGHT, source="seg.mrc")
        check("record: first detection is silent", warn == "")
        check("record: it is persisted",
              V.get(root, "isonet2")["polarity"] == V.BRIGHT)
        V.refresh(root)
        check("refresh: rediscovery keeps the detected polarity",
              V.get(root, "isonet2")["polarity"] == V.BRIGHT)
        warn2 = V.record_polarity(root, iso["path"], V.DARK)
        check("record: a contradicting detection warns", "isonet2" in warn2)
        check("record: and the new measurement wins",
              V.get(root, "isonet2")["polarity"] == V.DARK)

        # Two runs of the same backend are different variants; collapsing them
        # into one row is how two folders got confused before.
        j18 = root / "jobs" / "J18" / "corrected"
        j18.mkdir(parents=True)
        (j18 / "Position003_6.28Apx.mrc").write_text("")
        names2 = {e["name"] for e in V.discover(root)}
        check(f"discover: a second isonet2 run gets its own name ({sorted(names2)})",
              len(names2) == 3 and any(n.startswith("isonet2 ") for n in names2))

    with tempfile.TemporaryDirectory() as td:
        check("load: a project with no registry yields an empty one",
              V.load(Path(td)) == {"variants": []})

    
    # ---- a measured polarity must never be lost ------------------------------
    # record_polarity used to loop over the registry looking for a matching path
    # and, finding none, return silently. With no registry file at all — the normal
    # state of a fresh project — the tool printed 'dark', exited 0, and stored
    # NOTHING, so every downstream run still said 'polarity unknown'.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "warp_tiltseries" / "reconstruction").mkdir(parents=True)
        V.record_polarity(td, "warp_tiltseries/reconstruction", V.BRIGHT,
                          source="Position003")
        reg = V.load(td)
        check("a polarity measured with NO registry present is still recorded",
              reg["variants"] and reg["variants"][0]["polarity"] == V.BRIGHT)
        check("and it records how it was measured",
              reg["variants"][0].get("polarity_source") == "Position003")
        # An auto-created entry must not default to extraction-legal: make_entry's
        # default is 'original', and a wedge-restored folder marked legal is
        # exactly what the extraction rule exists to prevent.
        corr = root / "jobs" / "J31" / "corrected"
        corr.mkdir(parents=True)
        (corr / "PROVENANCE.json").write_text(
            '{"variant": "wedge_restored", "extraction_allowed": false}')
        V.record_polarity(td, "jobs/J31/corrected", V.DARK)
        entry = [e for e in V.load(td)["variants"] if "corrected" in e["path"]][0]
        check("an auto-created entry reads provenance from the folder",
              entry["provenance"] == "wedge_restored")
        check("so a wedge-restored variant is never marked extraction-legal",
              entry["extraction_legal"] is False)
        check("while a real reconstruction still is",
              [e for e in V.load(td)["variants"]
               if "warp_tiltseries" in e["path"]][0]["extraction_legal"] is True)
        check("a later contradiction warns instead of silently flipping",
              bool(V.record_polarity(td, "jobs/J31/corrected", V.BRIGHT)))

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
