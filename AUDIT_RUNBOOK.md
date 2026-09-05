> **SUPERSEDED 2026-09-02.** The re-extraction question is settled: export the
> Subset-selection star DIRECTLY (`ts_export_particles --input_star
> Select/jobNNN/particles.star --coords_angpix <rlnImagePixelSize>`, no
> `--normalized_coords`). Warp subtracts the refined `rlnOrigin*Angst` shifts itself
> (verified in WarpTools source, present since the first WarpTools release), and
> the EML46 bin1 re-extraction done this way gave a clean reference. The tests
> below chased the pick-star converter, which is now retired. Kept for the record.

# Why the re-extractions come back as noise — diagnostic runbook

Run these in order. Each one answers a question the previous one leaves open.
Stop as soon as a test fails; that failure is the answer.

Everything below assumes:

    cd /ceph/users/haq21239/EMDatasets/EML46/OC43-3mM-disacch
    TOM=/ceph/users/haq21239/EMDatasets/processing_scripts/tomogration/tomogration2

Section 2 reads subtomograms, so it needs numpy + mrcfile. A plain `python3`
has neither — it will run sections 1 and 3 and tell you to come back. Use:

    module load miniconda/latest && conda activate membrainseg

---

## TEST 1 — the whole chain, in one pass  (5 min)

    python3 $TOM/ml_audit_chain.py \
        --good relion4/cryolo_isonet_J52_ts-export-particles \
        --bad  relion4/reextract_bin2_v2_J69_ts-export-particles \
        --source-star Select/job029/particles.star \
        --star warp_tiltseries/matching_reextract_bin2_v2/Position003_6.28Apx_reextract_bin2_v2.star \
        -n 60

**Section 2 is the one that matters** and it is the test nothing else in this
pipeline performs: it averages 60 real subtomograms from each export and asks
whether the average has a centred object, in numbers.

| GOOD | BAD | means |
|------|-----|-------|
| SNR > 3 | SNR < 1 | The bin4 boxes hold a particle, the re-extracted ones hold ice. Position is lost between them — go to TEST 2. |
| SNR > 3 | SNR 1–3 | Object present but smeared: every particle displaced by a *different* vector. That is the signature of a wrong-sign recentring — go to TEST 2. |
| SNR < 1 | SNR < 1 | The bin4 export was never centred either. The fault predates all of this: the picks, or the pixel size they were made at. Go to TEST 3. |
| SNR > 3 | SNR > 3 | Both centred. Extraction is fine and the fault is downstream — orientations or reference. |

---

## TEST 2 — is it the recentring?  (needs one re-extraction)

The recentring is the only step that MOVES a particle, and it is the one thing
`ml_verify_reextract` structurally cannot check: it verifies
`new == (source − origin/apx) × ratio`, which is the same formula the converter
applies. A PASS proves the converter did what it intended, not that the
intention was right. If RELION's origin sign is opposite to what is assumed,
every particle moves by *twice* its refined shift instead of zero — up to
199.6 Å here, against a 240 Å particle.

Re-run the pick conversion with recentring OFF, export, and audit again:

    python3 $TOM/ml_relion4_select_picks.py Select/job029/particles.star \
        --keep-all --relion-coords --no-recenter \
        --out-dir warp_tiltseries/matching_norecentre \
        --suffix norecentre --execute

Then export with the SAME settings as J69 (coords_angpix 6.28, output_angpix
1.57, box 224, diameter 240) into `relion4/norecentre_test`, and:

    python3 $TOM/ml_audit_chain.py \
        --good relion4/cryolo_isonet_J52_ts-export-particles \
        --bad  relion4/norecentre_test -n 60

If the SNR jumps, the sign was inverted and the fix is in `mode_c`.

---

## TEST 3 — look at it  (2 min, no compute)

The question nobody has asked in this entire investigation: **do the
coordinates sit on particles?**

    python3 $TOM/ml_show_picks.py \
        warp_tiltseries/reconstruction/Position003_12.56Apx.mrc \
        jobs/J50_cryolo-picks/matching/Position003_12.56Apx_cryolo_isonet.star \
        warp_tiltseries/matching_reextract_bin2_v2/Position003_6.28Apx_reextract_bin2_v2.star

Two point layers on the tomogram: the original crYOLO picks (red) and the
re-extract picks (cyan). Scroll in Z.

* Both on density, together → positions are fine, look downstream.
* Red on density, cyan beside it → the recentring moved them. The offset you
  see IS the bug, and its size tells you the sign.
* Neither on density → the picks were never right, and the bin4 result was
  luck or something else entirely.

---

## TEST 4 — a control that removes every variable at once

If TEST 1 says NEITHER export is centred, re-extract the ORIGINAL crYOLO picks
at bin1 — same coordinates that produced the good bin4 map, only finer:

    (export the J50 pick stars with --normalized_coords, output_angpix 1.57,
     box 224, diameter 240, into relion4/cryolo_bin1_control)

    python3 $TOM/ml_audit_chain.py \
        --good relion4/cryolo_isonet_J52_ts-export-particles \
        --bad  relion4/cryolo_bin1_control -n 60

* Control centred → the coordinates survive fine at bin1, and the fault is
  specific to the RELION round trip.
* Control also flat → the fault is in extracting at 1.57 Å/px at all, not in
  the round trip. Box, pixel size or CTF handling at that scale.

This is the test that separates "the re-extraction is broken" from "bin1
extraction is broken", and no result so far distinguishes them.
