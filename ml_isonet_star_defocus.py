#!/usr/bin/env python3
"""Inject per-tomogram defocus into an IsoNet tomograms.star (v1 or v2).

    python3 ml_isonet_star_defocus.py <tomograms.star> <warp_xml_dir>

`isonet.py prepare_star` writes ONE defocus value into every row, but the
defocus is per tilt series and lives in warp_tiltseries/<series>.xml in
MICROMETRES. This rewrites the defocus column per row (×10000 → Å, the unit
IsoNet documents), matching each row's tomogram name to the LONGEST xml stem
that prefixes it — the same underscore-safe rule the membrane wrappers use, so
Position_1 never claims Position_10's defocus.

IsoNet 1 and IsoNet 2 name the columns differently (_rlnMicrographName vs
_rlnTomoName), so both spellings are accepted and the chosen pair is echoed.
IsoNet 2 also takes --defocus as a per-tomogram LIST, but that binds values to
star ROW ORDER; matching by name here cannot silently pair a defocus with the
wrong tomogram, which is the failure that matters.

Rows with no matching xml (or no readable defocus) are left unchanged and
reported — never guessed. Edits are atomic (temp file + rename). Exit 0 with
a per-row report; exit 1 if the star is unusable or NOTHING could be matched.
"""
import re
import sys
from pathlib import Path

_DEFOCUS_RE = re.compile(r'<Param\s+Name="Defocus"\s+Value="([^"]+)"')

# Column spellings seen in the wild, most specific first. IsoNet 1 writes
# _rlnMicrographName/_rlnDefocus; IsoNet 2's star is keyed on _rlnTomoName
# (its own deconv --input_column default). Order decides which wins when a
# star carries both.
NAME_COLS = ("_rlnMicrographName", "_rlnTomoName",
             "_rlnTomoReconstructedTomogramHalf1")
DEFOCUS_COLS = ("_rlnDefocus", "_rlnDefocusU", "_rlnTomoDefocus")


def defocus_A(xml_path):
    try:
        m = _DEFOCUS_RE.search(xml_path.read_text())
        return float(m.group(1)) * 10000.0 if m else None
    except (OSError, ValueError):
        return None


def stem_for(name, stems):
    """Longest xml stem that prefixes `name` at a '_'/'.' boundary (or fully)."""
    best = ""
    for s in stems:
        if name == s or name.startswith(s + "_") or name.startswith(s + "."):
            if len(s) > len(best):
                best = s
    return best


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    star_path, xml_dir = Path(sys.argv[1]), Path(sys.argv[2])
    if not star_path.is_file():
        sys.exit(f"ERROR: star file not found: {star_path}")
    if not xml_dir.is_dir():
        sys.exit(f"ERROR: xml dir not found: {xml_dir}")
    stems = sorted(p.stem for p in xml_dir.glob("*.xml"))
    if not stems:
        sys.exit(f"ERROR: no .xml in {xml_dir}")

    lines = star_path.read_text().splitlines()
    # Header-driven column lookup: never assume positions.
    header = {}
    n_cols = 0
    for ln in lines:
        m = re.match(r"\s*(_rln\w+)\s+#(\d+)", ln)
        if m:
            n_cols = max(n_cols, int(m.group(2)))
            header[m.group(1)] = int(m.group(2)) - 1
    # A column can be PRESENT and still useless: prepare_star writes the literal
    # string 'None' into columns it has no data for, so a star built from
    # even/odd halves has an _rlnTomoName full of 'None'. Pick the first name
    # column that actually holds a path.
    def first_value(col):
        idx = header[col]
        for ln in lines:
            f = ln.split()
            if (len(f) >= n_cols and not ln.lstrip().startswith(("_", "#"))
                    and f[0] not in ("data_", "loop_")):
                return f[idx]
        return ""

    name_key = next((c for c in NAME_COLS
                     if c in header and first_value(c) not in ("None", "none", "")),
                    None)
    defocus_key = next((c for c in DEFOCUS_COLS if c in header), None)
    if name_key is None or defocus_key is None:
        # Print what the star DOES have: a renamed column upstream is a
        # one-line fix here, but only if the actual name is visible.
        print("ERROR: no name/defocus column pair in " + str(star_path),
              file=sys.stderr)
        print("  looked for name:    " + " ".join(NAME_COLS), file=sys.stderr)
        print("  looked for defocus: " + " ".join(DEFOCUS_COLS), file=sys.stderr)
        print("  this star has:      "
              + (" ".join(sorted(header)) or "(no _rln columns at all)"),
              file=sys.stderr)
        sys.exit(1)
    name_col, defocus_col = header[name_key], header[defocus_key]
    print(f"columns: name={name_key} defocus={defocus_key}")

    out, patched, unmatched = [], 0, []
    for ln in lines:
        f = ln.split()
        # A data row has at least as many fields as declared columns and no
        # leading underscore/keyword.
        if (len(f) >= n_cols and not ln.lstrip().startswith(("_", "#"))
                and f[0] not in ("data_", "loop_")
                and not ln.lstrip().startswith(("data_", "loop_"))):
            base = Path(f[name_col]).name
            base = re.sub(r"\.(mrc|rec)$", "", base)
            stem = stem_for(base, stems)
            df = defocus_A(xml_dir / f"{stem}.xml") if stem else None
            if df is not None:
                f[defocus_col] = f"{df:.1f}"
                out.append("  " + "\t".join(f))
                print(f"{base}: defocus {df / 10000:.2f} um -> {df:.1f} A")
                patched += 1
                continue
            unmatched.append(base)
        out.append(ln)

    for b in unmatched:
        print(f"WARN: {b}: no readable defocus in {xml_dir} — row left as-is.")
    if patched == 0:
        sys.exit("ERROR: no star row could be matched to a defocus — check "
                 "the xml dir and tomogram naming.")
    tmp = star_path.with_name(star_path.name + ".tmp")
    tmp.write_text("\n".join(out) + "\n")
    tmp.replace(star_path)
    print(f"patched {patched} row(s), {len(unmatched)} left unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
