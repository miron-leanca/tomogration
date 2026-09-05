#!/usr/bin/env python3
# Part of tomogration2 — the membrane branch's tomogram VARIANT REGISTRY
# (cards spec draft 2, Part B). Pure stdlib, no Qt and no numpy: the numeric
# work that needs mrcfile lives in ml_variant_polarity.py, which calls back
# into the one decision rule here.
#
# Import graph stays a DAG: this module imports only core + stdlib, so stages,
# jobs and the app can all use it.

"""Tomogram variants: what a folder of tomograms IS, in units that survive
being compared.

Three things went wrong often enough to need a registry rather than paths:

  * A card pointed at a hard-coded folder, so switching bin or backend meant
    editing every card (and forgetting one).
  * A size cutoff of "1000 voxels" means a different physical object at every
    binning — 8x different between bin4 and bin8 — so two runs compared
    nothing (Trap 2).
  * The density-support gate assumes membranes are DARK. Feed it a variant
    whose contrast is the other way round and it silently inverts: every real
    virion fails, and phantoms in bulk ice pass (Trap 1).

Each entry records name, path, pixel size, binning, provenance, polarity and
whether extraction may read it. Only the original reconstruction may.
"""

import json
import os
import re
from pathlib import Path

REGISTRY_FILE = ".tomogration_variants.json"

# Warp writes the reconstruction pixel size into the filename:
# Position003_12.56Apx.mrc. It is the cheapest honest source of Å/px — the
# alternative is opening an MRC header, which needs mrcfile.
_APX_IN_NAME = re.compile(r"_(\d+(?:\.\d+)?)Apx", re.IGNORECASE)

# Provenance values. Only ORIGINAL may be extracted from (spec §5 / A5).
ORIGINAL = "reconstruction"
EXTRACTION_LEGAL = {ORIGINAL}

# Polarity of MEMBRANES in a variant, which is what the density gate keys on.
DARK = "dark"        # membranes darker than their surroundings
BRIGHT = "bright"    # membranes brighter (Warp's inverted convention)
UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Trap 1 — polarity is DETECTED, never declared
# ---------------------------------------------------------------------------
def polarity_from_medians(membrane_median, global_median, tol=0.0):
    """Membrane polarity from two medians: the tomogram sampled at voxels the
    SEGMENTATION already calls membrane, against the whole volume.

    A user-set flag can be wrong, and when it is wrong nothing complains — the
    gate just selects the opposite of what it should. Two medians cannot be
    wrong in that way. `tol` guards the degenerate case where a segmentation
    covers nearly everything and the two medians coincide; below it the answer
    is UNKNOWN rather than a coin flip.
    """
    if membrane_median is None or global_median is None:
        return UNKNOWN
    diff = float(membrane_median) - float(global_median)
    if abs(diff) <= tol:
        return UNKNOWN
    return DARK if diff < 0 else BRIGHT


def polarity_disagreement(entry, detected):
    """A message when a fresh detection contradicts what this variant recorded
    before, else ''. Silent re-labelling is how a gate ends up inverted between
    two runs of 'the same' variant."""
    old = str((entry or {}).get("polarity", UNKNOWN))
    if detected in (UNKNOWN, "") or old in (UNKNOWN, "", None):
        return ""
    if old == detected:
        return ""
    return (f"polarity of variant '{entry.get('name', '?')}' was detected as "
            f"{detected}, but this registry recorded {old}. One of the two "
            f"runs measured a different volume — check the path before "
            f"trusting any density-support gate on it.")


# ---------------------------------------------------------------------------
# Trap 2 — sizes are physical, never raw voxels
# ---------------------------------------------------------------------------
def voxel_volume_nm3(angpix):
    """Volume of one voxel in nm³ (angpix is Å/px; 10 Å = 1 nm)."""
    a = float(angpix) / 10.0
    return a ** 3


def nm3_to_voxels(nm3, angpix):
    """Physical volume -> voxel count at this pixel size, rounded to an int."""
    return int(round(float(nm3) / voxel_volume_nm3(angpix)))


def voxels_to_nm3(voxels, angpix):
    return float(voxels) * voxel_volume_nm3(angpix)


_SIZE_RE = re.compile(
    r"^\s*(?P<num>\d+(?:\.\d+)?)\s*(?:"
    r"(?P<nm>nm3|nm\^3|nm³)"                       # 520nm3
    r"|(?P<vox>v|vox|voxels)?\s*@\s*(?P<ref>\d+(?:\.\d+)?)"   # 1000@12.56
    r")\s*$", re.IGNORECASE)


def parse_size(text):
    """A size specification -> ('nm3', value). Accepts

        520nm3        an explicit physical volume
        1000@12.56    1000 voxels AT a reference pixel size (Å/px)

    and REFUSES a bare number. A bare '1000' is the whole trap: it reads as a
    voxel count, and a voxel count without its pixel size is not a size — it
    is 8x different between bin4 and bin8. Raises ValueError with the two
    forms spelled out."""
    m = _SIZE_RE.match(str(text or ""))
    if not m:
        raise ValueError(
            f"{text!r} is not a size. Give a physical volume ('520nm3') or a "
            f"voxel count WITH the pixel size it was measured at "
            f"('1000@12.56'). A bare voxel count means a different object at "
            f"every binning — 8x different between bin4 and bin8.")
    num = float(m.group("num"))
    if m.group("nm"):
        return ("nm3", num)
    return ("nm3", voxels_to_nm3(num, m.group("ref")))


def resolve_size(text, angpix):
    """A size specification -> the voxel count that means it AT `angpix`.
    This is what a tool flag receives; the UI shows both."""
    _, nm3 = parse_size(text)
    return nm3_to_voxels(nm3, angpix)


def describe_size(text, angpix):
    """'520 nm³ = 1000 voxels at 12.56 Å/px' — both halves, always, so a
    number in a log can be compared against a number from another binning."""
    _, nm3 = parse_size(text)
    return (f"{nm3:.6g} nm³ = {nm3_to_voxels(nm3, angpix)} voxels "
            f"at {float(angpix):.6g} Å/px")


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
def angpix_from_name(name):
    """Pixel size parsed out of a Warp tomogram filename, or None."""
    m = _APX_IN_NAME.search(str(name))
    return float(m.group(1)) if m else None


def binning_label(angpix, native=1.57):
    """'bin8' for 12.56 Å/px off a 1.57 Å/px detector. Reported, never
    computed with — the pixel size is the number that means something."""
    if not angpix:
        return ""
    ratio = float(angpix) / float(native)
    nearest = min((1, 2, 4, 8, 16, 32), key=lambda b: abs(b - ratio))
    return f"bin{nearest}" if abs(nearest - ratio) < 0.25 * nearest else f"{angpix:g}Å"


def make_entry(name, path, angpix=None, provenance=ORIGINAL, polarity=UNKNOWN,
               native=1.57, note=""):
    return {
        "name": name,
        "path": str(path).rstrip("/"),
        "angpix": float(angpix) if angpix else None,
        "binning": binning_label(angpix, native),
        "provenance": provenance,
        "polarity": polarity,
        "extraction_legal": provenance in EXTRACTION_LEGAL,
        "note": note,
    }


def _provenance_of(dirpath):
    """Read a folder's PROVENANCE.json: (variant, extraction_allowed|None)."""
    p = Path(dirpath) / "PROVENANCE.json"
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError):
        return ("", None)
    if not isinstance(data, dict):
        return ("", None)
    allowed = data.get("extraction_allowed")
    return (str(data.get("variant", "")),
            allowed if isinstance(allowed, bool) else None)


def _first_tomogram(dirpath):
    """One .mrc name from a folder, for the pixel size. Bounded: these folders
    hold 72 volumes of several GB and this runs during a UI refresh."""
    try:
        for p in Path(dirpath).iterdir():
            if p.suffix.lower() == ".mrc":
                return p.name
    except OSError:
        pass
    return ""


# Where tomogram folders live. Globs are one level deep by design — a
# recursive walk here would enumerate the subtomo trees.
_SEARCH = [
    ("raw", "warp_tiltseries/reconstruction", ORIGINAL),
    ("raw", "jobs/*/reconstruction", ORIGINAL),
    ("deconv", "membrane/deconv/*", "tomo_preprocessing"),
    ("isonet1", "membrane/isonet_corrected*/corrected", "isonet1"),
    ("isonet2", "membrane/isonet2_corrected*/corrected", "isonet2"),
    ("isonet2", "jobs/*/corrected", "isonet2"),
]


def discover(root):
    """Every tomogram folder under `root` that looks like a variant.

    Discovery over configuration: the folders already exist and already carry
    PROVENANCE.json. A registry the user has to fill in by hand is a registry
    that disagrees with the disk."""
    root = Path(root)
    out, seen = [], set()
    for base_name, pattern, provenance in _SEARCH:
        try:
            hits = sorted(root.glob(pattern))
        except OSError:
            continue
        for d in hits:
            if not d.is_dir():
                continue
            rel = os.path.relpath(d, root)
            if rel in seen:
                continue
            first = _first_tomogram(d)
            if not first:
                continue                      # a folder with no tomograms is not a variant
            seen.add(rel)
            tagged, allowed = _provenance_of(d)
            prov = provenance
            if tagged:
                # The folder's own tag beats the path pattern: a variant
                # written somewhere unexpected still declares what it is.
                prov = tagged
            entry = make_entry(
                name=_unique_name(base_name, rel, {e["name"] for e in out}),
                path=rel,
                angpix=angpix_from_name(first),
                provenance=prov)
            if allowed is False:
                entry["extraction_legal"] = False
            out.append(entry)
    return out


def _unique_name(base, rel, taken):
    """'isonet2' for the first one, then 'isonet2 (jobs/J18)' — two runs of the
    same backend are different variants and must not collapse into one row."""
    if base not in taken:
        return base
    parent = str(Path(rel).parent)
    cand = f"{base} ({parent})" if parent not in (".", "") else f"{base} ({rel})"
    n = 2
    while cand in taken:
        cand, n = f"{base} #{n}", n + 1
    return cand


def registry_path(root):
    return Path(root) / REGISTRY_FILE


def load(root):
    """The stored registry ({} when there is none). Never merges: what was
    detected before is data, and rediscovery is an explicit action."""
    try:
        data = json.loads(registry_path(root).read_text())
    except (OSError, ValueError):
        return {"variants": []}
    if not isinstance(data, dict) or not isinstance(data.get("variants"), list):
        return {"variants": []}
    return data


def save(root, registry):
    """Write via a temp file + replace, so a crash mid-write cannot leave a
    truncated registry behind."""
    p = registry_path(root)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(registry, indent=1))
    os.replace(tmp, p)


def refresh(root):
    """Rediscover, keeping every DETECTED fact already recorded for a path.

    Discovery knows where volumes are and what wrote them; it cannot know a
    polarity, which costs a pass over the voxels. Losing that on every refresh
    would mean re-measuring constantly, so detected fields survive and only
    the discovered ones are refreshed."""
    old = {e.get("path"): e for e in load(root).get("variants", [])}
    fresh = discover(root)
    for e in fresh:
        prev = old.get(e["path"])
        if not prev:
            continue
        for key in ("polarity", "polarity_source", "polarity_date", "note"):
            if prev.get(key) not in (None, "", UNKNOWN):
                e[key] = prev[key]
    reg = {"variants": fresh}
    save(root, reg)
    return reg


def get(root, name):
    for e in load(root).get("variants", []):
        if e.get("name") == name:
            return e
    return None


def record_polarity(root, path, detected, source=""):
    """Store a detected polarity against a variant. Returns a warning string
    when it contradicts what was there (never silently overwrites without
    saying so)."""
    reg = load(root)
    warn = ""
    for e in reg.get("variants", []):
        if e.get("path") != str(path).rstrip("/"):
            continue
        warn = polarity_disagreement(e, detected)
        e["polarity"] = detected
        if source:
            e["polarity_source"] = source
        save(root, reg)
        return warn
    # No entry for this path — and previously that meant the measurement was
    # simply DROPPED: the tool printed 'dark', exited 0, and nothing was
    # stored, so every downstream run still said 'polarity unknown'. A
    # measurement is too expensive to lose over a missing registry row, so
    # create one.
    # Provenance is READ from the folder, never defaulted: make_entry's default
    # is 'original', which would mark an IsoNet-corrected or deconvolved folder
    # extraction_legal=True — the one thing the §5 rule exists to prevent.
    full = Path(root) / str(path) if not os.path.isabs(str(path)) else Path(path)
    tagged, _allowed = _provenance_of(full)     # returns a PAIR, not a string
    prov = tagged or ORIGINAL
    reg.setdefault("variants", []).append(
        make_entry(name=Path(str(path)).name or "variant", path=path,
                   angpix=angpix_from_name(str(path)), provenance=prov,
                   polarity=detected,
                   note="created by a polarity measurement"))
    if source:
        reg["variants"][-1]["polarity_source"] = source
    save(root, reg)
    return warn


def extraction_refusal(entry):
    """Why extraction may not read this variant, or '' when it may.

    The §5 hard rule: coordinates may come from a deconvolved, denoised or
    wedge-restored volume; the particles themselves must be extracted from the
    ORIGINAL reconstruction."""
    if not entry:
        return ""
    if entry.get("extraction_legal"):
        return ""
    return (f"'{entry.get('name', entry.get('path', '?'))}' is a "
            f"{entry.get('provenance', 'processed')} variant "
            f"({entry.get('path', '?')}). Coordinates may come from it; "
            f"EXTRACTION must reference the original reconstruction — the "
            f"densities in a processed volume are not the measured ones.")
