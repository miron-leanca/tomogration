#!/usr/bin/env python3
"""
ml_imodtowarpkey_generator_warp_auto.py

Generates an IMOD-to-Warp tilt-numbering conversion key from a reference mdoc.

The mdoc ZValue blocks are in dose-symmetric acquisition order (the order Warp
expects). When you inspect a tilt series in IMOD (3dmod), the images are
displayed in tilt-angle order (negative to positive). This script produces a
lookup that lets us translate IMOD tilt numbers into acquisition-order numbers.

Usage:
    python ml_imodtowarpkey_generator_warp_auto.py <input_mdoc> <output_key>

Arguments:
    input_mdoc   Path to a complete reference mdoc (all tilts present)
    output_key   Path where the conversion key will be written

Output:
    A text file with one integer per line. Line N (1-indexed) contains the
    acquisition-order index of the tilt that appears at IMOD position N
    (sorted by tilt angle ascending).

Example:
    Acquisition order:  1 (0°), 2 (+3°), 3 (-3°), 4 (+6°), 5 (-6°)
    Sorted by angle:    -6° (5), -3° (3), 0° (1), +3° (2), +6° (4)
    Conversion key:     5\n3\n1\n2\n4\n
"""

import sys
import os


def parse_mdoc_tilts(mdoc_path):
    """Extract (acquisition_order, tilt_angle) tuples from an mdoc."""
    tilts = []
    acquisition_order = 0
    with open(mdoc_path) as f:
        for line in f:
            line_stripped = line.strip()
            # Match TiltAngle line (after the ZValue block starts)
            if line_stripped.startswith("TiltAngle"):
                # Parse "TiltAngle = -0.03" robustly
                if "=" in line_stripped:
                    try:
                        angle = float(line_stripped.split("=", 1)[1].strip())
                        acquisition_order += 1
                        tilts.append((acquisition_order, angle))
                    except ValueError:
                        print(f"WARNING: Could not parse tilt angle from line: {line_stripped}")
                        continue
    return tilts


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    mdoc_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.isfile(mdoc_path):
        print(f"ERROR: Input mdoc not found: {mdoc_path}")
        sys.exit(1)

    print(f"Reading tilt angles from: {mdoc_path}")
    tilts = parse_mdoc_tilts(mdoc_path)

    if not tilts:
        print("ERROR: No TiltAngle entries found in mdoc.")
        sys.exit(1)

    print(f"Found {len(tilts)} tilts")

    # Sort by tilt angle (negative to positive = IMOD display order)
    sorted_tilts = sorted(tilts, key=lambda x: x[1])

    # Write conversion key: one line per IMOD tilt, containing the
    # acquisition-order index of that tilt
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        for acq_order, angle in sorted_tilts:
            f.write(f"{acq_order}\n")

    print(f"Wrote conversion key: {output_path}")
    print()
    print("Mapping (IMOD position -> acquisition order, tilt angle):")
    for imod_pos, (acq_order, angle) in enumerate(sorted_tilts, start=1):
        print(f"  IMOD {imod_pos:3d} -> acq {acq_order:3d} ({angle:+.2f}°)")
    print()
    print(f"Successfully generated conversion key with {len(sorted_tilts)} entries.")


if __name__ == "__main__":
    main()
