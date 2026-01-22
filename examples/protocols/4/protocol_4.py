import re
from typing import List, Tuple, Optional, Sequence, Any

import matplotlib.pyplot as plt


def load_jcamp_xy_lists(path: str) -> Tuple[List[float], List[float]]:
    # Metadata defaults
    xfactor = 1.0
    yfactor = 1.0
    firstx: Optional[float] = None
    lastx: Optional[float] = None
    deltax: Optional[float] = None
    npoints: Optional[int] = None

    # State
    in_data = False
    data_mode: Optional[str] = None  # "xy_pairs" or "xpp_ylist"

    frequencies: List[float] = []
    intensities: List[float] = []

    # Regex helpers
    keyval_re = re.compile(r"^##\s*([^=]+)\s*=\s*(.*)\s*$")
    float_re = re.compile(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?")

    def extract_numbers(line: str) -> List[float]:
        return [float(x) for x in float_re.findall(line)]

    def parse_float(s: str) -> Optional[float]:
        m = float_re.search(s)
        return float(m.group(0)) if m else None

    def parse_int(s: str) -> Optional[int]:
        m = re.search(r"[-+]?\d+", s)
        return int(m.group(0)) if m else None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # End marker
            if line.upper().startswith("##END"):
                break

            # Header line
            if line.startswith("##"):
                m = keyval_re.match(line)
                if not m:
                    continue

                key = m.group(1).strip().upper()
                val = m.group(2).strip()

                if key == "XYDATA":
                    in_data = True
                    uval = val.upper()
                    data_mode = "xpp_ylist" if "X++" in uval else "xy_pairs"
                    continue

                if key == "XFACTOR":
                    v = parse_float(val)
                    if v is not None:
                        xfactor = v
                elif key == "YFACTOR":
                    v = parse_float(val)
                    if v is not None:
                        yfactor = v
                elif key == "FIRSTX":
                    firstx = parse_float(val)
                elif key == "LASTX":
                    lastx = parse_float(val)
                elif key == "DELTAX":
                    deltax = parse_float(val)
                elif key in ("NPOINTS", "POINTS"):
                    npoints = parse_int(val)

                continue

            # Ignore non-data lines
            if not in_data or line.startswith("$$"):
                continue

            nums = extract_numbers(line)
            if not nums:
                continue

            if data_mode == "xy_pairs":
                # x y x y ...
                if len(nums) % 2 == 1:
                    nums = nums[:-1]

                for i in range(0, len(nums), 2):
                    frequencies.append(nums[i] * xfactor)
                    intensities.append(nums[i + 1] * yfactor)

            elif data_mode == "xpp_ylist":
                # x0 y1 y2 y3 ...
                x0 = nums[0]
                yvals = nums[1:]

                if not yvals:
                    continue

                dx = deltax
                if dx is None and firstx is not None and lastx is not None and npoints and npoints > 1:
                    dx = (lastx - firstx) / (npoints - 1)

                if dx is None:
                    raise ValueError(
                        "X++(Y..Y) data encountered but DELTAX is missing."
                    )

                for j, y in enumerate(yvals):
                    frequencies.append((x0 + j * dx) * xfactor)
                    intensities.append(y * yfactor)

    if not frequencies:
        raise ValueError("No XYDATA found in JCAMP-DX file.")

    if npoints is not None and len(frequencies) > npoints:
        frequencies = frequencies[:npoints]
        intensities = intensities[:npoints]

    return frequencies, intensities


def peak_indices_in_range(
    freqs: Sequence[float],
    intens: Sequence[float],
    f_min: float,
    f_max: float,
    *,
    prominence: Optional[float] = None,
    min_distance: Optional[float] = None,
) -> List[int]:
    if len(freqs) != len(intens):
        raise ValueError("freqs and intens must be the same length.")
    n = len(freqs)
    if n < 3:
        return []

    lo, hi = (f_min, f_max) if f_min <= f_max else (f_max, f_min)

    # Candidate peaks
    candidates: List[int] = []
    for i in range(1, n - 1):
        f = freqs[i]
        if f < lo or f > hi:
            continue

        y0, y1, y2 = intens[i - 1], intens[i], intens[i + 1]
        if not (y1 > y0 and y1 >= y2):
            continue

        if prominence is not None and (y1 - max(y0, y2) < prominence):
            continue

        candidates.append(i)

    if not candidates:
        return []

    # Optional: minimum distance filtering
    if min_distance is not None and min_distance > 0:
        # Sort by height (desc), keep far-enough peaks, then return sorted by index
        by_height = sorted(candidates, key=lambda i: intens[i], reverse=True)
        kept: List[int] = []
        for i in by_height:
            fi = freqs[i]
            if all(abs(fi - freqs[j]) >= min_distance for j in kept):
                kept.append(i)
        return sorted(kept)

    return candidates


if __name__ == "__main__":
    x, y = load_jcamp_xy_lists("651c9779-e75d-4c7c-ad41-1ce312a9e281")

    # sort the data by x values
    xy_sorted = sorted(zip(x, y), key=lambda pair: pair[0])
    x, y = zip(*xy_sorted)

    plt.plot(x, y)
    # limit x axis to 0-4000
    plt.xlim(500, 1500)

    plt.xlabel("Freq. 1/cm")
    plt.ylabel("Intensity")


    locs = peak_indices_in_range(x, y, f_min=500, f_max=1500, prominence=None, min_distance=None)
    print(f"Number of peaks between 500 and 1500 cm^-1: {len(locs)}", flush=True)
    print("Peak locations (1/cm):", flush=True)
    for loc in locs:
        print(f"{x[loc]:.2f}", flush=True)

    plt.scatter([x[i] for i in locs], [y[i] for i in locs], color='red')
    plt.show()
