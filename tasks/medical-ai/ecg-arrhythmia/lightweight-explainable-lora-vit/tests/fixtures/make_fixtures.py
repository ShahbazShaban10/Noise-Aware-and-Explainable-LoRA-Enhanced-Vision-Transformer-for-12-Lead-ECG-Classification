"""Deterministic synthetic WFDB fixtures for the offline test tier.

Why synthetic: the real corpus is 2.5 GB, open access but not vendored, and cannot be
committed. The unit tier must still be able to exercise header parsing, label resolution,
denoising and windowing against real *files*, byte-for-byte identically on every machine.

Why these are NOT a substitute for the corpus: the waveforms here are procedurally generated
from a small set of class-conditioned templates. They are adequate to prove the pipeline is
wired correctly and that filters do what they claim. They are NOT adequate to say anything
about diagnostic accuracy, and no test in this repository reports an accuracy number computed
on them. The corpus-gated tier does that.

Determinism: every value derives from `numpy.random.default_rng(SEED + record_index)`.
`manifest.sha256` pins the resulting bytes; `verify()` checks them.

    python make_fixtures.py --out ./corpus          # generate
    python make_fixtures.py --out ./corpus --check  # verify against the manifest
    python make_fixtures.py --out ./corpus --write-manifest
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.io import savemat

SEED = 20240917
FS = 500.0
N_SAMPLES = 5000
N_LEADS = 12
GAIN = 1000  # ADC units per mV, matching the real corpus
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# SNOMED codes chosen so each fixture resolves to a known class under class_map_7.json.
# The extra codes are realistic multi-label noise (T-wave abnormality, axis deviation).
CLASS_FIXTURES: Dict[str, Dict[str, object]] = {
    "NSR":   {"codes": ["426783006"],            "rate": (62, 92),  "n": 6},
    "AFIB":  {"codes": ["164889003", "164934002"], "rate": (95, 150), "n": 6},
    "SB":    {"codes": ["426177001"],            "rate": (42, 57),  "n": 6},
    "ST":    {"codes": ["427084000"],            "rate": (105, 145), "n": 6},
    "SVT":   {"codes": ["426761007"],            "rate": (165, 210), "n": 6},
    "CD":    {"codes": ["426783006", "59118001"], "rate": (65, 90),  "n": 6},
    "OTHER": {"codes": ["164934002", "39732003"], "rate": (70, 95),  "n": 6},
}

# Per-lead amplitude scaling, loosely following real 12-lead R-wave progression.
LEAD_SCALE = np.array([0.9, 1.2, 0.5, -1.0, 0.4, 0.8, 0.3, 0.7, 1.1, 1.4, 1.2, 0.9])


def _beat(t: np.ndarray, qrs_width_s: float, p_amp: float, rsr: bool) -> np.ndarray:
    """One P-QRS-T complex on a time axis centred at the R peak."""
    g = lambda c, w, a: a * np.exp(-0.5 * ((t - c) / w) ** 2)  # noqa: E731
    p = g(-0.18, 0.022, p_amp)
    q = g(-qrs_width_s * 0.55, qrs_width_s * 0.18, -0.12)
    r = g(0.0, qrs_width_s * 0.22, 1.0)
    s = g(qrs_width_s * 0.55, qrs_width_s * 0.20, -0.22)
    tw = g(0.30, 0.055, 0.28)
    beat = p + q + r + s + tw
    if rsr:  # RSR' notch of a bundle-branch block
        beat = beat + g(qrs_width_s * 0.95, qrs_width_s * 0.22, 0.45)
    return beat


def _synthesise(label: str, rng: np.random.Generator) -> np.ndarray:
    """Return a clean (12, 5000) mV-scale signal for `label`."""
    spec = CLASS_FIXTURES[label]
    lo, hi = spec["rate"]  # type: ignore[misc]
    bpm = float(rng.uniform(lo, hi))
    rr = 60.0 / bpm

    wide = label == "CD"
    qrs_w = 0.14 if wide else 0.075
    p_amp = 0.0 if label in ("AFIB", "SVT") else 0.13
    rsr = wide

    # Beat times. AFIB gets irregularly irregular RR; everything else is regular with jitter.
    times: List[float] = []
    t_cur = float(rng.uniform(0.15, 0.45))
    duration = N_SAMPLES / FS
    while t_cur < duration:
        times.append(t_cur)
        if label == "AFIB":
            t_cur += rr * float(rng.uniform(0.55, 1.55))
        else:
            t_cur += rr * (1.0 + float(rng.normal(0, 0.02)))

    tax = np.arange(N_SAMPLES) / FS
    trace = np.zeros(N_SAMPLES)
    win = 0.55
    for i, bt in enumerate(times):
        m = np.abs(tax - bt) < win
        if not m.any():
            continue
        # OTHER: every 4th beat is a wide, P-less ventricular ectopic.
        ectopic = label == "OTHER" and i % 4 == 3
        trace[m] += _beat(
            tax[m] - bt,
            qrs_width_s=0.16 if ectopic else qrs_w,
            p_amp=0.0 if ectopic else p_amp,
            rsr=rsr,
        ) * (1.5 if ectopic else 1.0)

    if label == "AFIB":  # coarse fibrillatory baseline in place of P waves
        for f in (6.0, 8.5, 11.0):
            trace += 0.045 * np.sin(2 * np.pi * f * tax + float(rng.uniform(0, 6.28)))

    sig = np.outer(LEAD_SCALE, trace)
    if wide:  # exaggerate the RSR' in V1-V3, as a real RBBB does
        sig[6:9] *= 1.6
    return sig


def _corrupt(sig: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add exactly the three noise types the denoising stage claims to remove."""
    tax = np.arange(N_SAMPLES) / FS
    out = sig.copy()
    for l in range(N_LEADS):
        # baseline wander, below the 0.5 Hz passband edge
        out[l] += 0.35 * np.sin(2 * np.pi * 0.15 * tax + float(rng.uniform(0, 6.28)))
        out[l] += 0.20 * np.sin(2 * np.pi * 0.30 * tax + float(rng.uniform(0, 6.28)))
        # 50 Hz powerline
        out[l] += 0.12 * np.sin(2 * np.pi * 50.0 * tax + float(rng.uniform(0, 6.28)))
        # EMG, above the 40 Hz passband edge
        out[l] += 0.05 * rng.standard_normal(N_SAMPLES)
        # a DC offset, so z-scoring has something to do
        out[l] += float(rng.uniform(-0.4, 0.4))
    return out


def _write_header(path: Path, rid: str, codes: List[str], age: int, sex: str) -> None:
    lines = [f"{rid} {N_LEADS} {int(FS)} {N_SAMPLES}"]
    for name in LEAD_NAMES:
        lines.append(f"{rid}.mat 16+24 {GAIN}/mV 16 0 0 0 0 {name}")
    lines += [
        f"#Age: {age}",
        f"#Sex: {sex}",
        f"#Dx: {','.join(codes)}",
        "#Rx: Unknown",
        "#Hx: Unknown",
        "#Sx: Unknown",
        "",
    ]
    path.write_text("\n".join(lines))


def generate(out_dir: Path) -> List[Tuple[str, str]]:
    """Write the fixture corpus. Returns [(record_id, expected_class), ...]."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Tuple[str, str]] = []
    idx = 0

    for label in sorted(CLASS_FIXTURES):
        for k in range(int(CLASS_FIXTURES[label]["n"])):  # type: ignore[arg-type]
            rng = np.random.default_rng(SEED + idx)
            rid = f"FX{idx:05d}"
            sig = _corrupt(_synthesise(label, rng), rng)

            adc = np.clip(np.round(sig * GAIN), -32768, 32767).astype(np.int16)
            # MATLAB level-4, variable 'val': the 24-byte header matches the '16+24'
            # format field, and column-major storage of (12, 5000) is exactly the
            # channel-interleaved order WFDB expects.
            savemat(str(out_dir / f"{rid}.mat"), {"val": adc}, format="4")
            _write_header(
                out_dir / f"{rid}.hea",
                rid,
                list(CLASS_FIXTURES[label]["codes"]),  # type: ignore[arg-type]
                age=int(rng.integers(28, 85)),
                sex="Male" if idx % 2 == 0 else "Female",
            )
            written.append((rid, label))
            idx += 1

    # Deliberate negatives, so rejection paths are covered by real files.
    bad = out_dir / "FX90001.hea"
    bad.write_text(
        f"FX90001 {N_LEADS} {int(FS)} {N_SAMPLES}\n"
        + "\n".join(f"FX90001.mat 16+24 {GAIN}/mV 16 0 0 0 0 {n}" for n in LEAD_NAMES)
        + "\n#Age: 55\n#Sex: Male\n#Rx: Unknown\n"          # no #Dx line
    )
    savemat(str(out_dir / "FX90001.mat"),
            {"val": np.zeros((N_LEADS, N_SAMPLES), dtype=np.int16)}, format="4")

    lead8 = out_dir / "FX90002.hea"
    lead8.write_text(
        f"FX90002 8 {int(FS)} {N_SAMPLES}\n"
        + "\n".join(f"FX90002.mat 16+24 {GAIN}/mV 16 0 0 0 0 {n}" for n in LEAD_NAMES[:8])
        + "\n#Age: 61\n#Sex: Female\n#Dx: 426783006\n"       # only 8 leads
    )
    savemat(str(out_dir / "FX90002.mat"),
            {"val": np.zeros((8, N_SAMPLES), dtype=np.int16)}, format="4")

    orphan = out_dir / "FX90003.hea"                          # header with no signal file
    orphan.write_text(
        f"FX90003 {N_LEADS} {int(FS)} {N_SAMPLES}\n"
        + "\n".join(f"FX90003.mat 16+24 {GAIN}/mV 16 0 0 0 0 {n}" for n in LEAD_NAMES)
        + "\n#Age: 44\n#Sex: Male\n#Dx: 426177001\n"
    )
    return written


def _digests(out_dir: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in sorted(Path(out_dir).iterdir()):
        if p.suffix in (".hea", ".mat"):
            out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def write_manifest(out_dir: Path, manifest: Path) -> None:
    lines = [f"{h}  {n}" for n, h in sorted(_digests(out_dir).items())]
    manifest.write_text("\n".join(lines) + "\n")


def verify(out_dir: Path, manifest: Path) -> List[str]:
    """Returns a list of mismatch descriptions; empty means the bytes are identical."""
    expected: Dict[str, str] = {}
    for line in Path(manifest).read_text().splitlines():
        if line.strip():
            h, n = line.split(None, 1)
            expected[n.strip()] = h
    actual = _digests(out_dir)
    problems = []
    for name, h in expected.items():
        if name not in actual:
            problems.append(f"missing: {name}")
        elif actual[name] != h:
            problems.append(f"digest mismatch: {name}")
    for name in actual:
        if name not in expected:
            problems.append(f"unexpected file: {name}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--manifest", type=Path,
                    default=Path(__file__).parent / "manifest.sha256")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write-manifest", action="store_true")
    a = ap.parse_args(argv)

    written = generate(a.out)
    print(f"generated {len(written)} labelled records + 3 rejection cases in {a.out}")

    if a.write_manifest:
        write_manifest(a.out, a.manifest)
        print(f"wrote manifest {a.manifest}")
        return 0
    if a.check:
        problems = verify(a.out, a.manifest)
        if problems:
            print("FIXTURES ARE NOT REPRODUCIBLE:", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            return 1
        print("fixtures match the manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
