"""Lead-aware denoising and normalisation (manuscript section 3.1).

Pipeline, per record, per lead:
    1. 4th-order Butterworth band-pass, 0.5-40 Hz      -> eq. (1)
    2. IIR notch at 50 Hz, Q = 30                      -> sec. 3.1
    3. per-lead z-score with epsilon                   -> eq. (2)
    4. window to exactly 10 s / 5000 samples           -> eq. (3)

Two things here differ from a naive implementation and both matter:

* `filtfilt` needs `padlen` samples of signal. A short or constant lead makes SciPy raise;
  we fall back to a single-pass `lfilter` and record that we did, instead of returning the
  raw signal while claiming it was denoised.
* The notch is skipped when 50 Hz is at or above Nyquist, which happens if someone points
  the loader at a 100 Hz corpus. Silently designing a notch at w0 >= 1 is undefined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sps

from .config import N_LEADS, N_SAMPLES, PreprocessConfig


@dataclass
class PreprocessReport:
    """What actually happened, so the run is auditable rather than assumed."""

    fs_hz: float = 0.0
    fs_assumed: bool = False
    bandpass_applied: bool = False
    notch_applied: bool = False
    notch_skipped_reason: Optional[str] = None
    zerophase: bool = True
    padded_samples: int = 0
    cropped_samples: int = 0
    flat_leads: List[int] = field(default_factory=list)
    nan_leads: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "fs_hz": self.fs_hz,
            "fs_assumed": self.fs_assumed,
            "bandpass_applied": self.bandpass_applied,
            "notch_applied": self.notch_applied,
            "notch_skipped_reason": self.notch_skipped_reason,
            "zerophase": self.zerophase,
            "padded_samples": self.padded_samples,
            "cropped_samples": self.cropped_samples,
            "flat_leads": self.flat_leads,
            "nan_leads": self.nan_leads,
        }


# ---------------------------------------------------------------------------
# Filter design (separated from application so the tests can check the response)
# ---------------------------------------------------------------------------
def design_bandpass(fs: float, low: float, high: float, order: int):
    nyq = 0.5 * fs
    if not (0 < low < high < nyq):
        raise ValueError(
            f"invalid band-pass {low}-{high} Hz for fs={fs} Hz (Nyquist {nyq} Hz)"
        )
    return sps.butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")


def design_notch(fs: float, freq: float, q: float):
    nyq = 0.5 * fs
    if not (0 < freq < nyq):
        raise ValueError(f"notch {freq} Hz is not below Nyquist {nyq} Hz for fs={fs} Hz")
    b, a = sps.iirnotch(freq / nyq, q)
    return b, a


def bandpass_response_db(fs: float, cfg: PreprocessConfig, freqs_hz) -> np.ndarray:
    """Magnitude response of the *zero-phase* cascade, in dB, at the given frequencies.

    filtfilt applies the filter forwards and backwards, so the effective magnitude is
    |H(f)|^2 -- i.e. twice the dB attenuation of the one-pass design. Tests assert against
    this, not against the one-pass response.
    """
    sos = design_bandpass(fs, cfg.bandpass_low_hz, cfg.bandpass_high_hz, cfg.bandpass_order)
    w, h = sps.sosfreqz(sos, worN=np.asarray(freqs_hz, dtype=float), fs=fs)
    mag = np.abs(h) ** 2  # forward + backward
    return 20.0 * np.log10(np.maximum(mag, 1e-30))


def notch_response_db(fs: float, cfg: PreprocessConfig, freqs_hz) -> np.ndarray:
    b, a = design_notch(fs, cfg.notch_freq_hz, cfg.notch_q)
    w, h = sps.freqz(b, a, worN=np.asarray(freqs_hz, dtype=float), fs=fs)
    mag = np.abs(h) ** 2
    return 20.0 * np.log10(np.maximum(mag, 1e-30))


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
def _apply_sos(sos: np.ndarray, x: np.ndarray, report: PreprocessReport) -> np.ndarray:
    # filtfilt padlen default is 3 * (max section order); be explicit and safe.
    padlen = 3 * (sos.shape[0] * 2)
    if x.shape[-1] > padlen:
        return sps.sosfiltfilt(sos, x, axis=-1)
    report.zerophase = False
    return sps.sosfilt(sos, x, axis=-1)


def _apply_ba(b: np.ndarray, a: np.ndarray, x: np.ndarray, report: PreprocessReport) -> np.ndarray:
    padlen = 3 * max(len(a), len(b))
    if x.shape[-1] > padlen:
        return sps.filtfilt(b, a, x, axis=-1)
    report.zerophase = False
    return sps.lfilter(b, a, x, axis=-1)


def denoise(
    x: np.ndarray,
    fs: float,
    cfg: Optional[PreprocessConfig] = None,
    report: Optional[PreprocessReport] = None,
) -> Tuple[np.ndarray, PreprocessReport]:
    """Band-pass + notch. `x` is (leads, samples) float."""
    cfg = cfg or PreprocessConfig()
    report = report or PreprocessReport()
    report.fs_hz = float(fs)

    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected (leads, samples), got shape {x.shape}")

    nan_mask = ~np.isfinite(x)
    if nan_mask.any():
        report.nan_leads = sorted(set(np.where(nan_mask.any(axis=1))[0].tolist()))
        # Linear-interpolate short gaps rather than dropping the record.
        for lead in report.nan_leads:
            row = x[lead]
            bad = ~np.isfinite(row)
            if bad.all():
                row[:] = 0.0
            else:
                idx = np.arange(row.size)
                row[bad] = np.interp(idx[bad], idx[~bad], row[~bad])

    sos = design_bandpass(fs, cfg.bandpass_low_hz, cfg.bandpass_high_hz, cfg.bandpass_order)
    x = _apply_sos(sos, x, report)
    report.bandpass_applied = True

    if cfg.apply_notch:
        if cfg.notch_freq_hz < 0.5 * fs:
            b, a = design_notch(fs, cfg.notch_freq_hz, cfg.notch_q)
            x = _apply_ba(b, a, x, report)
            report.notch_applied = True
        else:
            report.notch_skipped_reason = (
                f"notch {cfg.notch_freq_hz} Hz >= Nyquist {0.5 * fs} Hz"
            )

    return x, report


def zscore_per_lead(
    x: np.ndarray,
    eps: float = 1e-8,
    report: Optional[PreprocessReport] = None,
) -> np.ndarray:
    """Equation (2): x_tilde_l = (x_l - mu_l) / (sigma_l + eps), per lead."""
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    if report is not None:
        flat = np.where(sd.squeeze(-1) < 1e-12)[0]
        if flat.size:
            report.flat_leads = sorted(flat.tolist())
    return (x - mu) / (sd + eps)


def window(
    x: np.ndarray,
    n_samples: int = N_SAMPLES,
    report: Optional[PreprocessReport] = None,
) -> np.ndarray:
    """Equation (3): centre-crop longer records, zero-pad shorter ones."""
    n = x.shape[-1]
    if n == n_samples:
        return x
    if n > n_samples:
        start = (n - n_samples) // 2
        if report is not None:
            report.cropped_samples = n - n_samples
        return x[..., start : start + n_samples]
    pad = n_samples - n
    if report is not None:
        report.padded_samples = pad
    left = pad // 2
    right = pad - left
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(left, right)], mode="constant")


def preprocess_signal(
    x: np.ndarray,
    fs: Optional[float] = None,
    cfg: Optional[PreprocessConfig] = None,
    n_samples: int = N_SAMPLES,
) -> Tuple[np.ndarray, PreprocessReport]:
    """Full section-3.1 pipeline. Returns (12, n_samples) float32 and a report."""
    cfg = cfg or PreprocessConfig()
    report = PreprocessReport()

    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {x.shape}")
    # Accept (samples, leads) and transpose, but only when it is unambiguous.
    if x.shape[0] != N_LEADS and x.shape[1] == N_LEADS:
        x = x.T
    if x.shape[0] != N_LEADS:
        raise ValueError(
            f"expected {N_LEADS} leads, got array of shape {x.shape}; "
            "this record is not a standard 12-lead ECG"
        )

    if fs is None:
        fs = cfg.assumed_fs_hz
        report.fs_assumed = True

    x = window(x, n_samples=n_samples, report=report)
    x, report = denoise(x, fs=fs, cfg=cfg, report=report)
    x = zscore_per_lead(x, eps=cfg.zscore_eps, report=report)
    return np.ascontiguousarray(x, dtype=np.float32), report


__all__ = [
    "PreprocessReport", "design_bandpass", "design_notch",
    "bandpass_response_db", "notch_response_db",
    "denoise", "zscore_per_lead", "window", "preprocess_signal",
]
