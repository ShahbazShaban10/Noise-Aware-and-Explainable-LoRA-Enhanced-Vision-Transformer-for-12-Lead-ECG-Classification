"""Denoising, normalisation and windowing.

These tests assert on the *measured filter response*, not on whether a function was called.
"Ran a Butterworth" is not the claim being made; "suppresses baseline wander below 0.5 Hz
and powerline at 50 Hz while preserving the 1-35 Hz band where P-QRS-T lives" is.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from numpy.fft import rfft, rfftfreq

from ecgvit.config import N_LEADS, N_SAMPLES, PreprocessConfig
from ecgvit.preprocess import (bandpass_response_db, denoise, design_bandpass,
                               design_notch, notch_response_db, preprocess_signal,
                               window, zscore_per_lead)

FS = 500.0
CFG = PreprocessConfig()

PASSBAND = [1.0, 5.0, 10.0, 20.0, 30.0, 35.0]
STOPBAND_LOW = [0.01, 0.05, 0.1]
STOPBAND_HIGH = [60.0, 80.0, 120.0]


# ---------------------------------------------------------------------------
# Filter response
# ---------------------------------------------------------------------------
def test_bandpass_preserves_the_diagnostic_band():
    db = bandpass_response_db(FS, CFG, PASSBAND)
    for hz, d in zip(PASSBAND, db):
        assert d >= -3.0, f"band-pass attenuates {hz} Hz by {d:.2f} dB; P-QRS-T lives here"


def test_bandpass_rejects_baseline_wander():
    db = bandpass_response_db(FS, CFG, STOPBAND_LOW)
    for hz, d in zip(STOPBAND_LOW, db):
        assert d <= -20.0, f"baseline wander at {hz} Hz only attenuated {d:.2f} dB"


def test_bandpass_rejects_high_frequency_noise():
    db = bandpass_response_db(FS, CFG, STOPBAND_HIGH)
    for hz, d in zip(STOPBAND_HIGH, db):
        assert d <= -20.0, f"EMG-band noise at {hz} Hz only attenuated {d:.2f} dB"


def test_bandpass_edges_are_where_the_manuscript_says():
    """-6 dB at the corner is the zero-phase signature: -3 dB one-pass, doubled by
    filtfilt. Finding -3 dB here would mean the filter was applied only once."""
    lo, hi = bandpass_response_db(FS, CFG, [CFG.bandpass_low_hz, CFG.bandpass_high_hz])
    assert -7.5 < lo < -4.5, f"0.5 Hz corner at {lo:.2f} dB"
    assert -7.5 < hi < -4.5, f"40 Hz corner at {hi:.2f} dB"


def test_notch_attenuates_powerline_and_spares_neighbours():
    at50, at45, at55 = notch_response_db(FS, CFG, [50.0, 45.0, 55.0])
    assert at50 <= -20.0, f"50 Hz only attenuated {at50:.2f} dB"
    assert at45 >= -3.0 and at55 >= -3.0, (
        f"notch is too wide: {at45:.2f} dB at 45 Hz, {at55:.2f} dB at 55 Hz. "
        f"Q={CFG.notch_q} should keep it narrow."
    )


def test_filter_design_rejects_impossible_parameters():
    with pytest.raises(ValueError):
        design_bandpass(100.0, 0.5, 60.0, 4)          # 60 Hz above Nyquist for fs=100
    with pytest.raises(ValueError):
        design_bandpass(FS, 40.0, 0.5, 4)             # inverted band
    with pytest.raises(ValueError):
        design_notch(80.0, 50.0, 30.0)                # notch above Nyquist


def test_notch_is_skipped_not_faked_when_above_nyquist():
    """At 100 Hz sampling, a 50 Hz notch is undefined. It must be skipped and reported,
    never designed at w0 >= 1."""
    x = np.random.default_rng(0).standard_normal((N_LEADS, 2000))
    _, rep = denoise(x, fs=100.0, cfg=CFG)
    assert rep.notch_applied is False
    assert "Nyquist" in (rep.notch_skipped_reason or "")


# ---------------------------------------------------------------------------
# End-to-end noise suppression on real fixture files
# ---------------------------------------------------------------------------
def _power_at(sig: np.ndarray, hz: float, fs: float = FS) -> float:
    f = rfftfreq(sig.shape[-1], 1.0 / fs)
    return float(np.abs(rfft(sig))[int(np.argmin(np.abs(f - hz)))])


def test_denoising_removes_injected_noise_from_fixture_records(fixture_corpus):
    import wfdb

    rec = wfdb.rdrecord(str(fixture_corpus / "FX00000"))
    raw = rec.p_signal.T
    clean, rep = preprocess_signal(raw, fs=FS)

    assert rep.bandpass_applied and rep.notch_applied and rep.zerophase

    for hz in (0.15, 0.30):   # baseline wander injected by the generator
        before, after = _power_at(raw[1], hz), _power_at(clean[1], hz)
        assert after < before * 0.2, (
            f"baseline wander at {hz} Hz went {before:.1f} -> {after:.1f}; "
            "expected at least a 5x reduction"
        )

    before, after = _power_at(raw[1], 50.0), _power_at(clean[1], 50.0)
    assert after < before * 0.05, (
        f"50 Hz powerline went {before:.1f} -> {after:.1f}; expected at least 20x"
    )


def test_denoising_preserves_the_qrs_band(fixture_corpus):
    """The point of the filter is to remove noise, not signal. QRS energy around 10 Hz
    must survive -- relatively more than the noise bands, after normalisation."""
    import wfdb

    raw = wfdb.rdrecord(str(fixture_corpus / "FX00000")).p_signal.T
    clean, _ = preprocess_signal(raw, fs=FS)
    qrs = _power_at(clean[1], 10.0)
    wander = _power_at(clean[1], 0.15)
    powerline = _power_at(clean[1], 50.0)
    assert qrs > 5 * wander, f"QRS band ({qrs:.1f}) not dominant over wander ({wander:.1f})"
    assert qrs > 5 * powerline, f"QRS band ({qrs:.1f}) not dominant over 50 Hz ({powerline:.1f})"


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def test_zscore_is_per_lead_not_global():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((N_LEADS, 1000))
    x = x * np.arange(1, N_LEADS + 1)[:, None] + np.arange(N_LEADS)[:, None] * 10.0
    z = zscore_per_lead(x)
    assert np.allclose(z.mean(axis=1), 0.0, atol=1e-9)
    assert np.allclose(z.std(axis=1), 1.0, atol=1e-6), (
        "each lead must be standardised independently; a global z-score would leave "
        "per-lead std proportional to the original scaling"
    )


def test_zscore_handles_a_flat_lead_without_nan():
    x = np.zeros((N_LEADS, 500))
    x[3] = np.linspace(-1, 1, 500)
    from ecgvit.preprocess import PreprocessReport

    rep = PreprocessReport()
    z = zscore_per_lead(x, eps=1e-8, report=rep)
    assert np.isfinite(z).all(), "a constant lead must not produce NaN or inf"
    assert 3 not in rep.flat_leads and 0 in rep.flat_leads


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def test_window_leaves_correct_length_untouched():
    x = np.random.default_rng(2).standard_normal((N_LEADS, N_SAMPLES))
    assert np.array_equal(window(x), x)


def test_window_centre_crops_longer_records():
    from ecgvit.preprocess import PreprocessReport

    x = np.tile(np.arange(6000, dtype=float), (N_LEADS, 1))
    rep = PreprocessReport()
    y = window(x, N_SAMPLES, rep)
    assert y.shape == (N_LEADS, N_SAMPLES)
    assert rep.cropped_samples == 1000
    assert y[0, 0] == 500, "crop must be centred, not taken from the start"


def test_window_zero_pads_shorter_records_symmetrically():
    from ecgvit.preprocess import PreprocessReport

    x = np.ones((N_LEADS, 4000))
    rep = PreprocessReport()
    y = window(x, N_SAMPLES, rep)
    assert y.shape == (N_LEADS, N_SAMPLES)
    assert rep.padded_samples == 1000
    assert y[0, 0] == 0 and y[0, -1] == 0 and y[0, N_SAMPLES // 2] == 1


def test_preprocess_output_contract(fixture_corpus):
    import wfdb

    raw = wfdb.rdrecord(str(fixture_corpus / "FX00003")).p_signal.T
    x, rep = preprocess_signal(raw, fs=FS)
    assert x.shape == (N_LEADS, N_SAMPLES)
    assert x.dtype == np.float32
    assert np.isfinite(x).all()
    assert isinstance(rep.to_dict(), dict)
    assert json.dumps(rep.to_dict())  # must be serialisable for preprocessing_report.json


def test_preprocess_accepts_samples_by_leads_orientation():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((N_LEADS, N_SAMPLES))
    xa, _ = preprocess_signal(a, fs=FS)
    xb, _ = preprocess_signal(a.T, fs=FS)
    assert np.allclose(xa, xb, atol=1e-5)


def test_preprocess_rejects_non_12_lead_input():
    with pytest.raises(ValueError, match="12 leads"):
        preprocess_signal(np.zeros((8, N_SAMPLES)), fs=FS)


def test_missing_sampling_rate_is_assumed_and_flagged():
    """The manuscript asks for this assumption to be marked rather than left implicit."""
    x = np.random.default_rng(4).standard_normal((N_LEADS, N_SAMPLES))
    _, rep = preprocess_signal(x, fs=None)
    assert rep.fs_assumed is True
    assert rep.fs_hz == CFG.assumed_fs_hz


def test_preprocessing_is_deterministic(fixture_corpus):
    import wfdb

    raw = wfdb.rdrecord(str(fixture_corpus / "FX00010")).p_signal.T
    a, _ = preprocess_signal(raw, fs=FS)
    b, _ = preprocess_signal(raw, fs=FS)
    assert np.array_equal(a, b)
