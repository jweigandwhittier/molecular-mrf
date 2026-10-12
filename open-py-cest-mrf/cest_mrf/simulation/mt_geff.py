#!/usr/bin/env python3
"""
mt_geff.py -- spectrally averaged MT lineshape (g_eff) for every RF pulse in a .seq file.

The stock simulator saturates the semi-solid (MT) pool at  R = w1(t)^2 * g(carrier),
i.e. it evaluates the lineshape at ONE frequency (Graham & Henkelman, JMRI 1997,
Eq. [3]/[8]). For short, broadband, on-resonance pulses that is wrong: on resonance the
super-Lorentzian is singular and the C++ code substitutes a spline value (~98 us at 3T
for T2s = 10 us). The correct rate model averages g over the pulse's power spectrum
(Graham Eq. [4]):

    g_eff = sum_w g(w + carrier) |S(w)|^2 / sum_w |S(w)|^2

This module computes g_eff for each unique pulse and hands the table to the patched
C++ simulator via SimulationParameters.SetMTLineOverride(mag, phase, time, freq, g).

Design choices
- The .seq file is parsed here directly (no pypulseq import): same shape IDs, same
  decompression and same time-shape resampling as the C++ reader, so the waveform we
  average over is exactly the one the simulator plays. Works in Python 3.9.
- g is returned "pi included" (seconds), the same convention as the C++
  MTPool::GetMTLineAtCurrentOffset, so it can be passed straight through.
- Carrier convention matches the C++: carrier = 2*pi*freq + w0*b0_inhom - mt_dw*w0,
  with w0 = b0 * gamma [rad/s per ppm], gamma in rad/s/uT as in the scenario YAML.
- g_eff does not depend on pulse amplitude/flip angle, so one entry covers e.g. the
  readout at every flip angle.

Typical use
    from cest_mrf.simulation.mt_geff import build_overrides, apply_overrides
    table = build_overrides(seq_file, mt_t2=10e-6, lineshape='SuperLorentzian',
                            mt_dw_ppm=0.0, b0=3.0, gamma=267.5153)
    apply_overrides(sp, table)       # sp = BMCSimulator.SimulationParameters

Command line (prints the table, with the stock C++ value for comparison)
    python mt_geff.py my_sequence.seq --t2 10e-6 --b0 3 --gamma 267.5153
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import quad

__all__ = ['RFPulse', 'read_unique_pulses', 'lineshape_table', 'g_eff',
           'build_overrides', 'apply_overrides', 'g_stock_cpp']


# ============================================================================
# 1. .seq parsing (mirrors open-py-cest-mrf ExternalSequence.cpp)
# ============================================================================
@dataclass
class RFPulse:
    mag_id: int
    phase_id: int
    time_id: int
    freq_hz: float          # RF frequency offset [Hz], as written in the file (C++ key)
    waveform: np.ndarray    # complex, unit peak magnitude, on a uniform raster
    dt: float               # raster of `waveform` [s]
    n_events: int = 1       # how many RF events in the sequence use this pulse

    @property
    def key(self):
        return (self.mag_id, self.phase_id, self.time_id, self.freq_hz)

    @property
    def duration(self):
        return self.waveform.size * self.dt


def _decompress(data: np.ndarray, num_samples: int) -> np.ndarray:
    """Pulseq shape decompression (port of pypulseq.decompress_shape, MIT licence).

    Stored uncompressed when len(data) == num_samples; otherwise run-length-encoded
    derivative: a repeated value is followed by (count - 2), then cumulative sum.
    """
    if len(data) == num_samples:
        return np.asarray(data, dtype=float)
    out = np.zeros(num_samples)
    markers = np.where(np.diff(data) == 0.0)[0]
    cp = cu = 0
    for nxt in markers:
        n = nxt - cp
        if n < 0:                       # false positive inside a run
            continue
        if n > 0:
            out[cu:cu + n] = data[cp:nxt]
            cp += n
            cu += n
        rep = int(data[cp + 2] + 2)
        out[cu:cu + rep] = data[cp]
        cp += 3
        cu += rep
    if cp <= len(data) - 1:
        out[cu:] = data[cp:]
    return np.cumsum(out)


def _parse_seq(seq_file: str):
    """Return (version_tuple, definitions, rf_events {id: dict}, shapes {id: ndarray}, block_rf_ids)."""
    version, defs, rf, shapes, block_rf = {}, {}, {}, {}, []
    section, shape_id, shape_n, shape_data = None, None, None, []

    def flush_shape():
        if shape_id is not None:
            shapes[shape_id] = _decompress(np.array(shape_data, dtype=float), shape_n)

    with open(seq_file) as fh:
        for raw in fh:
            s = raw.strip()
            if not s or s.startswith('#'):
                continue
            if s.startswith('['):
                if section == '[SHAPES]':
                    flush_shape()
                    shape_id, shape_data = None, []
                section = s
                continue
            v = s.split()
            if section == '[VERSION]':
                version[v[0]] = int(v[1])
            elif section == '[DEFINITIONS]':
                defs[v[0]] = v[1:]
            elif section == '[BLOCKS]':
                block_rf.append(int(v[2]))
            elif section == '[RF]':
                rid = int(v[0])
                if len(v) == 7:          # Pulseq <= 1.3: id amp mag phase delay freq phase
                    rf[rid] = dict(amp=float(v[1]), mag=int(v[2]), phase=int(v[3]), time=0,
                                   freq=float(v[5]), freq_ppm=0.0)
                elif len(v) == 8:        # Pulseq 1.4: id amp mag phase time delay freq phase
                    rf[rid] = dict(amp=float(v[1]), mag=int(v[2]), phase=int(v[3]), time=int(v[4]),
                                   freq=float(v[6]), freq_ppm=0.0)
                elif len(v) >= 11:       # Pulseq 1.5: id amp mag phase time center delay fPPM pPPM freq phase use
                    rf[rid] = dict(amp=float(v[1]), mag=int(v[2]), phase=int(v[3]), time=int(v[4]),
                                   freq=float(v[9]), freq_ppm=float(v[7]))
                else:
                    raise ValueError(f'{seq_file}: unrecognised [RF] row ({len(v)} columns): {s}')
            elif section == '[SHAPES]':
                if v[0] == 'shape_id':
                    flush_shape()
                    shape_id, shape_n, shape_data = int(v[1]), None, []
                elif v[0] == 'num_samples':
                    shape_n = int(v[1])
                else:
                    shape_data.append(float(v[0]))
        if section == '[SHAPES]':
            flush_shape()
    ver = (version.get('major', 0), version.get('minor', 0), version.get('revision', 0))
    return ver, defs, rf, shapes, block_rf


def _waveform(ev: dict, shapes: dict, rf_raster: float):
    """Complex unit-peak waveform on a uniform raster, as the C++ reader builds it."""
    mag = shapes[ev['mag']]
    ph = shapes[ev['phase']] * 2 * np.pi
    if ev['time'] > 0:
        # time shape: samples sit at the listed times (raster units) -> resample to
        # the raster centres (k + 0.5) by linear interpolation (same as C++ decodeBlock)
        t = shapes[ev['time']]
        n = int(round(t[-1]))
        tq = np.arange(n) + 0.5
        mag, ph = np.interp(tq, t, mag), np.interp(tq, t, ph)
    return mag * np.exp(1j * ph), rf_raster


@lru_cache(maxsize=32)
def _read_unique_pulses_cached(seq_file: str, mtime: float):
    ver, defs, rf, shapes, block_rf = _parse_seq(seq_file)
    if ver >= (1, 4, 0) and ver < (1, 5, 0):
        print(f'mt_geff: WARNING {os.path.basename(seq_file)} is Pulseq {ver}; '
              'the patched C++ reader only parses <=1.3 and 1.5 RF rows.')
    rf_raster = float(defs.get('RadiofrequencyRasterTime', ['1e-6'])[0])
    counts = {}
    for rid in block_rf:
        if rid > 0:
            counts[rid] = counts.get(rid, 0) + 1
    pulses = {}
    for rid, n in counts.items():
        ev = rf[rid]
        if ev['freq_ppm'] != 0.0:
            print(f'mt_geff: WARNING RF event {rid} uses a ppm frequency offset '
                  f'({ev["freq_ppm"]} ppm); the C++ simulator ignores it.')
        key = (ev['mag'], ev['phase'], ev['time'], ev['freq'])
        if key in pulses:
            pulses[key].n_events += n
            continue
        wf, dt = _waveform(ev, shapes, rf_raster)
        pulses[key] = RFPulse(ev['mag'], ev['phase'], ev['time'], ev['freq'], wf, dt, n)
    return list(pulses.values())


def read_unique_pulses(seq_file: str) -> list:
    """Unique RF pulses in a .seq file, keyed exactly as the C++ simulator keys them."""
    seq_file = os.path.abspath(seq_file)
    return _read_unique_pulses_cached(seq_file, os.path.getmtime(seq_file))


# ============================================================================
# 2. Lineshapes (pi included, C++ convention)
# ============================================================================
def _sl_unit_pi(x: float) -> float:
    """pi * dimensionless super-Lorentzian h(x), x = |w| T2s, so that g(w) = T2s * h(|w| T2s)."""
    f = lambda u: np.sqrt(2 / np.pi) / abs(3 * u * u - 1) * np.exp(-2 * (x / abs(3 * u * u - 1)) ** 2)
    return np.pi * quad(f, 0, 1, points=[1 / np.sqrt(3)], limit=500)[0]


@lru_cache(maxsize=1)
def _sl_unit_table():
    """Nodes of the dimensionless super-Lorentzian: log x, log h(x), and H(x) = int_0^x h.

    x from 1e-7 to 20 (h(20) ~ 1e-87, zero beyond). h is stored as log h vs log x: it goes
    like -log x near the magic-angle singularity and like a Gaussian at large x, so
    log-log interpolation is accurate to ~1e-3 or better. H is what g_eff actually uses:
    averaging h over each FFT bin makes the integrable singularity at x = 0 harmless.
    """
    xt = np.geomspace(1e-7, 20.0, 800)
    ht = np.array([_sl_unit_pi(x) for x in xt])
    # cumulative integral; first piece int_0^x0 h ~ x0 * (h(x0) + 1) for a log singularity
    Ht = np.concatenate([[xt[0] * (ht[0] + 1.0)],
                         xt[0] * (ht[0] + 1.0) + np.cumsum(0.5 * (ht[1:] + ht[:-1]) * np.diff(xt))])
    return xt, np.log(ht), Ht


class Lineshape:
    """MT lineshape g(w) [s, pi included] for one T2, in the C++ convention.

    g(w)                 point value
    g.cell_mean(lo, hi)  mean of g over [lo, hi] (rad/s), exact for the singular bin
    Both lineshapes scale as g(w; T2) = T2 * h(w T2), so one dimensionless table
    (computed once, ~0.3 s) serves every T2 in a dictionary.
    """

    def __init__(self, kind: str, t2: float):
        self.kind, self.t2 = kind, float(t2)
        if kind == 'superlorentzian':
            self._xt, self._lh, self._Ht = _sl_unit_table()
            self._lx = np.log(self._xt)

    def __call__(self, w):
        x = np.abs(np.asarray(w, dtype=float)) * self.t2
        if self.kind == 'lorentzian':
            return self.t2 / (1 + x ** 2)
        x = np.maximum(x, 1e-300)
        return self.t2 * np.exp(np.interp(np.log(x), self._lx, self._lh, left=self._lh[0], right=-np.inf))

    def _H(self, x):
        """Odd antiderivative of h (dimensionless)."""
        if self.kind == 'lorentzian':
            return np.arctan(x)
        ax = np.abs(x)
        return np.sign(x) * np.interp(ax, np.concatenate([[0.0], self._xt]),
                                      np.concatenate([[0.0], self._Ht]), right=self._Ht[-1])

    def cell_mean(self, lo, hi):
        xl, xh = np.asarray(lo) * self.t2, np.asarray(hi) * self.t2
        return (self._H(xh) - self._H(xl)) / (xh - xl) * self.t2


@lru_cache(maxsize=64)
def lineshape_table(lineshape: str, t2: float) -> Lineshape:
    """Return the Lineshape object (callable g(w)) for the MT pool lineshape."""
    ls = lineshape.lower()
    if ls not in ('lorentzian', 'superlorentzian'):
        raise ValueError(f"lineshape '{lineshape}' not supported by the C++ simulator "
                         "(only 'Lorentzian' and 'SuperLorentzian')")
    return Lineshape(ls, t2)


def g_stock_cpp(dw: float, t2: float, w0: float) -> float:
    """Replica of the stock C++ GetMTLineAtCurrentOffset (SuperLorentzian), for comparison."""
    def ssum(x):
        u = 0.01 * np.arange(101)
        p = np.abs(3 * u ** 2 - 1)
        return np.pi * 0.01 * np.sum(np.sqrt(2 / np.pi) * t2 / p * np.exp(-2 * (x * t2 / p) ** 2))
    if abs(dw) >= w0:
        return ssum(dw)
    px = [-300 - w0, -100 - w0, 100 + w0, 300 + w0]
    py = [ssum(x) for x in px]
    d0, d1 = 30 * (py[1] - py[0]), 30 * (py[3] - py[2])
    c = abs((dw - px[1] + 1) / (px[2] - px[1] + 1))
    return ((2*c**3 - 3*c**2 + 1) * py[1] + (-2*c**3 + 3*c**2) * py[2]
            + (c**3 - 2*c**2 + c) * d0 + (c**3 - c**2) * d1)


# ============================================================================
# 3. Spectral average
# ============================================================================
def g_eff(waveform: np.ndarray, dt: float, carrier: float, g) -> float:
    """Power-spectrum-weighted mean of g (Graham Eq. [4], normalised by pulse energy).

    waveform : complex B1 shape (any scaling -- it cancels)
    dt       : its raster [s]
    carrier  : offset of the pulse carrier from the MT pool centre [rad/s]
    g        : Lineshape from lineshape_table() (uses the bin-averaged g, so the
               log singularity of the super-Lorentzian at 0 Hz is handled exactly),
               or any callable g(w) [s] (point values at the bin centres)
    """
    wf = np.asarray(waveform, dtype=complex)
    if not np.any(wf):
        return 0.0
    n = 1 << max(16, int(np.ceil(np.log2(8 * wf.size))))      # zero-pad: fine frequency grid
    P = np.abs(np.fft.fft(wf, n)) ** 2
    w = 2 * np.pi * np.fft.fftfreq(n, dt) + carrier
    if hasattr(g, 'cell_mean'):
        half = np.pi / (n * dt)                                 # half a bin [rad/s]
        gw = g.cell_mean(w - half, w + half)
    else:
        gw = g(w)
    return float(np.sum(P * gw) / np.sum(P))


# ============================================================================
# 4. Table for the C++ simulator
# ============================================================================
@lru_cache(maxsize=256)
def _build_cached(seq_file, mtime, mt_t2, lineshape, mt_dw_ppm, b0, gamma, b0_inhom, max_offset_hz):
    w0 = b0 * gamma                                   # rad/s per ppm
    g = lineshape_table(lineshape, float(mt_t2))
    table = []
    for p in _read_unique_pulses_cached(seq_file, mtime):
        if max_offset_hz is not None and abs(p.freq_hz) > max_offset_hz:
            continue                                  # keep stock behaviour (e.g. CEST saturation)
        carrier = 2 * np.pi * p.freq_hz + w0 * b0_inhom - mt_dw_ppm * w0
        table.append((p.mag_id, p.phase_id, p.time_id, p.freq_hz, g_eff(p.waveform, p.dt, carrier, g)))
    return tuple(table)


def build_overrides(seq_file: str, mt_t2: float, lineshape: str, mt_dw_ppm: float,
                    b0: float, gamma: float, b0_inhom: float = 0.0,
                    max_offset_hz: float | None = None) -> list:
    """[(mag_id, phase_id, time_id, freq_hz, g_eff), ...] for SetMTLineOverride.

    mt_t2         : MT pool T2 [s]  (= 1 / r2)
    lineshape     : 'SuperLorentzian' or 'Lorentzian'
    mt_dw_ppm     : MT pool centre [ppm] (scenario 'dw' of mt_pool)
    b0, gamma     : scenario b0 [T] and gamma [rad/s/uT]
    b0_inhom      : [ppm]; negligible for kHz-wide on-resonance pulses
    max_offset_hz : only override pulses with |RF freq offset| <= this (relative to water).
                    None (default) overrides every pulse, which is what you want: for long
                    off-resonance saturation pulses g_eff equals the single-tone value to
                    <1%, so CEST behaviour is unchanged. Careful with thresholds: -3.5 ppm
                    is only 447 Hz at 3 T. Use e.g. 50 Hz to restrict to on-resonance pulses.
    """
    seq_file = os.path.abspath(seq_file)
    return list(_build_cached(seq_file, os.path.getmtime(seq_file), float(mt_t2), str(lineshape),
                              float(mt_dw_ppm), float(b0), float(gamma), float(b0_inhom),
                              None if max_offset_hz is None else float(max_offset_hz)))


def apply_overrides(sp, table) -> None:
    """Replace all MT lineshape overrides on a BMCSimulator.SimulationParameters object."""
    sp.ClearMTLineOverrides()
    for m, p, t, f, gv in table:
        sp.SetMTLineOverride(int(m), int(p), int(t), float(f), float(gv))


# ============================================================================
# CLI: inspect a sequence
# ============================================================================
def _main():
    import argparse
    ap = argparse.ArgumentParser(description='Print g_eff for every unique RF pulse in a .seq file.')
    ap.add_argument('seq_file')
    ap.add_argument('--t2', type=float, default=10e-6, help='MT pool T2 [s]')
    ap.add_argument('--lineshape', default='SuperLorentzian')
    ap.add_argument('--dw', type=float, default=0.0, help='MT pool centre [ppm]')
    ap.add_argument('--b0', type=float, default=3.0, help='[T]')
    ap.add_argument('--gamma', type=float, default=267.5153, help='[rad/s/uT]')
    a = ap.parse_args()

    w0 = a.b0 * a.gamma
    g = lineshape_table(a.lineshape, a.t2)
    pulses = read_unique_pulses(a.seq_file)
    print(f'{len(pulses)} unique RF pulses in {a.seq_file}   (T2s = {a.t2*1e6:g} us, {a.lineshape}, '
          f'MT dw = {a.dw} ppm, B0 = {a.b0} T)')
    print(f"{'mag':>4} {'ph':>4} {'time':>5} {'freq [Hz]':>10} {'dur [ms]':>9} {'events':>7} "
          f"{'g_eff [us]':>11} {'stock [us]':>11} {'stock/eff':>9}")
    for p in sorted(pulses, key=lambda p: (abs(p.freq_hz), p.duration)):
        carrier = 2 * np.pi * p.freq_hz - a.dw * w0
        ge = g_eff(p.waveform, p.dt, carrier, g)
        gs = g_stock_cpp(carrier, a.t2, w0) if a.lineshape.lower() == 'superlorentzian' else float('nan')
        print(f'{p.mag_id:4d} {p.phase_id:4d} {p.time_id:5d} {p.freq_hz:10.1f} {p.duration*1e3:9.3f} '
              f'{p.n_events:7d} {ge*1e6:11.2f} {gs*1e6:11.2f} {gs/ge:9.2f}')


if __name__ == '__main__':
    _main()
