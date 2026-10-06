#!/usr/bin/env python3
"""
Tests for the patched open-py-cest-mrf C++ simulator (BMCSimulator).

  Test 1  Block pulses (pypulseq >= 1.5 writes them as 2 samples + a time shape).
          Does the C++ simulator give them their real duration?
          Also: two block pulses that share magnitude/phase shapes and differ
          only in duration (time shape) must not be mixed up.
  Test 2  MT lineshape override (SetMTLineOverride / GetMTLineOverride),
          keyed on (mag_id, phase_id, time_shape_id, freq).
          Stock behaviour unchanged, override actually reaches the matrix,
          and the overridden result matches an independent reference.

Every C++ result is compared against an independent numpy Bloch-McConnell
reference (water + MT pool, rate model, matrix exponentials) that uses the
pulse's TRUE waveform and duration, so it does not share any code or
assumptions with the C++ reader.

Two stages, so they can run in different environments:

    python test_patch.py prepare   # needs pypulseq >= 1.5 (Python >= 3.10): writes the
                                   # .seq files + reference results into test_patch_cases/
    python test_patch.py run       # needs BMCSimulator only (e.g. the Python 3.9 mt-mrf env):
                                   # simulates those .seq files and compares
    python test_patch.py           # both, if one env has both

The cases folder is created next to this script.
"""
import os
import sys
import json
import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(HERE, 'test_patch_cases')

# ----------------------------------------------------------------------------
# Tissue / scanner (keep in sync between reference and C++)
# ----------------------------------------------------------------------------
B0 = 3.0                      # T
GAMMA = 42.577 * 2 * np.pi    # rad/s/uT  (C++ InitScanner convention)
W0 = B0 * GAMMA               # rad/s per ppm
R1W, R2W = 1 / 1.0, 1 / 0.04  # water
F_MT, R1M, T2S, K_CA, DW_MT = 0.08, 1 / 1.0, 10e-6, 30.0, 0.0   # MT pool (k = MT -> water, Hz)
K_AC = K_CA * F_MT            # water -> MT (C++: k_ac = k_ca * f)

def _pp():
    import pypulseq as pp
    return pp, pp.Opts(rf_dead_time=100e-6, rf_ringdown_time=60e-6, B0=B0)


# ----------------------------------------------------------------------------
# Lineshapes (pi included, as in C++ GetMTLineAtCurrentOffset)
# ----------------------------------------------------------------------------
def sl_sum(dw, t2=T2S):
    """C++ InterpolateSuperLorentzianShape: 101-point Riemann sum."""
    u = 0.01 * np.arange(101)
    p = np.abs(3 * u**2 - 1)
    return np.pi * 0.01 * np.sum(np.sqrt(2 / np.pi) * t2 / p * np.exp(-2 * (dw * t2 / p) ** 2))


def g_stock(dw, t2=T2S, w0=W0):
    """Exact replica of C++ GetMTLineAtCurrentOffset (SuperLorentzian branch)."""
    if abs(dw) >= w0:
        return sl_sum(dw, t2)
    px = [-300 - w0, -100 - w0, 100 + w0, 300 + w0]
    py = [sl_sum(x, t2) for x in px]
    p0, p1 = py[1], py[2]
    d0, d1 = 30 * (p0 - py[0]), 30 * (py[3] - p1)
    c = abs((dw - px[1] + 1) / (px[2] - px[1] + 1))
    c2, c3 = c * c, c * c * c
    return (2*c3 - 3*c2 + 1) * p0 + (-2*c3 + 3*c2) * p1 + (c3 - 2*c2 + c) * d0 + (c3 - c2) * d1


def sl_accurate(dw, t2=T2S):
    dw = abs(dw)
    f = lambda u: np.sqrt(2 / np.pi) * t2 / abs(3*u*u - 1) * np.exp(-2 * (dw * t2 / abs(3*u*u - 1)) ** 2)
    return np.pi * quad(f, 0, 1, points=[1 / np.sqrt(3)], limit=500)[0]


_W_TAB = np.geomspace(1.0, 60 / T2S, 500)
_G_TAB = np.array([sl_accurate(w) for w in _W_TAB])


def g_table(w):
    return np.interp(np.abs(w), _W_TAB, _G_TAB, left=_G_TAB[0], right=0.0)


# ----------------------------------------------------------------------------
# Pulse waveform on a uniform raster (handles pypulseq's 2-point block pulses)
# ----------------------------------------------------------------------------
def uniform_waveform(rf, raster=1e-6):
    sig = np.asarray(rf.signal, dtype=complex)
    t = np.asarray(rf.t, dtype=float)
    dur = float(getattr(rf, 'shape_dur', t[-1] - t[0]))
    dts = np.diff(t)
    if sig.size > 2 and np.allclose(dts, dts[0], rtol=1e-6):
        return sig, float(dts[0])
    n = max(int(round(dur / raster)), 1)
    tu = (np.arange(n) + 0.5) * dur / n
    return np.interp(tu, t, sig.real) + 1j * np.interp(tu, t, sig.imag), dur / n


def g_eff(rf):
    """Graham spectral average of the lineshape over the pulse power spectrum (pi included)."""
    sig, dt = uniform_waveform(rf)
    n = 1 << max(16, int(np.ceil(np.log2(8 * sig.size))))
    P = np.abs(np.fft.fft(sig, n)) ** 2
    w = 2 * np.pi * np.fft.fftfreq(n, dt)
    carrier = 2 * np.pi * rf.freq_offset - DW_MT * W0
    return float(np.sum(P * g_table(w + carrier)) / np.sum(P))


# ----------------------------------------------------------------------------
# Independent reference: water (x,y,z) + MT (z), rate model, true waveform
# ----------------------------------------------------------------------------
def reference(rf, g_mode='stock', M=None):
    """Return (water Mz, MT Mz / F_MT) right after the pulse (from equilibrium, or from state M)."""
    if isinstance(rf, (list, tuple)):                     # several pulses back to back
        for r in rf:
            M = reference(r, g_mode, M)[2]
        return M[2], M[3] / F_MT, M
    sig, dt = uniform_waveform(rf)
    dw = 2 * np.pi * rf.freq_offset                       # water on resonance
    if g_mode == 'stock':
        g = g_stock(2 * np.pi * rf.freq_offset - DW_MT * W0)
    elif g_mode == 'eff':
        g = g_eff(rf)
    else:
        g = float(g_mode)
    # collapse runs of identical samples into single steps (block pulse -> 1 step)
    steps, start = [], 0
    for i in range(1, sig.size + 1):
        if i == sig.size or sig[i] != sig[start]:
            steps.append((sig[start], (i - start) * dt)); start = i
    M = np.array([0.0, 0.0, 1.0, F_MT, 1.0]) if M is None else M   # x_w, y_w, z_w, z_mt, 1
    for s, tau in steps:
        w1 = 2 * np.pi * abs(s); ph = np.angle(s)
        wx, wy = w1 * np.cos(ph), w1 * np.sin(ph)
        A = np.array([
            [-R2W,  dw,  -wy,                0,                          0],
            [-dw,  -R2W,  wx,                0,                          0],
            [ wy,  -wx,  -R1W - K_AC,        K_CA,                       R1W],
            [ 0,    0,    K_AC,             -R1M - K_CA - w1**2 * g,     R1M * F_MT],
            [ 0,    0,    0,                 0,                          0]])
        M = expm(A * tau) @ M
    return M[2], M[3] / F_MT, M


# ----------------------------------------------------------------------------
# .seq helpers
# ----------------------------------------------------------------------------
def write_seq(rf, fname):
    pp, SYS = _pp()
    seq = pp.Sequence(system=SYS)
    for r in (rf if isinstance(rf, (list, tuple)) else [rf]):
        seq.add_block(r)
    seq.add_block(pp.make_adc(num_samples=1, duration=10e-6, system=SYS))
    seq.write(fname)
    return fname


def seq_version_and_rf_keys(fname):
    """Return ('major.minor.rev', [(mag_id, phase_id, time_id, freq_hz), ...]) as written in the file."""
    ver, keys, section = {}, [], None
    for line in open(fname):
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        if s.startswith('['):
            section = s; continue
        v = s.split()
        if section == '[VERSION]':
            ver[v[0]] = v[1]
        elif section == '[RF]':
            if len(v) == 7:            # Pulseq <= 1.3: id amp mag phase delay freq phase
                keys.append((int(v[2]), int(v[3]), 0, float(v[5])))
            elif len(v) >= 11:         # Pulseq 1.5: id amp mag phase time center delay fPPM pPPM freq phase use
                keys.append((int(v[2]), int(v[3]), int(v[4]), float(v[9])))
            else:
                raise ValueError(f'unrecognised [RF] row ({len(v)} cols): {s}')
    return f"{ver.get('major')}.{ver.get('minor')}.{ver.get('revision')}", keys


# ----------------------------------------------------------------------------
# C++ simulator
# ----------------------------------------------------------------------------
def cpp_run(fname, override=None):
    from BMCSimulator import BMCSimulator, SimulationParameters, WaterPool, MTPool, SuperLorentzian
    sp = SimulationParameters()
    sp.SetInitialMagnetizationVector(np.array([0.0, 0.0, 1.0, F_MT]))   # [x_w, y_w, z_w, z_mt]
    sp.SetWaterPool(WaterPool(R1W, R2W, 1.0))
    sp.SetMTPool(MTPool(R1M, 1 / T2S, F_MT, DW_MT, K_CA, SuperLorentzian))
    sp.SetNumberOfCESTPools(0)
    sp.InitScanner(B0, 1.0, 0.0, GAMMA)
    sp.SetUseInitMagnetization(True)
    sp.SetMaxNumberOfPulseSamples(200)
    if override is not None:
        for m, p, t, f in seq_version_and_rf_keys(fname)[1]:
            sp.SetMTLineOverride(m, p, t, f, float(override))
    sim = BMCSimulator(sp, fname)
    M = np.asarray(sim.RunSimulation())
    return M[2, 0], M[3, 0] / F_MT


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
def check(name, cpp, ref, tol=0.02):
    ok = all(abs(c - r) <= tol for c, r in zip(cpp[:2], ref[:2]))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:42s} C++ Mz_w {cpp[0]:+.3f}  Mz_mt {cpp[1]:+.3f}"
          f"   | ref Mz_w {ref[0]:+.3f}  Mz_mt {ref[1]:+.3f}")
    return ok


def prepare():
    """Write the test .seq files and the reference results (needs pypulseq >= 1.5)."""
    pp, SYS = _pp()
    os.makedirs(CASES, exist_ok=True)
    cest_flip = 2 * np.pi * 42.577 * 2.0 * 0.1                          # 2 uT for 100 ms
    cest_freq = -3.5 * 42.577 * B0                                      # Hz
    pulses = {
        # Test 1: block pulses -- written by pypulseq 1.5 as 2 samples + time shape
        'block_cest':   pp.make_block_pulse(cest_flip, duration=0.1, freq_offset=cest_freq, system=SYS),
        'block_180':    pp.make_block_pulse(np.pi, duration=1e-3, system=SYS),
        # controls: the same pulses written as fully sampled shapes (1 us raster)
        'arb_cest':     pp.make_arbitrary_rf(np.ones(100000), cest_flip, freq_offset=cest_freq,
                                             system=SYS, return_gz=False),
        'arb_180':      pp.make_arbitrary_rf(np.ones(1000), np.pi, system=SYS, return_gz=False),
        # same amplitude (500 Hz) and same [1 1] shapes, different durations (time shapes)
        'block_180_90': [pp.make_block_pulse(np.pi, duration=1e-3, system=SYS),
                         pp.make_block_pulse(np.pi / 2, duration=0.5e-3, system=SYS)],
        # Test 2: readout sinc
        'sinc_50':      pp.make_sinc_pulse(np.deg2rad(50), duration=0.8e-3, time_bw_product=8,
                                           apodization=0.5, system=SYS),
    }
    for k, rf in pulses.items():
        write_seq(rf, os.path.join(CASES, f'{k}.seq'))
    ver, _ = seq_version_and_rf_keys(os.path.join(CASES, 'block_cest.seq'))
    print(f"pypulseq {getattr(pp, '__version__', '?')} wrote .seq version {ver} into {CASES}")

    ge = g_eff(pulses['sinc_50'])
    r = lambda x: [float(x[0]), float(x[1])]
    ref = {
        'block_cest':   r(reference(pulses['block_cest'])),
        'block_180':    r(reference(pulses['block_180'])),
        'block_180_90': r(reference(pulses['block_180_90'])),
        'sinc_stock':   r(reference(pulses['sinc_50'], 'stock')),
        'sinc_zero':    r(reference(pulses['sinc_50'], 0.0)),
        'sinc_eff':     r(reference(pulses['sinc_50'], 'eff')),
    }
    json.dump({'g_eff_sinc': ge, 'ref': ref, 'seq_version': ver},
              open(os.path.join(CASES, 'reference.json'), 'w'), indent=2)
    print(f"Reference (true waveform): g_stock(0) = {g_stock(0.0)*1e6:.1f} us, g_eff(sinc) = {ge*1e6:.1f} us")
    for k, v in ref.items():
        print(f"  {k:12s} Mz_w {v[0]:+.3f}  Mz_mt/f {v[1]:+.3f}  (MT sat {1-v[1]:.3f})")


def run():
    """Simulate the prepared .seq files with BMCSimulator and compare (no pypulseq needed)."""
    info = json.load(open(os.path.join(CASES, 'reference.json')))
    ref, ge = info['ref'], info['g_eff_sinc']
    f = lambda k: os.path.join(CASES, f'{k}.seq')
    import BMCSimulator
    print(f"BMCSimulator from {BMCSimulator.__file__}; .seq version {info['seq_version']}")
    ok = True
    print("\nTest 1: block pulses keep their real duration")
    ok &= check('100 ms 2 uT block, -3.5 ppm (2-point)',     cpp_run(f('block_cest')),   ref['block_cest'])
    ok &= check('  control: same pulse fully sampled',        cpp_run(f('arb_cest')),     ref['block_cest'])
    ok &= check('1 ms 180 deg block, on-res (2-point)',       cpp_run(f('block_180')),    ref['block_180'])
    ok &= check('  control: same pulse fully sampled',        cpp_run(f('arb_180')),      ref['block_180'])
    ok &= check('180 deg (1 ms) then 90 deg (0.5 ms) block',  cpp_run(f('block_180_90')), ref['block_180_90'])
    print("\nTest 2: MT lineshape override")
    ok &= check('no override == stock',                       cpp_run(f('sinc_50')),          ref['sinc_stock'])
    ok &= check('override g = 0 -> no MT saturation',         cpp_run(f('sinc_50'), 0.0),     ref['sinc_zero'])
    ok &= check(f'override g_eff = {ge*1e6:.1f} us',          cpp_run(f('sinc_50'), ge),      ref['sinc_eff'])
    print("\nALL PASS" if ok else "\nSOME TESTS FAILED")
    return ok


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if stage in ('prepare', 'both'):
        prepare()
    if stage in ('run', 'both'):
        run()
