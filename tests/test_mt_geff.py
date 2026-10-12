#!/usr/bin/env python3
"""
test_mt_geff.py -- checks for cest_mrf/simulation/mt_geff.py

Uses the .seq files and reference.json written by `python test_patch.py prepare`.

  Test 1  The .seq decoder reproduces pypulseq's waveforms (block pulses with time
          shapes, arbitrary shapes, sinc). Skipped if pypulseq >= 1.5 can't be imported
          (e.g. Python 3.9).
  Test 2  Known numbers: sinc g_eff matches the reference built from pypulseq's own
          waveform; a block pulse matches the analytic sinc^2 integral and does not depend
          on FFT zero padding; a 2-point block and the same pulse fully sampled agree;
          a long off-resonance CEST block reproduces the stock C++ value; the
          max_offset_hz filter leaves CEST pulses alone.
  Test 3  End to end: build_overrides -> apply_overrides -> BMCSimulator on sinc_50.seq
          matches the independent reference with g = g_eff (needs the patched build).

Run from molecular-mrf/tests:   python test_mt_geff.py
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(HERE, 'test_patch_cases')
sys.path.insert(0, HERE)

try:
    from cest_mrf.simulation import mt_geff
except ImportError:                                   # not pip-installed: use the source tree
    sys.path.insert(0, os.path.join(HERE, '..', 'open-py-cest-mrf'))
    from cest_mrf.simulation import mt_geff

# tissue/scanner constants shared with test_patch.py (no pypulseq needed to import them)
from test_patch import B0, GAMMA, W0, R1W, R2W, F_MT, R1M, T2S, K_CA, DW_MT, g_stock, check

SEQ = lambda k: os.path.join(CASES, f'{k}.seq')
CASE_NAMES = ['block_cest', 'arb_cest', 'block_180', 'arb_180', 'block_180_90', 'sinc_50']


def report(ok, name, detail):
    print(f"  {'PASS' if ok else 'FAIL'}  {name:46s} {detail}")
    return ok


# ----------------------------------------------------------------------------
# Test 1: decoder vs pypulseq
# ----------------------------------------------------------------------------
def test_decoder():
    print('Test 1: .seq decoder vs pypulseq')
    try:
        import pypulseq as pp
        from test_patch import uniform_waveform
    except Exception as e:                            # TypeError on Python 3.9, ImportError if absent
        print(f'  SKIP  pypulseq not usable here ({type(e).__name__}: {e})')
        return True
    ok = True
    for name in CASE_NAMES:
        seq = pp.Sequence()
        seq.read(SEQ(name))
        ref = []
        for b in seq.block_events:
            rf = seq.get_block(b).rf
            if rf is not None:
                sig, dt = uniform_waveform(rf)
                ref.append((sig / np.max(np.abs(sig)), dt, rf.freq_offset))
        mine = mt_geff.read_unique_pulses(SEQ(name))
        # match each pypulseq pulse to a decoded pulse with the same length/frequency
        for sig, dt, f in ref:
            cand = [p for p in mine if p.waveform.size == sig.size and abs(p.freq_hz - f) < 1e-6]
            err = min((np.max(np.abs(p.waveform - sig)) for p in cand), default=np.inf)
            good = err < 1e-4 and all(abs(p.dt - dt) < 1e-12 for p in cand)
            ok &= report(good, f'{name}: {sig.size} samples, {f:+.1f} Hz', f'max |diff| = {err:.1e}')
        n_events = sum(p.n_events for p in mine)
        ok &= report(n_events == len(ref), f'{name}: event count', f'{n_events} decoded vs {len(ref)} pypulseq')
    return ok


# ----------------------------------------------------------------------------
# Test 2: known numbers
# ----------------------------------------------------------------------------
def _g_of(name, i=0, carrier=None):
    p = sorted(mt_geff.read_unique_pulses(SEQ(name)), key=lambda p: -p.duration)[i]
    g = mt_geff.lineshape_table('SuperLorentzian', T2S)
    c = 2 * np.pi * p.freq_hz - DW_MT * W0 if carrier is None else carrier
    return mt_geff.g_eff(p.waveform, p.dt, c, g), p


def test_numbers():
    print('\nTest 2: known numbers')
    info = json.load(open(os.path.join(CASES, 'reference.json')))
    ok = True

    ge, _ = _g_of('sinc_50')
    ref = info['g_eff_sinc']
    ok &= report(abs(ge / ref - 1) < 0.01, 'sinc g_eff vs pypulseq-waveform reference',
                 f'{ge*1e6:.2f} vs {ref*1e6:.2f} us')

    # block pulse: |S(f)|^2 = T sinc^2(fT) analytically -> continuous Graham integral
    from scipy.integrate import quad
    from test_patch import sl_accurate
    T = 1e-3
    f = lambda fr: T * np.sinc(fr * T) ** 2 * sl_accurate(2 * np.pi * fr, T2S)
    edges = np.arange(0, 301) / T
    exact = 2 * sum(quad(f, a, b, limit=200)[0] for a, b in zip(edges[:-1], edges[1:]))
    gb, p = _g_of('block_180')
    ok &= report(abs(gb / exact - 1) < 2e-3, '1 ms block vs analytic sinc^2 integral',
                 f'{gb*1e6:.2f} vs {exact*1e6:.2f} us')

    # result must not depend on the FFT grid (the 0 Hz bin holds the log singularity)
    g = mt_geff.lineshape_table('SuperLorentzian', T2S)
    pads = [mt_geff.g_eff(np.concatenate([p.waveform, np.zeros(p.waveform.size * k)]), p.dt, 0.0, g)
            for k in (0, 63)]
    ok &= report(abs(pads[1] / pads[0] - 1) < 1e-4, 'independent of zero padding (x1 vs x64)',
                 f'{pads[0]*1e6:.3f} vs {pads[1]*1e6:.3f} us')

    gs = mt_geff.g_stock_cpp(0.0, T2S, W0)
    ok &= report(abs(gs - g_stock(0.0)) < 1e-12, 'g_stock_cpp(0) replica', f'{gs*1e6:.2f} us (stock/eff = {gs/ge:.2f})')

    ga, _ = _g_of('arb_180')
    ok &= report(abs(gb / ga - 1) < 1e-3, '2-point block == fully sampled (1 ms 180)',
                 f'{gb*1e6:.2f} vs {ga*1e6:.2f} us')

    g180, _ = _g_of('block_180_90', 0)
    g90, _ = _g_of('block_180_90', 1)
    ok &= report(g90 < g180 and abs(g180 / gb - 1) < 1e-3, 'shared [1 1] shapes, different time shapes',
                 f'1 ms {g180*1e6:.2f} us, 0.5 ms {g90*1e6:.2f} us')

    gc, p = _g_of('block_cest')
    gsc = g_stock(2 * np.pi * p.freq_hz - DW_MT * W0)
    ok &= report(abs(gc / gsc - 1) < 0.02, '100 ms CEST block at -3.5 ppm == stock',
                 f'{gc*1e6:.2f} vs {gsc*1e6:.2f} us')

    # Lorentzian sanity: long pulse far off resonance -> g_eff = T2/(1+(dw T2)^2)
    gl = mt_geff.lineshape_table('Lorentzian', 1e-3)
    dw = 2 * np.pi * p.freq_hz
    gle = mt_geff.g_eff(p.waveform, p.dt, dw, gl)
    exact = 1e-3 / (1 + (dw * 1e-3) ** 2)
    ok &= report(abs(gle / exact - 1) < 0.02, 'Lorentzian, long off-res block', f'{gle*1e6:.3f} vs {exact*1e6:.3f} us')

    kw = dict(mt_t2=T2S, lineshape='SuperLorentzian', mt_dw_ppm=DW_MT, b0=B0, gamma=GAMMA)
    n_on = len(mt_geff.build_overrides(SEQ('block_cest'), max_offset_hz=50, **kw))
    n_all = len(mt_geff.build_overrides(SEQ('block_cest'), **kw))
    ok &= report(n_on == 0 and n_all == 1, 'max_offset_hz=50 skips the CEST pulse', f'{n_on} vs {n_all} entries')

    # the dimensionless table scales correctly to another T2
    from test_patch import sl_accurate
    g12 = mt_geff.lineshape_table('SuperLorentzian', 12e-6)
    err = max(abs(g12(w) / sl_accurate(w, 12e-6) - 1) for w in [30.0, 2e3, 5e4, 2e5])
    ok &= report(err < 1e-3, 'SL table at T2s = 12 us vs direct quadrature', f'max rel err {err:.1e}')

    try:
        mt_geff.lineshape_table('Gaussian', T2S)
        ok &= report(False, 'unsupported lineshape raises', 'no error')
    except ValueError:
        ok &= report(True, 'unsupported lineshape raises', 'ValueError')
    return ok


# ----------------------------------------------------------------------------
# Test 3: end to end through the patched C++
# ----------------------------------------------------------------------------
def _sim(fname, table=None):
    from BMCSimulator import BMCSimulator, SimulationParameters, WaterPool, MTPool, SuperLorentzian
    sp = SimulationParameters()
    sp.SetInitialMagnetizationVector(np.array([0.0, 0.0, 1.0, F_MT]))
    sp.SetWaterPool(WaterPool(R1W, R2W, 1.0))
    sp.SetMTPool(MTPool(R1M, 1 / T2S, F_MT, DW_MT, K_CA, SuperLorentzian))
    sp.SetNumberOfCESTPools(0)
    sp.InitScanner(B0, 1.0, 0.0, GAMMA)
    sp.SetUseInitMagnetization(True)
    sp.SetMaxNumberOfPulseSamples(200)
    if table is not None:
        mt_geff.apply_overrides(sp, table)
    M = np.asarray(BMCSimulator(sp, fname).RunSimulation())
    return M[2, 0], M[3, 0] / F_MT


def test_end_to_end():
    print('\nTest 3: build_overrides -> apply_overrides -> BMCSimulator')
    try:
        import BMCSimulator
    except ImportError as e:
        print(f'  SKIP  BMCSimulator not importable ({e})')
        return True
    if not hasattr(BMCSimulator.SimulationParameters, 'SetMTLineOverride'):
        print('  FAIL  this BMCSimulator build has no SetMTLineOverride (rebuild sim_lib)')
        return False
    ref = json.load(open(os.path.join(CASES, 'reference.json')))['ref']
    kw = dict(mt_t2=T2S, lineshape='SuperLorentzian', mt_dw_ppm=DW_MT, b0=B0, gamma=GAMMA)
    table = mt_geff.build_overrides(SEQ('sinc_50'), **kw)
    ok = report(len(table) == 1, 'override table for sinc_50.seq', f'{table}')
    ok &= check('stock (no overrides)',      _sim(SEQ('sinc_50')),        ref['sinc_stock'])
    ok &= check('mt_geff overrides',         _sim(SEQ('sinc_50'), table), ref['sinc_eff'])
    # overriding a long off-resonance CEST block changes nothing (g_eff == single tone)
    t_cest = mt_geff.build_overrides(SEQ('block_cest'), **kw)
    ok &= check('CEST block overridden == stock', _sim(SEQ('block_cest'), t_cest), ref['block_cest'])
    # and clearing works: apply an empty table after a non-empty one
    ok &= check('empty table == stock',      _sim(SEQ('sinc_50'), []),    ref['sinc_stock'])
    return ok


if __name__ == '__main__':
    ok = test_decoder()
    ok &= test_numbers()
    ok &= test_end_to_end()
    print('\nALL PASS' if ok else '\nSOME TESTS FAILED')
    sys.exit(0 if ok else 1)
