from options_pricing import heston_price, merton_price, calibrate_heston, calibrate_merton

S, r = 100, 0.05
STRIKES = [80, 90, 100, 110, 120]
MATURITIES = [0.5, 1.0]


def test_heston_calibration_recovers_known_parameters():
    # Generate a synthetic "market" from known Heston parameters, then check
    # the calibrator recovers a parameter set that reprices the smile tightly.
    true = dict(v0=0.04, kappa=1.5, theta=0.05, sigma=0.4, rho=-0.6)
    quotes = [(Kx, T, heston_price(S, Kx, T, r, option_type="call", **true), "call")
              for T in MATURITIES for Kx in STRIKES]
    result = calibrate_heston(quotes, S, r)
    assert result["rmse"] < 0.05


def test_merton_calibration_recovers_known_parameters():
    true = dict(sigma=0.18, lam=0.6, muJ=-0.12, sigmaJ=0.15)
    quotes = [(Kx, T, merton_price(S, Kx, T, r, option_type="call", **true), "call")
              for T in MATURITIES for Kx in STRIKES]
    result = calibrate_merton(quotes, S, r)
    assert result["rmse"] < 0.05


def test_calibration_accepts_per_quote_rates():
    # Six-tuple quotes carry their own (r, q); the calibrator must honour them.
    true = dict(v0=0.04, kappa=1.5, theta=0.05, sigma=0.4, rho=-0.6)
    quotes = [(Kx, T, heston_price(S, Kx, T, 0.03, option_type="call", q=0.01, **true),
               "call", 0.03, 0.01)
              for T in MATURITIES for Kx in STRIKES]
    result = calibrate_heston(quotes, S)  # no scalar r/q passed; must use per-quote
    assert result["rmse"] < 0.05
