from options_pricing import heston_price, calibrate_heston

S, r = 100, 0.05


def test_calibration_recovers_known_parameters():
    # Generate a synthetic "market" from known Heston parameters, then check
    # the calibrator recovers a parameter set that reprices the smile tightly.
    true = dict(v0=0.04, kappa=1.5, theta=0.05, sigma=0.4, rho=-0.6)
    strikes = [80, 90, 100, 110, 120]
    maturities = [0.5, 1.0]
    quotes = []
    for T in maturities:
        for Kx in strikes:
            price = heston_price(S, Kx, T, r, option_type="call", **true)
            quotes.append((Kx, T, price, "call"))

    result = calibrate_heston(quotes, S, r)
    # The fit should reprice the surface to well under a cent RMSE.
    assert result["rmse"] < 0.05
