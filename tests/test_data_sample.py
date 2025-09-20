from factorlab.data import load_ff3_sample

def test_load_ff3_sample_ok():
    df = load_ff3_sample("data/sample_ff3_monthly.csv")
    assert {"Mkt-RF", "SMB", "HML", "RF"} <= set(df.columns)
    assert df.index.is_monotonic_increasing
    assert abs(df["Mkt-RF"].iloc[0]) < 1.0
