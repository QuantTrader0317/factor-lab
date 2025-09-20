import click
from .data import (
    load_ff3_sample,
    fetch_prices_yf,
    compute_simple_returns,
    resample_to_month_end,
    excess_returns,
    join_factors_and_assets,
)
from .models import ff3_batch


@click.group()
def app():
    """factor-lab CLI"""


@app.command()
@click.option("--tickers", multiple=True, required=True, help="One or more tickers, e.g., AAPL MSFT")
@click.option("--start", type=str, required=True)
@click.option("--end", type=str, required=True)
@click.option("--freq", type=click.Choice(["D", "M"]), default="M", show_default=True)
@click.option("--ff-source", type=click.Choice(["sample", "download"]), default="sample", show_default=True)
@click.option("--ff-sample-path", type=str, default="data/sample_ff3_monthly.csv", show_default=True)
@click.option("--hac", type=int, default=None, help="HAC/Newey–West lags (None = OLS).")
def ff3(tickers, start, end, freq, ff_source, ff_sample_path, hac):
    """Run FF3 regressions for tickers over a period."""
    prices = fetch_prices_yf(tickers, start=start, end=end, freq=freq)
    rets = compute_simple_returns(prices)
    if freq == "M":
        rets = resample_to_month_end(rets)
    if ff_source == "sample":
        factors = load_ff3_sample(ff_sample_path)
        if freq == "D":
            click.echo("Sample FF3 is monthly; resampling returns to monthly.")
            rets = resample_to_month_end(rets)
    else:
        raise click.UsageError("ff-source=download not implemented in offline demo. Use --ff-source sample.")
    assets_excess = excess_returns(rets, factors["RF"])
    joined = join_factors_and_assets(assets_excess, factors)
    fac_cols = ["Mkt-RF", "SMB", "HML"]
    results = ff3_batch(joined[assets_excess.columns], joined[fac_cols], hac_lags=hac)
    click.echo(results.to_string())
