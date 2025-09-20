from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import statsmodels.api as sm


def _as_2d(x: pd.DataFrame | pd.Series) -> pd.DataFrame:
    return x.to_frame() if isinstance(x, pd.Series) else x


def ols(y: pd.Series, X: pd.DataFrame, add_const: bool = True, hac_lags: Optional[int] = None) -> Dict[str, Any]:
    X2 = _as_2d(X).copy()
    if add_const:
        X2 = sm.add_constant(X2)
    model = sm.OLS(y, X2, missing="drop")
    if hac_lags is None:
        res = model.fit()
    else:
        res = model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    params = res.params
    tvals = res.tvalues
    alpha = params.get("const", np.nan)
    if "const" in params.index:
        betas = params.drop("const")
        t_betas = tvals.drop("const")
    else:
        betas = params
        t_betas = tvals
    out = {
        "alpha": float(alpha),
        "betas": betas,
        "tvalues": t_betas,
        "r2": float(res.rsquared),
        "n": int(res.nobs),
        "df_resid": float(res.df_resid),
        "residuals": res.resid,
    }
    return out


def ff3_single(asset_excess: pd.Series, factors: pd.DataFrame, hac_lags: Optional[int] = None):
    use = factors[["Mkt-RF", "SMB", "HML"]].reindex(asset_excess.index).dropna()
    y = asset_excess.reindex(use.index).dropna()
    use = use.reindex(y.index)
    return ols(y, use, add_const=True, hac_lags=hac_lags)


def ff3_batch(assets_excess: pd.DataFrame, factors: pd.DataFrame, hac_lags: Optional[int] = None) -> pd.DataFrame:
    results = []
    for col in assets_excess.columns:
        res = ff3_single(assets_excess[col].dropna(), factors=factors, hac_lags=hac_lags)
        row = {"asset": col, "alpha": res["alpha"], "r2": res["r2"], "n": res["n"]}
        for bname, bval in res["betas"].items():
            row[f"beta_{bname}"] = float(bval)
        results.append(row)
    return pd.DataFrame(results).set_index("asset")
