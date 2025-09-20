from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd

def plot_betas(betas: pd.DataFrame, title: str = "Rolling Factor Exposures"):
    ax = betas.plot(figsize=(10, 4))
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Beta")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return ax
