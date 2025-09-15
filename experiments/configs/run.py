"""
Experiment runner
=================
Usage:  poetry run finexp <dataset=spx_daily> <model=msm> <risk=var99>

The CLI parses arguments via Typer, then boots Hydra to load the
corresponding YAML cascade into a single dict called *cfg*.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import rich
import typer
from omegaconf import DictConfig, OmegaConf

app = typer.Typer(add_completion=False, rich_help_panel="root")

CONFIG_PATH = Path(__file__).resolve().parent

@app.command()
def main(
    dataset: str = typer.Argument("spx_daily"),
    model: str = typer.Argument("msm"),
    risk: str = typer.Argument("var99"),
) -> None:
    with hydra.initialize_config_dir(str(CONFIG_PATH)):
        cfg: DictConfig = hydra.compose(
            config_name="config",
            overrides=[f"dataset={dataset}", f"model={model}", f"risk={risk}"],
        )
    rich.print(OmegaConf.to_yaml(cfg))

    # 1) load data
    from fractalfinance.io import load_csv, load_yahoo

    data_path = Path(cfg.dataset.path)
    if data_path.exists():
        series = load_csv(data_path)
    elif getattr(cfg.dataset, "symbol", None):
        series = load_yahoo(
            cfg.dataset.symbol, cfg.dataset.start, cfg.dataset.get("end")
        )
        series.to_frame(name="close").to_csv(data_path, index_label="timestamp")
    else:
        raise FileNotFoundError(data_path)

    # 2) fit model and forecast variance
    if cfg.model.name == "msm":
        from fractalfinance.models import msm_fit
        mdl = msm_fit(series, K=cfg.model.K)
        sigma2 = np.array([mdl.sigma2])       # stub forecast
    elif cfg.model.name == "garch":
        from fractalfinance.models import GARCH
        sigma2 = GARCH().fit(series).forecast(cfg.model.h)
    else:
        raise NotImplementedError

    # 3) risk metric
    from fractalfinance.risk import var_gaussian
    VaR = var_gaussian(np.sqrt(sigma2), cfg.risk.p)
    rich.print(f"[cyan]VaR[/cyan] = {VaR}")

if __name__ == "__main__":
    app()
