import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    import yfinance as yf
except ImportError:
    yf = None


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


@dataclass
class Config:
    ticker: str = "SPY"
    vix_ticker: str = "^VIX"
    start_date: str = "2012-01-01"
    end_date: str = "2025-01-01"

    price_csv: Optional[str] = None
    vix_csv: Optional[str] = None

    seq_len: int = 20
    vol_window: int = 20
    mean_window: int = 20

    batch_size: int = 32
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.2
    lr: float = 1e-3
    epochs: int = 30
    patience: int = 6

    train_ratio: float = 0.70
    val_ratio: float = 0.15

    kappa: float = 3.0
    theta: float = 0.04
    xi: float = 0.35
    rho: float = -0.7
    dt: float = 1 / 252

    heston_horizon_days: int = 21
    n_scenarios: int = 1000
    s0: float = 1.0

    seed: int = 42


cfg = Config()


def _read_close_csv(path: str, out_name: str) -> pd.Series:
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError(f"{path} must contain a Date column")

    close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    if close_col not in df.columns:
        raise ValueError(f"{path} must contain Adj Close or Close")

    series = df[["Date", close_col]].copy()
    series["Date"] = pd.to_datetime(series["Date"])
    series = series.rename(columns={close_col: out_name}).set_index("Date").sort_index()
    return series[out_name].astype(float)


def _extract_close_from_yf(frame: pd.DataFrame, out_name: str) -> pd.Series:
    if frame.empty:
        raise ValueError("Downloaded data is empty")

    if isinstance(frame.columns, pd.MultiIndex):
        level0 = frame.columns.get_level_values(0)
        for candidate in ("Close", "Adj Close"):
            if candidate in level0:
                col = [c for c in frame.columns if c[0] == candidate][0]
                return frame[col].rename(out_name).astype(float)

    for candidate in ("Close", "Adj Close"):
        if candidate in frame.columns:
            return frame[candidate].rename(out_name).astype(float)

    raise ValueError("Expected Close or Adj Close column in downloaded data")


def load_market_data(cfg: Config) -> pd.DataFrame:
    if cfg.price_csv is not None:
        close = _read_close_csv(cfg.price_csv, "close")
    else:
        if yf is None:
            raise ImportError("yfinance is not installed. Either install it or provide price_csv/vix_csv paths.")
        asset_df = yf.download(
            cfg.ticker,
            start=cfg.start_date,
            end=cfg.end_date,
            auto_adjust=True,
            progress=False,
        )
        close = _extract_close_from_yf(asset_df, "close")

    if cfg.vix_csv is not None:
        vix = _read_close_csv(cfg.vix_csv, "vix")
    else:
        if yf is None:
            vix = pd.Series(index=close.index, dtype=float, name="vix")
        else:
            try:
                vix_df = yf.download(
                    cfg.vix_ticker,
                    start=cfg.start_date,
                    end=cfg.end_date,
                    auto_adjust=True,
                    progress=False,
                )
                vix = _extract_close_from_yf(vix_df, "vix")
            except Exception:
                vix = pd.Series(index=close.index, dtype=float, name="vix")

    df = pd.concat([close, vix], axis=1).sort_index()
    df = df.dropna(subset=["close"])
    return df


def build_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    data = df.copy()
    data["log_price"] = np.log(data["close"])
    data["log_return"] = data["log_price"].diff()

    data["realized_vol"] = (
        data["log_return"].rolling(cfg.vol_window).std(ddof=0) * np.sqrt(252)
    )
    data["realized_vol"] = data["realized_vol"].clip(lower=1e-8)
    data["log_realized_vol"] = np.log(data["realized_vol"])

    data["rolling_mean_return"] = data["log_return"].rolling(cfg.mean_window).mean()

    if "vix" not in data.columns or data["vix"].notna().sum() == 0:
        data["vix"] = data["realized_vol"] * 100.0
    else:
        data["vix"] = data["vix"].ffill().bfill()

    data["log_vix"] = np.log(data["vix"].clip(lower=1e-8))

    # Target: next-day log realized volatility
    data["target_log_vol"] = data["log_realized_vol"].shift(-1)

    data = data.dropna().copy()
    return data


class FeatureScaler:
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0, ddof=0)
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler must be fit before transform")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def temporal_split_dates(index: pd.Index, train_ratio: float, val_ratio: float) -> Tuple[pd.Timestamp, pd.Timestamp]:
    n = len(index)
    train_end_pos = max(int(n * train_ratio) - 1, 0)
    val_end_pos = max(int(n * (train_ratio + val_ratio)) - 1, train_end_pos + 1)
    val_end_pos = min(val_end_pos, n - 1)
    return index[train_end_pos], index[val_end_pos]


def make_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
    X_list, y_list, idx_list = [], [], []

    values = df[feature_cols].to_numpy(dtype=np.float32)
    targets = df[target_col].to_numpy(dtype=np.float32)

    for i in range(seq_len, len(df)):
        X_list.append(values[i - seq_len : i])
        y_list.append(targets[i])
        idx_list.append(df.index[i])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    idx = pd.Index(idx_list)
    return X, y, idx


class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


class VolatilityLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait = 0

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(cfg.epochs):
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model(X_batch)
                val_losses.append(criterion(preds, y_batch).item())

        avg_train = float(np.mean(train_losses))
        avg_val = float(np.mean(val_losses))

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        print(f"Epoch {epoch + 1:02d}/{cfg.epochs} | Train: {avg_train:.6f} | Val: {avg_val:.6f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_state)
    return model, history


def predict_model(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        preds = model(X_tensor).cpu().numpy().flatten()
    return preds


def simulate_heston_terminal_log_returns(
    mu_annual: float,
    v0: float,
    cfg: Config,
) -> np.ndarray:
    v = np.full(cfg.n_scenarios, max(v0, 1e-8), dtype=float)
    log_s = np.zeros(cfg.n_scenarios, dtype=float)
    rho_scale = np.sqrt(max(1.0 - cfg.rho ** 2, 1e-8))

    for _ in range(cfg.heston_horizon_days):
        z1 = np.random.normal(size=cfg.n_scenarios)
        z2 = np.random.normal(size=cfg.n_scenarios)

        dW_price = np.sqrt(cfg.dt) * z1
        dW_var = np.sqrt(cfg.dt) * (cfg.rho * z1 + rho_scale * z2)

        v_prev = np.maximum(v, 1e-8)
        log_s += (mu_annual - 0.5 * v_prev) * cfg.dt + np.sqrt(v_prev) * dW_price

        v = v_prev + cfg.kappa * (cfg.theta - v_prev) * cfg.dt + cfg.xi * np.sqrt(v_prev) * dW_var
        v = np.maximum(v, 1e-8)

    return log_s


def scenario_summary(scenarios: np.ndarray) -> Dict[str, float]:
    q05 = float(np.quantile(scenarios, 0.05))
    cvar05 = float(scenarios[scenarios <= q05].mean())
    return {
        "scenario_mean": float(np.mean(scenarios)),
        "scenario_std": float(np.std(scenarios, ddof=0)),
        "scenario_var_5": q05,
        "scenario_cvar_5": cvar05,
    }


def generate_heston_scenarios_static(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []

    for dt_idx, row in df.iterrows():
        sigma = float(row["realized_vol"])
        mu_annual = float(row["rolling_mean_return"] * 252.0)
        v0 = max(sigma ** 2, 1e-8)

        scen = simulate_heston_terminal_log_returns(mu_annual, v0, cfg)
        rows.append({"date": dt_idx, **scenario_summary(scen)})

    return pd.DataFrame(rows).set_index("date")


def generate_heston_scenarios_lstm(
    base_df: pd.DataFrame,
    pred_log_vol: np.ndarray,
    pred_index: pd.Index,
    cfg: Config,
) -> pd.DataFrame:
    rows = []
    pred_series = pd.Series(pred_log_vol, index=pred_index)

    for dt_idx, log_vol_pred in pred_series.items():
        row = base_df.loc[dt_idx]
        sigma_pred = float(np.exp(log_vol_pred))
        mu_annual = float(row["rolling_mean_return"] * 252.0)
        v0 = max(sigma_pred ** 2, 1e-8)

        scen = simulate_heston_terminal_log_returns(mu_annual, v0, cfg)
        rows.append(
            {
                "date": dt_idx,
                "pred_sigma": sigma_pred,
                **scenario_summary(scen),
            }
        )

    return pd.DataFrame(rows).set_index("date")


def evaluate_predictions(y_true_log_vol: np.ndarray, y_pred_log_vol: np.ndarray) -> Dict[str, float]:
    err = y_true_log_vol - y_pred_log_vol
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true_log_vol - np.mean(y_true_log_vol)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


def plot_training_history(history: dict) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Training History")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_predicted_vs_actual(index, y_true_log_vol, y_pred_log_vol) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(index, np.exp(y_true_log_vol), label="Actual Vol")
    plt.plot(index, np.exp(y_pred_log_vol), label="Predicted Vol")
    plt.title("Actual vs Predicted Realized Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_scenario_comparison(static_df: pd.DataFrame, lstm_df: pd.DataFrame) -> None:
    common_idx = static_df.index.intersection(lstm_df.index)
    static_df = static_df.loc[common_idx]
    lstm_df = lstm_df.loc[common_idx]

    plt.figure(figsize=(12, 4))
    plt.plot(common_idx, static_df["scenario_cvar_5"], label="Static Heston CVaR(5%)")
    plt.plot(common_idx, lstm_df["scenario_cvar_5"], label="LSTM-Heston CVaR(5%)")
    plt.title("Scenario Tail Risk Comparison")
    plt.xlabel("Date")
    plt.ylabel("Terminal log-return CVaR")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def run_pipeline(cfg: Config) -> Dict[str, object]:
    raw_df = load_market_data(cfg)
    feature_df = build_features(raw_df, cfg)

    feature_cols = [
        "log_return",
        "log_realized_vol",
        "rolling_mean_return",
        "log_vix",
    ]
    target_col = "target_log_vol"

    train_end_date, val_end_date = temporal_split_dates(feature_df.index, cfg.train_ratio, cfg.val_ratio)

    scaler = FeatureScaler()
    scaler.fit(feature_df.loc[:train_end_date, feature_cols].to_numpy(dtype=np.float32))

    scaled_df = feature_df.copy()
    scaled_df[feature_cols] = scaler.transform(scaled_df[feature_cols].to_numpy(dtype=np.float32))

    X_all, y_all, idx_all = make_sequences(scaled_df, feature_cols, target_col, cfg.seq_len)

    train_mask = idx_all <= train_end_date
    val_mask = (idx_all > train_end_date) & (idx_all <= val_end_date)
    test_mask = idx_all > val_end_date

    X_train, y_train, idx_train = X_all[train_mask], y_all[train_mask], idx_all[train_mask]
    X_val, y_val, idx_val = X_all[val_mask], y_all[val_mask], idx_all[val_mask]
    X_test, y_test, idx_test = X_all[test_mask], y_all[test_mask], idx_all[test_mask]

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError("One of the train/val/test sequence sets is empty. Check the date range or split ratios.")

    print("Train shape:", X_train.shape, y_train.shape)
    print("Val shape:", X_val.shape, y_val.shape)
    print("Test shape:", X_test.shape, y_test.shape)

    train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(SeqDataset(X_val, y_val), batch_size=cfg.batch_size, shuffle=False)

    model = VolatilityLSTM(
        input_size=len(feature_cols),
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(DEVICE)

    model, history = train_model(model, train_loader, val_loader, cfg, DEVICE)

    y_pred_test = predict_model(model, X_test, DEVICE)
    metrics = evaluate_predictions(y_test.flatten(), y_pred_test)

    print("\nVolatility Forecast Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    plot_training_history(history)
    plot_predicted_vs_actual(idx_test, y_test.flatten(), y_pred_test)

    base_test_df = feature_df.loc[idx_test].copy()

    static_scenarios = generate_heston_scenarios_static(base_test_df, cfg)
    lstm_scenarios = generate_heston_scenarios_lstm(base_test_df, y_pred_test, idx_test, cfg)

    plot_scenario_comparison(static_scenarios, lstm_scenarios)

    results = base_test_df.copy()
    results["actual_log_vol"] = y_test.flatten()
    results["pred_log_vol"] = y_pred_test
    results["actual_vol"] = np.exp(results["actual_log_vol"])
    results["pred_vol"] = np.exp(results["pred_log_vol"])

    results = results.join(static_scenarios, how="left", rsuffix="_static")
    results = results.join(lstm_scenarios, how="left", rsuffix="_lstm")

    return {
        "raw_df": raw_df,
        "feature_df": feature_df,
        "model": model,
        "history": history,
        "metrics": metrics,
        "results": results,
        "static_scenarios": static_scenarios,
        "lstm_scenarios": lstm_scenarios,
        "feature_cols": feature_cols,
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
    }


if __name__ == "__main__":
    output = run_pipeline(cfg)

    print("\nResults head:")
    print(
        output["results"][
            [
                "actual_vol",
                "pred_vol",
                "scenario_cvar_5",
                "scenario_cvar_5_lstm",
            ]
        ].head()
    )
