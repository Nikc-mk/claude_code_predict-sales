"""
train.py — скрипт обучения FT-Transformer модели прогнозирования продаж.

Запуск:
    python train.py --data data/sales.csv
    python train.py --data data/sales.csv --epochs 50 --batch_size 128

Что делает:
  1. Загружает и преобразует данные (ETL)
  2. Создаёт TabularDataset, разбивает по времени (train/val)
  3. Обучает FTTransformerModel с HuberLoss
  4. Считает SMAPE и MAE на валидации
  5. Сохраняет лучшую модель + скейлер + энкодер категорий
"""

import argparse
import csv
import os
import time
from calendar import monthrange
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from build_features import (
    load_raw_data, pivot_to_wide, build_calendar_df,
    get_month_sales, get_month_total, get_rolling_sum,
    get_cumulative_to_day, get_previous_month_total,
)
from config import CATEGORIES, PATHS, TRAIN_CONFIG
from dataset import TabularDataset
from model import build_model, save_model


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (%)."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


# ---------------------------------------------------------------------------
# Один проход по DataLoader
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler_mean: float = 0.0,
    scaler_std: float = 1.0,
) -> tuple[float, float, float, float, float]:
    """
    Выполняет один проход (train или eval).

    Returns
    -------
    avg_loss : float
    smape_sales : float (%)
    mae_sales : float
    smape_profit : float (%)
    mae_profit : float
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_true_sales, all_pred_sales = [], []
    all_true_profit, all_pred_profit = [], []

    with torch.set_grad_enabled(is_train):
        for X, cat_ids, y in loader:
            X = X.to(device)
            cat_ids = cat_ids.to(device)
            y = y.to(device)  # (B, 2)

            preds = model(X, cat_ids)  # (B, 2)
            loss = criterion(preds, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(y)

            # Переводим в исходный масштаб для метрик (таргеты не нормализованы)
            p = preds.detach().cpu().numpy()
            t = y.detach().cpu().numpy()
            all_pred_sales.append(p[:, 0] * scaler_std + scaler_mean)
            all_true_sales.append(t[:, 0] * scaler_std + scaler_mean)
            all_pred_profit.append(p[:, 1])
            all_true_profit.append(t[:, 1])

    pred_sales = np.concatenate(all_pred_sales)
    true_sales = np.concatenate(all_true_sales)
    pred_profit = np.concatenate(all_pred_profit)
    true_profit = np.concatenate(all_true_profit)
    avg_loss = total_loss / len(loader.dataset)
    return (
        avg_loss,
        smape(true_sales, pred_sales),
        mae(true_sales, pred_sales),
        smape(true_profit, pred_profit),
        mae(true_profit, pred_profit),
    )


# ---------------------------------------------------------------------------
# Визуализация
# ---------------------------------------------------------------------------

def plot_training_curves(log_rows: list[dict], artifacts_dir: str) -> str:
    """
    Строит и сохраняет графики обучения:
      - Train Loss vs Val Loss
      - Val SMAPE (%)
      - Val MAE
      - Learning Rate

    Возвращает путь к сохранённому PNG.
    """
    epochs = [r["epoch"] for r in log_rows]
    train_loss = [r["train_loss"] for r in log_rows]
    val_loss = [r["val_loss"] for r in log_rows]
    val_smape = [r["val_smape"] for r in log_rows]
    val_mae = [r["val_mae"] for r in log_rows]
    lr = [r["lr"] for r in log_rows]

    # Эпоха лучшей модели
    best_epoch = log_rows[int(np.argmin(val_loss))]["epoch"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Training curves — FT-Transformer Sales Forecast", fontsize=14, fontweight="bold")

    # --- Loss ---
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, label="Train Loss", color="#4C72B0", linewidth=1.5)
    ax.plot(epochs, val_loss, label="Val Loss", color="#DD8452", linewidth=1.5)
    ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label=f"Best epoch {best_epoch}")
    ax.set_title("Huber Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- SMAPE ---
    ax = axes[0, 1]
    ax.plot(epochs, val_smape, color="#55A868", linewidth=1.5)
    best_smape = min(val_smape)
    ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
    ax.axhline(best_smape, color="#55A868", linestyle=":", linewidth=1,
               label=f"Best SMAPE = {best_smape:.2f}%")
    ax.set_title("Val SMAPE (%)")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- MAE ---
    ax = axes[1, 0]
    ax.plot(epochs, val_mae, color="#C44E52", linewidth=1.5)
    best_mae = min(val_mae)
    ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
    ax.axhline(best_mae, color="#C44E52", linestyle=":", linewidth=1,
               label=f"Best MAE = {best_mae:,.0f}")
    ax.set_title("Val MAE")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Learning Rate ---
    ax = axes[1, 1]
    ax.plot(epochs, lr, color="#8172B3", linewidth=1.5)
    ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(artifacts_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return plot_path


# ---------------------------------------------------------------------------
# Визуализация MAPE по дням валидационных месяцев
# ---------------------------------------------------------------------------

def plot_val_mape_by_day(
    wide_df: pd.DataFrame,
    categories: list[str],
    val_months: list[tuple[int, int]],
    model: nn.Module,
    scaler,
    artifacts_dir: str,
    device: torch.device,
    wide_create_df: pd.DataFrame | None = None,
    blind_test_months: list[tuple[int, int]] | None = None,
    wide_profit_df: pd.DataFrame | None = None,
) -> str:
    """
    Строит график MAPE по дням месяца для валидационных и слепых тестовых месяцев.

    val_months        — синие графики (валидация)
    blind_test_months — оранжевые графики (слепой тест)

    Ряд 0: MAPE по продажам (sales)
    Ряд 1: MAPE по прибыли (profit)

    Сохраняет: artifacts/val_mape_by_day.png
    """
    if blind_test_months is None:
        blind_test_months = []

    # Все месяцы: сначала val, потом blind_test
    all_months = [(ym, "val") for ym in val_months] + [(ym, "blind_test") for ym in blind_test_months]
    n_months = len(all_months)
    if n_months == 0:
        return ""

    cat2idx = {c: i for i, c in enumerate(categories)}
    has_profit = wide_profit_df is not None

    fig, axes = plt.subplots(2, n_months, figsize=(6 * n_months, 10), squeeze=False)
    fig.suptitle(
        "MAPE по дням месяца (агрегировано по всем категориям)",
        fontsize=13, fontweight="bold"
    )

    model.eval()
    all_records = []

    for col, ((year, month), split) in enumerate(all_months):
        ax_sales  = axes[0, col]
        ax_profit = axes[1, col]
        _, n_days = monthrange(year, month)
        cal = build_calendar_df(year, month)

        line_color = "#4C72B0" if split == "val" else "#DD8452"

        # Фактический итог за весь месяц (сумма по всем категориям)
        actual_total = sum(get_month_total(wide_df, year, month, cat) for cat in categories)
        if abs(actual_total) < 1:
            ax_sales.set_title(f"{year}-{month:02d}  (нет данных)")
            ax_profit.set_title(f"{year}-{month:02d}  (нет данных)")
            continue

        actual_profit_total = (
            sum(get_month_total(wide_profit_df, year, month, cat) for cat in categories)
            if has_profit else 0.0
        )

        days, mapes, predicted_totals = [], [], []
        mapes_profit, predicted_profit_totals = [], []

        # --- Предвычисления (не меняются с t) ---
        month_sales_cache    = {cat: get_month_sales(wide_df, year, month, cat)     for cat in categories}
        lastyear_ms_cache    = {cat: get_month_sales(wide_df, year - 1, month, cat) for cat in categories}
        lastyear_total_cache = {cat: float(lastyear_ms_cache[cat].sum())             for cat in categories}
        prev_total_cache     = {cat: get_previous_month_total(wide_df, year, month, cat) for cat in categories}

        if has_profit:
            month_profit_cache        = {cat: get_month_sales(wide_profit_df, year, month, cat)     for cat in categories}
            lastyear_profit_ms_cache  = {cat: get_month_sales(wide_profit_df, year - 1, month, cat) for cat in categories}
            prev_profit_cache         = {cat: get_previous_month_total(wide_profit_df, year, month, cat) for cat in categories}

        # Векторизованные скользящие суммы по всем дням месяца сразу
        wcd = wide_create_df if wide_create_df is not None else wide_df
        roll_start   = pd.Timestamp(year=year, month=month, day=1) - pd.Timedelta(days=27)
        month_end_ts = pd.Timestamp(year=year, month=month, day=n_days)
        df_win  = wide_df.loc[roll_start:month_end_ts]
        wcd_win = wcd.loc[roll_start:month_end_ts]
        roll7    = df_win.rolling(7,  min_periods=1).sum()
        roll14   = df_win.rolling(14, min_periods=1).sum()
        roll28   = df_win.rolling(28, min_periods=1).sum()
        c_roll3  = wcd_win.rolling(3,  min_periods=1).sum()
        c_roll7  = wcd_win.rolling(7,  min_periods=1).sum()
        c_roll10 = wcd_win.rolling(10, min_periods=1).sum()
        c_roll14 = wcd_win.rolling(14, min_periods=1).sum()

        if has_profit:
            pft_win   = wide_profit_df.loc[roll_start:month_end_ts]
            p_roll7   = pft_win.rolling(7,  min_periods=1).sum()
            p_roll14  = pft_win.rolling(14, min_periods=1).sum()
            p_roll28  = pft_win.rolling(28, min_periods=1).sum()

        print(f"  {year}-{month:02d} ({split})...", end="", flush=True)

        for t in range(1, n_days):  # t=1..T-1 (последний день — только факт)
            day_date = pd.Timestamp(year=year, month=month, day=t)
            if day_date not in cal.index:
                continue
            cal_row = cal.loc[day_date]

            # Признаки для всех категорий сразу
            X_list, cumulative_total = [], 0.0
            cumulative_profit_total = 0.0
            for cat in categories:
                month_sales = month_sales_cache[cat]
                cumulative  = float(month_sales.iloc[:t].sum())
                cumulative_total += cumulative

                cumul_profit = 0.0
                if has_profit:
                    cumul_profit = float(month_profit_cache[cat].iloc[:t].sum())
                    cumulative_profit_total += cumul_profit

                s7   = float(roll7.at[day_date,   cat]) if day_date in roll7.index   else 0.0
                s14  = float(roll14.at[day_date,  cat]) if day_date in roll14.index  else 0.0
                s28  = float(roll28.at[day_date,  cat]) if day_date in roll28.index  else 0.0
                cs3  = float(c_roll3.at[day_date,  cat]) if day_date in c_roll3.index  else 0.0
                cs7  = float(c_roll7.at[day_date,  cat]) if day_date in c_roll7.index  else 0.0
                cs10 = float(c_roll10.at[day_date, cat]) if day_date in c_roll10.index else 0.0
                cs14 = float(c_roll14.at[day_date, cat]) if day_date in c_roll14.index else 0.0

                lastyear_cumul       = float(lastyear_ms_cache[cat].iloc[:t].sum())
                lastyear_month_total = lastyear_total_cache[cat]
                prev_month_total     = prev_total_cache[cat]

                # Profit rolling sums
                if has_profit:
                    p7   = float(p_roll7.at[day_date,  cat]) if day_date in p_roll7.index  else 0.0
                    p14  = float(p_roll14.at[day_date, cat]) if day_date in p_roll14.index else 0.0
                    p28  = float(p_roll28.at[day_date, cat]) if day_date in p_roll28.index else 0.0
                    lastyear_profit_cumul = float(lastyear_profit_ms_cache[cat].iloc[:t].sum())
                    lastyear_profit_total = float(lastyear_profit_ms_cache[cat].sum())
                    prev_profit_total     = prev_profit_cache[cat]
                else:
                    p7 = p14 = p28 = 0.0
                    lastyear_profit_cumul = lastyear_profit_total = prev_profit_total = 0.0

                feats = [
                    float(cal_row["days_left"]),
                    float(cal_row["work_days_left"]),
                    float(cal_row["days_passed"]),
                    float(cal_row["work_days_passed"]),
                    float(cal_row["is_weekend"]),
                    cumulative,
                    s7, s14, s28,
                    cs3, cs7, cs10, cs14,
                    lastyear_cumul,
                    lastyear_month_total,
                    prev_month_total,
                    float(year),
                    float(year * 12 + month),
                    float(day_date.toordinal()),
                    float(cal_row["month_sin"]),
                    float(cal_row["month_cos"]),
                    cumul_profit,
                    p7, p14, p28,
                    lastyear_profit_cumul,
                    lastyear_profit_total,
                    prev_profit_total,
                ]
                X_list.append(feats)

            X_scaled = scaler.transform(np.array(X_list, dtype=np.float32))
            cat_ids = np.array([cat2idx[c] for c in categories], dtype=np.int64)

            with torch.no_grad():
                preds = model(
                    torch.tensor(X_scaled, dtype=torch.float32).to(device),
                    torch.tensor(cat_ids, dtype=torch.long).to(device),
                ).cpu().numpy()  # (n_cats, 2)

            # Sales MAPE
            predicted_total = cumulative_total + float(preds[:, 0].sum())
            mape_t = abs(actual_total - predicted_total) / abs(actual_total) * 100
            days.append(t)
            mapes.append(mape_t)
            predicted_totals.append(predicted_total)

            # Profit MAPE
            if has_profit and abs(actual_profit_total) > 1:
                predicted_profit_total = cumulative_profit_total + float(preds[:, 1].sum())
                mape_profit_t = abs(actual_profit_total - predicted_profit_total) / abs(actual_profit_total) * 100
            else:
                predicted_profit_total = 0.0
                mape_profit_t = 0.0
            mapes_profit.append(mape_profit_t)
            predicted_profit_totals.append(predicted_profit_total)

        mean_mape = np.mean(mapes) if mapes else 0.0
        mean_mape_profit = np.mean(mapes_profit) if mapes_profit else 0.0
        print(f" sales MAPE={mean_mape:.1f}%  profit MAPE={mean_mape_profit:.1f}%")

        split_label = "val" if split == "val" else "blind test"

        # --- Ряд 0: Sales MAPE ---
        ax_sales.plot(days, mapes, color=line_color, linewidth=1.5, marker="o", markersize=3)
        ax_sales.axhline(
            mean_mape, color="gray", linestyle="--", linewidth=1,
            label=f"Средняя MAPE = {mean_mape:.1f}%"
        )
        ax_sales.set_title(f"{year}-{month:02d}  ({split_label})\nПродажи, факт={actual_total:,.0f}")
        ax_sales.set_xlabel("День месяца")
        ax_sales.set_ylabel("MAPE (%)")
        ax_sales.legend(fontsize=9)
        ax_sales.grid(True, alpha=0.3)

        # --- Ряд 1: Profit MAPE ---
        ax_profit.plot(days, mapes_profit, color=line_color, linewidth=1.5, marker="o", markersize=3)
        ax_profit.axhline(
            mean_mape_profit, color="gray", linestyle="--", linewidth=1,
            label=f"Средняя MAPE = {mean_mape_profit:.1f}%"
        )
        ax_profit.set_title(f"{year}-{month:02d}  ({split_label})\nПрибыль, факт={actual_profit_total:,.0f}")
        ax_profit.set_xlabel("День месяца")
        ax_profit.set_ylabel("MAPE (%)")
        ax_profit.legend(fontsize=9)
        ax_profit.grid(True, alpha=0.3)

        all_records.append(pd.DataFrame({
            "split": split,
            "year": year,
            "month": month,
            "day": days,
            "actual_total": actual_total,
            "predicted_total": [round(v, 0) for v in predicted_totals],
            "mape_pct": [round(v, 4) for v in mapes],
            "actual_profit_total": actual_profit_total,
            "predicted_profit_total": [round(v, 0) for v in predicted_profit_totals],
            "mape_profit_pct": [round(v, 4) for v in mapes_profit],
        }))

    plt.tight_layout()
    plot_path = os.path.join(artifacts_dir, "val_mape_by_day.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    if all_records:
        csv_path = os.path.join(artifacts_dir, "val_mape_by_day.csv")
        pd.concat(all_records, ignore_index=True).to_csv(csv_path, index=False, encoding="utf-8-sig")

    return plot_path


# ---------------------------------------------------------------------------
# Главная функция обучения
# ---------------------------------------------------------------------------

def train(
    data_path: str,
    epochs: int = TRAIN_CONFIG["epochs"],
    batch_size: int = TRAIN_CONFIG["batch_size"],
    lr: float = TRAIN_CONFIG["lr"],
    weight_decay: float = TRAIN_CONFIG["weight_decay"],
    val_months_count: int = TRAIN_CONFIG["val_months_count"],
    blind_test_months_count: int = TRAIN_CONFIG["blind_test_months_count"],
    patience: int = TRAIN_CONFIG["patience"],
    huber_delta: float = TRAIN_CONFIG["huber_delta"],
    artifacts_dir: str = PATHS["artifacts_dir"],
) -> None:
    os.makedirs(artifacts_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- 1. Загрузка и ETL ---
    print("Loading data...")
    df = load_raw_data(data_path)
    print(f"  Raw rows: {len(df)}, date range: {df['date'].min().date()} – {df['date'].max().date()}")

    categories = sorted(CATEGORIES) if CATEGORIES else sorted(df["category"].unique().tolist())
    print(f"  Categories ({len(categories)}): {categories[:5]}{'...' if len(categories) > 5 else ''}")

    wide_df = pivot_to_wide(df, categories=categories)
    wide_create_df = pivot_to_wide(df, categories=categories, value_col="create_sale")
    wide_profit_df = pivot_to_wide(df, categories=categories, value_col="profit")
    print(f"  Wide shape: {wide_df.shape}")

    # --- 2. Датасет ---
    print("Building datasets...")
    t0 = time.time()
    train_ds, val_ds, blind_test_ds = TabularDataset.train_val_split(
        wide_df, categories, wide_create_df=wide_create_df, wide_profit_df=wide_profit_df,
        val_months_count=val_months_count, blind_test_months_count=blind_test_months_count,
    )
    print(f"  Built in {time.time() - t0:.1f}s")

    # Сохраняем скейлер и энкодер категорий
    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
    train_ds.save_scaler(scaler_path)
    print(f"  Scaler saved → {scaler_path}")

    cat_encoder_path = os.path.join(artifacts_dir, "category_encoder.pkl")
    import pickle
    with open(cat_encoder_path, "wb") as f:
        pickle.dump(categories, f)
    print(f"  Category encoder saved → {cat_encoder_path}")

    # Масштаб таргета для денормализации при метриках
    # Таргет масштабируется скейлером только косвенно (как часть числовых признаков)
    # Здесь мы работаем с СЫРЫМ таргетом (не нормализованным), поэтому:
    scaler_mean, scaler_std = 0.0, 1.0  # таргет не нормализуется отдельно

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    has_val = len(val_ds) > 0

    # --- 3. Модель ---
    model = build_model(n_categories=len(categories)).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    criterion = nn.HuberLoss(delta=huber_delta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience
    )

    # --- 4. Цикл обучения ---
    best_val_loss = float("inf")
    best_model_path = os.path.join(artifacts_dir, "model_best.pt")
    log_path = os.path.join(artifacts_dir, "training_log.csv")

    log_rows = []
    val_col_header = "Val Loss" if has_val else "Tr.Loss*"
    print(f"\n{'Epoch':>5}  {'Train Loss':>11}  {val_col_header:>10}  {'Sales SMAPE':>12}  {'Sales MAE':>12}  {'Profit SMAPE':>13}  {'Profit MAE':>12}  {'LR':>10}")
    print("-" * 105)

    for epoch in range(1, epochs + 1):
        t_loss, *_ = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler_mean, scaler_std
        )
        if has_val:
            v_loss, v_smape, v_mae, v_smape_p, v_mae_p = run_epoch(
                model, val_loader, criterion, None, device, scaler_mean, scaler_std
            )
        else:
            v_loss, v_smape, v_mae, v_smape_p, v_mae_p = t_loss, 0.0, 0.0, 0.0, 0.0

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(v_loss)

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": round(t_loss, 6),
                "val_loss": round(v_loss, 6),
                "val_smape": round(v_smape, 4),
                "val_mae": round(v_mae, 2),
                "val_smape_profit": round(v_smape_p, 4),
                "val_mae_profit": round(v_mae_p, 2),
                "lr": current_lr,
            }
        )

        print(
            f"{epoch:>5}  {t_loss:>11.4f}  {v_loss:>10.4f}  "
            f"{v_smape:>11.2f}%  {v_mae:>12.0f}  "
            f"{v_smape_p:>12.2f}%  {v_mae_p:>12.0f}  {current_lr:>10.2e}"
        )

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            save_model(
                model,
                best_model_path,
                extra={
                    "categories": categories,
                    "best_val_loss": best_val_loss,
                    "best_epoch": epoch,
                    "val_smape": v_smape,
                    "val_mae": v_mae,
                    "val_smape_profit": v_smape_p,
                    "val_mae_profit": v_mae_p,
                },
            )
            loss_label = "val_loss" if has_val else "train_loss"
            print(f"  ✓ Saved best model ({loss_label}={v_loss:.4f})")

    # --- 5. Сохраняем лог ---
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\nTraining log saved → {log_path}")
    print(f"Best model saved  → {best_model_path}")
    print(f"Best val loss: {best_val_loss:.4f}")

    # --- 6. Визуализация кривых обучения ---
    plot_path = plot_training_curves(log_rows, artifacts_dir)
    print(f"Training plot saved → {plot_path}")

    # --- 7. Визуализация MAPE по дням валидационных месяцев ---
    print("\nBuilding val MAPE plot...")
    # Вычисляем списки месяцев для графика (та же логика, что в train_val_split)
    ym_all = sorted(
        {(d.year, d.month) for d in wide_df.index},
        key=lambda x: (x[0], x[1]),
    )
    today = date.today()
    _, last_day = monthrange(today.year, today.month)
    if ym_all and ym_all[-1] == (today.year, today.month) and today.day < last_day:
        ym_all = ym_all[:-1]
    n_blind = min(blind_test_months_count, max(0, len(ym_all) - 2))
    blind_test_months = ym_all[-n_blind:] if n_blind > 0 else []
    ym_for_val = ym_all[:-n_blind] if n_blind > 0 else ym_all
    n_val = min(val_months_count, max(0, len(ym_for_val) - 1))
    val_months = ym_for_val[-n_val:] if n_val > 0 else []

    # Загружаем лучшую модель для честного инференса
    from model import load_model as _load_model
    best_model, _ = _load_model(best_model_path, device=str(device))

    mape_plot_path = plot_val_mape_by_day(
        wide_df=wide_df,
        categories=categories,
        val_months=val_months,
        model=best_model,
        scaler=train_ds.scaler,
        artifacts_dir=artifacts_dir,
        device=device,
        wide_create_df=wide_create_df,
        blind_test_months=blind_test_months,
        wide_profit_df=wide_profit_df,
    )
    print(f"Val MAPE plot saved → {mape_plot_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train FT-Transformer for sales forecasting")
    p.add_argument("--data", type=str, default=PATHS["data"], help="Path to sales CSV")
    p.add_argument("--epochs", type=int, default=TRAIN_CONFIG["epochs"])
    p.add_argument("--batch_size", type=int, default=TRAIN_CONFIG["batch_size"])
    p.add_argument("--lr", type=float, default=TRAIN_CONFIG["lr"])
    p.add_argument("--artifacts_dir", type=str, default=PATHS["artifacts_dir"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        artifacts_dir=args.artifacts_dir,
    )
