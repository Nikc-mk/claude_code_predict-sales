"""
predict.py — инференс: прогноз итоговых продаж по всем категориям на текущий месяц.

Запуск:
    python predict.py --data data/sales.csv
    python predict.py --data data/sales.csv --date 2026-02-22 --output forecasts/

Логика:
  1. Загружаем все исторические данные вплоть до yesterday (или --date)
  2. Для каждой категории вычисляем признаки на момент дня t
  3. Запускаем модель → получаем predicted_remaining
  4. Итог = cumulative_sales_1_to_t + predicted_remaining
  5. Выводим таблицу + сохраняем CSV
"""

import argparse
import os
import pickle
from datetime import date, datetime, timedelta
from calendar import monthrange

import numpy as np
import pandas as pd
import torch

from build_features import (
    build_calendar_df,
    get_cumulative_to_day,
    get_previous_month_total,
    get_rolling_sum,
    load_raw_data,
    pivot_to_wide,
)
from config import CATEGORIES, PATHS
from model import load_model


# ---------------------------------------------------------------------------
# Основная функция прогнозирования
# ---------------------------------------------------------------------------

def predict_current_month(
    data_path: str,
    model_path: str = PATHS["model"],
    scaler_path: str = PATHS["scaler"],
    cat_encoder_path: str = PATHS["category_encoder"],
    as_of_date: date | None = None,
    output_path: str | None = None,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Прогнозирует итоговые продажи за текущий месяц для всех категорий.

    Параметры
    ---------
    data_path : str
        Путь к CSV с историческими данными (включая данные за вчерашний день).
    as_of_date : date | None
        Дата "сегодня". Если None — берётся datetime.today().
        Прогноз строится на основе данных по as_of_date - 1 (вчера).
    output_path : str | None
        Если задан — сохраняет CSV с прогнозом в эту папку.

    Returns
    -------
    pd.DataFrame с колонками:
        category, fact_so_far, predicted_remaining, total_forecast,
        days_passed, days_left
    """
    # --- 0. Дата "сегодня" и "вчера" ---
    if as_of_date is None:
        as_of_date = datetime.today().date()
    if isinstance(as_of_date, str):
        as_of_date = datetime.strptime(as_of_date, "%Y-%m-%d").date()

    yesterday = as_of_date - timedelta(days=1)
    year, month = as_of_date.year, as_of_date.month  # прогнозируем текущий календарный месяц
    t = as_of_date.day - 1  # данных за текущий месяц: 0 дней если 1-е число

    _, n_days = monthrange(year, month)
    days_left = n_days - t

    print(f"Forecast as of: {as_of_date}  (data through: {yesterday})")
    print(f"Month: {year}-{month:02d}, day t={t}/{n_days}, days_left={days_left}")

    # --- 1. Загрузка данных ---
    df = load_raw_data(data_path)
    # Отрезаем данные строго до вчера включительно (нет данных из будущего)
    df = df[df["date"].dt.date <= yesterday]

    categories = sorted(CATEGORIES) if CATEGORIES else None

    with open(cat_encoder_path, "rb") as f:
        categories = pickle.load(f)

    wide_df = pivot_to_wide(df, categories=categories)
    wide_create_df = pivot_to_wide(df, categories=categories, value_col="create_sale")

    # --- 2. Загрузка модели и скейлера ---
    model, checkpoint = load_model(model_path, device=device)
    model.eval()

    from sklearn.preprocessing import StandardScaler
    with open(scaler_path, "rb") as f:
        scaler: StandardScaler = pickle.load(f)

    cat2idx = {c: i for i, c in enumerate(categories)}

    # --- 3. Календарные признаки для дня t ---
    # При t=0 (1-е число) используем день 1 для признаков: тогда lastyear_cumul
    # вернёт продажи 1-го дня прошлого года (≠0, разные у категорий), а не 0.
    t_eff = max(t, 1)
    cal = build_calendar_df(year, month)
    day_date = pd.Timestamp(year=year, month=month, day=t_eff)
    cal_row = cal.loc[day_date] if day_date in cal.index else cal.iloc[-1]

    # --- 4. Формируем признаки для всех категорий ---
    records = []
    for cat in categories:
        feat = _compute_inference_features(
            wide_df=wide_df,
            year=year,
            month=month,
            cat=cat,
            t=t_eff,
            day_date=day_date,
            cal_row=cal_row,
            wide_create_df=wide_create_df,
        )
        records.append(feat)

    X_raw = np.array([r["features"] for r in records], dtype=np.float32)
    cat_ids = np.array([cat2idx[r["category"]] for r in records], dtype=np.int64)
    cumulative_sales = np.array([r["cumulative"] for r in records], dtype=np.float64)

    # --- 5. Нормализация числовых признаков ---
    X_scaled = scaler.transform(X_raw)

    # --- 6. Инференс ---
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    cat_tensor = torch.tensor(cat_ids, dtype=torch.long).to(device)

    with torch.no_grad():
        predicted_remaining = model(X_tensor, cat_tensor).cpu().numpy()

    # --- 7. Формируем итоговую таблицу ---
    results = []
    for i, cat in enumerate(categories):
        total_forecast = cumulative_sales[i] + predicted_remaining[i]
        results.append(
            {
                "category": cat,
                "fact_so_far": round(cumulative_sales[i], 0),
                "predicted_remaining": round(float(predicted_remaining[i]), 0),
                "total_forecast": round(float(total_forecast), 0),
                "days_passed": t,
                "days_left": days_left,
            }
        )

    result_df = pd.DataFrame(results)
    _print_results(result_df, year, month, as_of_date)

    # --- 8. Сохранение ---
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        filename = f"forecast_{year}{month:02d}_{as_of_date.strftime('%Y%m%d')}.csv"
        out_file = os.path.join(output_path, filename)
        result_df.to_csv(out_file, index=False, encoding="utf-8-sig")
        print(f"\nForecast saved → {out_file}")

    return result_df


# ---------------------------------------------------------------------------
# Вычисление признаков для инференса
# ---------------------------------------------------------------------------

def _compute_inference_features(
    wide_df: pd.DataFrame,
    year: int,
    month: int,
    cat: str,
    t: int,
    day_date: pd.Timestamp,
    cal_row: pd.Series,
    wide_create_df: pd.DataFrame | None = None,
) -> dict:
    """Аналог TabularDataset._compute_features, но для одной категории."""
    from build_features import get_month_sales, get_month_total

    month_sales = get_month_sales(wide_df, year, month, cat)
    cumulative = float(month_sales.iloc[:t].sum())

    sales_7 = get_rolling_sum(wide_df, day_date, cat, 7)
    sales_14 = get_rolling_sum(wide_df, day_date, cat, 14)
    sales_28 = get_rolling_sum(wide_df, day_date, cat, 28)

    wcd = wide_create_df if wide_create_df is not None else wide_df
    cs_3 = get_rolling_sum(wcd, day_date, cat, 3)
    cs_7 = get_rolling_sum(wcd, day_date, cat, 7)
    cs_10 = get_rolling_sum(wcd, day_date, cat, 10)
    cs_14 = get_rolling_sum(wcd, day_date, cat, 14)

    lastyear_cumul = get_cumulative_to_day(wide_df, year - 1, month, cat, t)
    lastyear_month_total = get_month_total(wide_df, year - 1, month, cat)
    prev_month_total = get_previous_month_total(wide_df, year, month, cat)

    features = [
        float(cal_row["days_left"]),
        float(cal_row["work_days_left"]),
        float(cal_row["days_passed"]),
        float(cal_row["work_days_passed"]),
        float(cal_row["is_weekend"]),
        cumulative,
        sales_7,
        sales_14,
        sales_28,
        cs_3,
        cs_7,
        cs_10,
        cs_14,
        lastyear_cumul,
        lastyear_month_total,
        prev_month_total,
        float(year),
        float(year * 12 + month),
        float(day_date.toordinal()),
        float(cal_row["month_sin"]),
        float(cal_row["month_cos"]),
    ]
    return {"category": cat, "features": features, "cumulative": cumulative}


# ---------------------------------------------------------------------------
# Вывод результатов
# ---------------------------------------------------------------------------

def _print_results(df: pd.DataFrame, year: int, month: int, as_of: date) -> None:
    print(f"\n{'='*70}")
    print(f"  Прогноз продаж за {year}-{month:02d}  |  Дата расчёта: {as_of}")
    print(f"{'='*70}")
    print(
        f"{'Категория':<12}  {'Факт 1-t':>14}  {'Прогноз остатка':>16}  {'Итоговый прогноз':>17}"
    )
    print("-" * 70)
    for _, row in df.iterrows():
        print(
            f"{row['category']:<12}  "
            f"{row['fact_so_far']:>14,.0f}  "
            f"{row['predicted_remaining']:>16,.0f}  "
            f"{row['total_forecast']:>17,.0f}"
        )
    print("-" * 70)
    total = df["total_forecast"].sum()
    total_fact = df["fact_so_far"].sum()
    print(f"{'ИТОГО':<12}  {total_fact:>14,.0f}  {'':>16}  {total:>17,.0f}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Predict monthly sales totals")
    p.add_argument("--data", type=str, default=PATHS["data"], help="Path to sales CSV")
    p.add_argument("--model", type=str, default=PATHS["model"])
    p.add_argument("--scaler", type=str, default=PATHS["scaler"])
    p.add_argument("--cat_encoder", type=str, default=PATHS["category_encoder"])
    p.add_argument(
        "--date", type=str, default=None,
        help="As-of date (YYYY-MM-DD). Default: today. Data through date-1.",
    )
    p.add_argument("--output", type=str, default=None, help="Output folder for CSV")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict_current_month(
        data_path=args.data,
        model_path=args.model,
        scaler_path=args.scaler,
        cat_encoder_path=args.cat_encoder,
        as_of_date=args.date,
        output_path=args.output,
        device=args.device,
    )
