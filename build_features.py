"""
build_features.py — ETL и генерация признаков для пайплайна прогнозирования продаж.

Основные функции:
  load_raw_data()      — загрузка и очистка сырого CSV
  pivot_to_wide()      — преобразование long → wide формат
  get_work_days_in_range() — подсчёт рабочих дней в диапазоне
  build_calendar_df()  — таблица календарных признаков для месяца
"""

import re
import numpy as np
import pandas as pd
from calendar import monthrange


# ---------------------------------------------------------------------------
# 1. Загрузка сырых данных
# ---------------------------------------------------------------------------

def load_raw_data(path: str) -> pd.DataFrame:
    """
    Читает CSV с продажами. Ожидаемые колонки (имена гибкие):
      - Дата         → date  (формат dd.mm.yyyy или yyyy-mm-dd)
      - Категория    → category
      - Оборот_факт  → sales (может содержать пробелы как разделители тысяч)

    Возвращает DataFrame с колонками: date (datetime64), category (str), sales (float).
    """
    df = pd.read_csv(path, sep=None, engine="python")

    # Переименуем колонки по позиции, если имена отличаются
    df.columns = [c.strip() for c in df.columns]

    # Ищем колонку create_sale ДО rename (пока все колонки видны)
    create_raw = None
    if "create_sale" in df.columns:
        create_raw = "create_sale"
    else:
        create_raw = next((c for c in df.columns if "create" in c.lower() and "margin" not in c.lower()), None)

    # Ищем колонку profit ДО rename
    profit_raw = "profit" if "profit" in df.columns else None

    col_map = _detect_columns(df.columns.tolist())
    df = df.rename(columns=col_map)

    # Сохраняем create_sale из сырого df до среза
    if create_raw is not None:
        # после rename имя могло остаться или смениться
        create_renamed = col_map.get(create_raw, create_raw)
        create_series = df[create_renamed].copy()
    else:
        create_series = None

    # Сохраняем profit до среза
    if profit_raw is not None:
        profit_renamed = col_map.get(profit_raw, profit_raw)
        profit_series = df[profit_renamed].copy()
    else:
        profit_series = None

    df = df[["date", "category", "sales"]]

    # Очистка числового поля от пробелов/неразрывных пробелов
    def _clean_numeric(s: pd.Series) -> pd.Series:
        return (
            s.astype(str)
            .str.replace(r"[\s\u00a0\u202f]", "", regex=True)
            .str.replace(",", ".")
            .astype(float)
        )

    df["sales"] = _clean_numeric(df["sales"])

    if create_series is not None:
        df["create_sale"] = _clean_numeric(create_series).values
    else:
        df["create_sale"] = 0.0

    if profit_series is not None:
        df["profit"] = _clean_numeric(profit_series).values
    else:
        df["profit"] = 0.0

    # Парсинг даты
    df["date"] = _parse_dates(df["date"])

    # Категория — строка без лишних пробелов
    df["category"] = df["category"].astype(str).str.strip()

    df = df.dropna(subset=["date", "category"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _detect_columns(cols: list[str]) -> dict:
    """Эвристика: определяет, какая колонка дата/категория/продажи."""
    mapping = {}
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in ("дат", "date")):
            mapping[c] = "date"
        elif any(k in cl for k in ("катег", "category", "товар")):
            mapping[c] = "category"
        elif any(k in cl for k in ("оборот", "продаж", "sales", "сумм", "выруч")):
            mapping[c] = "sales"
    # Если не нашли по имени — берём по позиции (0=date, 1=category, 2=sales)
    if len(mapping) < 3:
        for i, c in enumerate(cols[:3]):
            if i == 0 and "date" not in mapping.values():
                mapping[c] = "date"
            elif i == 1 and "category" not in mapping.values():
                mapping[c] = "category"
            elif i == 2 and "sales" not in mapping.values():
                mapping[c] = "sales"
    return mapping


def _parse_dates(series: pd.Series) -> pd.Series:
    """Пробует форматы dd.mm.yyyy, dd/mm/yyyy, yyyy-mm-dd."""
    for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d", "%Y.%m.%d"):
        try:
            parsed = pd.to_datetime(series, format=fmt)
            if parsed.notna().mean() > 0.9:
                return parsed
        except Exception:
            pass
    # Fallback — pandas сам угадывает
    return pd.to_datetime(series, dayfirst=True, errors="coerce")


# ---------------------------------------------------------------------------
# 2. Pivot: long → wide
# ---------------------------------------------------------------------------

def pivot_to_wide(
    df: pd.DataFrame,
    categories: list[str] | None = None,
    value_col: str = "sales",
) -> pd.DataFrame:
    """
    Преобразует long-таблицу в wide-таблицу:
      index  = date (ежедневная частота)
      columns = категории товаров
      values = value_col (пропуски → 0)

    Параметры
    ---------
    categories : список категорий для включения в результат.
                 Если None — берутся все категории из данных.
                 Если передан — недостающие колонки добавляются нулями.
    value_col : колонка со значениями (по умолчанию "sales").
    """
    # Агрегация: если одна категория встречается несколько раз в день — суммируем
    agg = df.groupby(["date", "category"])[value_col].sum().unstack(fill_value=0)

    # Ежедневная частота — заполняем пропущенные дни нулями
    date_range = pd.date_range(agg.index.min(), agg.index.max(), freq="D")
    agg = agg.reindex(date_range, fill_value=0)
    agg.index.name = "date"

    # Привести к нужному набору категорий
    if categories:
        for cat in categories:
            if cat not in agg.columns:
                agg[cat] = 0.0
        agg = agg[categories]

    return agg.sort_index()


# ---------------------------------------------------------------------------
# 3. Рабочие дни
# ---------------------------------------------------------------------------

def get_work_days_in_range(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """
    Возвращает количество рабочих дней (Пн–Пт) от start до end включительно.
    Использует np.busday_count (не учитывает праздники).
    """
    if start > end:
        return 0
    return int(np.busday_count(start.date(), (end + pd.Timedelta(days=1)).date()))


# ---------------------------------------------------------------------------
# 4. Календарные признаки
# ---------------------------------------------------------------------------

def build_calendar_df(year: int, month: int) -> pd.DataFrame:
    """
    Строит DataFrame с календарными признаками для каждого дня месяца.

    Колонки:
      date, days_passed, days_left,
      work_days_passed, work_days_left,
      is_weekend, month_sin, month_cos, day_of_week
    """
    _, n_days = monthrange(year, month)
    dates = pd.date_range(f"{year}-{month:02d}-01", periods=n_days, freq="D")
    month_start = dates[0]
    month_end = dates[-1]

    records = []
    for d in dates:
        t = d.day  # порядковый номер дня (1..n_days)
        records.append(
            {
                "date": d,
                "days_passed": t,
                "days_left": n_days - t,
                "work_days_passed": get_work_days_in_range(month_start, d),
                "work_days_left": get_work_days_in_range(d + pd.Timedelta(days=1), month_end),
                "is_weekend": 1 if d.dayofweek >= 5 else 0,
                "month_sin": np.sin(2 * np.pi * month / 12),
                "month_cos": np.cos(2 * np.pi * month / 12),
                "day_of_week": d.dayofweek,
            }
        )

    return pd.DataFrame(records).set_index("date")


# ---------------------------------------------------------------------------
# 5. Вспомогательные утилиты
# ---------------------------------------------------------------------------

def get_month_sales(wide_df: pd.DataFrame, year: int, month: int, category: str) -> pd.Series:
    """
    Возвращает Series с ежедневными продажами по категории за указанный год-месяц.
    Если данных нет — возвращает Series из нулей нужной длины.
    """
    _, n_days = monthrange(year, month)
    dates = pd.date_range(f"{year}-{month:02d}-01", periods=n_days, freq="D")

    if category not in wide_df.columns:
        return pd.Series(0.0, index=dates)

    mask = (wide_df.index.year == year) & (wide_df.index.month == month)
    month_data = wide_df.loc[mask, category]
    return month_data.reindex(dates, fill_value=0.0)


def get_cumulative_to_day(wide_df: pd.DataFrame, year: int, month: int,
                          category: str, day_t: int) -> float:
    """Накопленная сумма продаж с 1-го по day_t включительно."""
    s = get_month_sales(wide_df, year, month, category)
    return float(s.iloc[:day_t].sum())


def get_rolling_sum(wide_df: pd.DataFrame, end_date: pd.Timestamp,
                    category: str, window: int) -> float:
    """
    Сумма продаж за последние `window` дней до end_date включительно.
    Если данных меньше — берём то, что есть.
    """
    if category not in wide_df.columns:
        return 0.0
    start = end_date - pd.Timedelta(days=window - 1)
    mask = (wide_df.index >= start) & (wide_df.index <= end_date)
    return float(wide_df.loc[mask, category].sum())


def get_previous_month_total(wide_df: pd.DataFrame, year: int, month: int,
                             category: str) -> float:
    """Итог продаж за предыдущий месяц."""
    prev_month = month - 1
    prev_year = year
    if prev_month == 0:
        prev_month = 12
        prev_year -= 1
    s = get_month_sales(wide_df, prev_year, prev_month, category)
    return float(s.sum())


def get_month_total(wide_df: pd.DataFrame, year: int, month: int,
                    category: str) -> float:
    """Итог продаж за весь указанный месяц (используется для сезонности прошлого года)."""
    s = get_month_sales(wide_df, year, month, category)
    return float(s.sum())
