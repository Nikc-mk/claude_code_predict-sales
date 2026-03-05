"""
dataset.py — формирование обучающих сэмплов для TabTransformer.

Класс TabularDataset:
  - Для каждой пары (год-месяц, категория) и для каждого дня t от 1 до T-1
    создаёт один сэмпл: вектор признаков → таргет (остаток продаж с t+1 до T).
  - Поддерживает fit_scaler() и transform(), чтобы числовые признаки
    были нормализованы перед подачей в модель.
"""

import pickle
from calendar import monthrange
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from build_features import (
    build_calendar_df,
    get_cumulative_to_day,
    get_month_sales,
    get_month_total,
    get_previous_month_total,
    get_rolling_sum,
)
from config import NUMERICAL_FEATURES, NUM_NUMERICAL_FEATURES


class TabularDataset(Dataset):
    """
    Генерирует обучающие сэмплы из wide-таблицы продаж.

    Параметры
    ---------
    wide_df : pd.DataFrame
        Wide-таблица (индекс=дата, колонки=категории), результат pivot_to_wide().
    categories : list[str]
        Упорядоченный список категорий (определяет category_id → int).
    scaler : StandardScaler | None
        Если None — сначала вызовите fit_scaler(), затем transform().
        Если передан — используется для нормализации без повторного фита.
    fit_on_init : bool
        Если True, сразу подгоняет скейлер и нормализует.
    """

    def __init__(
        self,
        wide_df: pd.DataFrame,
        categories: list[str],
        wide_create_df: Optional[pd.DataFrame] = None,
        wide_profit_df: Optional[pd.DataFrame] = None,
        scaler: Optional[StandardScaler] = None,
        fit_on_init: bool = True,
    ):
        self.wide_df = wide_df
        self.wide_create_df = (
            wide_create_df if wide_create_df is not None
            else pd.DataFrame(0.0, index=wide_df.index, columns=wide_df.columns)
        )
        self.wide_profit_df = (
            wide_profit_df if wide_profit_df is not None
            else pd.DataFrame(0.0, index=wide_df.index, columns=wide_df.columns)
        )
        self.categories = categories
        self.cat2idx = {c: i for i, c in enumerate(categories)}
        self.n_cats = len(categories)

        self._raw_samples: list[dict] = []
        self._build_samples()

        self.scaler = scaler
        if fit_on_init and scaler is None:
            self.fit_scaler()
            self._apply_scaler()
        elif scaler is not None:
            self._apply_scaler()

        # После нормализации — массивы numpy для быстрого __getitem__
        self._finalize()

    # ------------------------------------------------------------------
    # Построение сырых сэмплов
    # ------------------------------------------------------------------

    def _build_samples(self) -> None:
        """Итерируется по всем (месяц, категория, день_t) и собирает сэмплы."""
        dates = self.wide_df.index
        if len(dates) == 0:
            return

        # Список уникальных (year, month) в хронологическом порядке
        ym_pairs = sorted(
            {(d.year, d.month) for d in dates},
            key=lambda x: (x[0], x[1]),
        )

        # --- Предвычисляем rolling-окна один раз для всего DataFrame ---
        # Вместо ~580 000 boolean-mask операций — 10 векторизованных вызовов
        roll7_s  = self.wide_df.rolling(7,  min_periods=1).sum()
        roll14_s = self.wide_df.rolling(14, min_periods=1).sum()
        roll28_s = self.wide_df.rolling(28, min_periods=1).sum()
        roll3_c  = self.wide_create_df.rolling(3,  min_periods=1).sum()
        roll7_c  = self.wide_create_df.rolling(7,  min_periods=1).sum()
        roll10_c = self.wide_create_df.rolling(10, min_periods=1).sum()
        roll14_c = self.wide_create_df.rolling(14, min_periods=1).sum()
        roll7_p  = self.wide_profit_df.rolling(7,  min_periods=1).sum()
        roll14_p = self.wide_profit_df.rolling(14, min_periods=1).sum()
        roll28_p = self.wide_profit_df.rolling(28, min_periods=1).sum()

        for year, month in ym_pairs:
            _, n_days = monthrange(year, month)
            cal = build_calendar_df(year, month)
            month_idx = float(year * 12 + month)
            month_sin = float(cal["month_sin"].iloc[0])
            month_cos = float(cal["month_cos"].iloc[0])

            for cat in self.categories:
                cat_idx = self.cat2idx[cat]
                month_sales  = get_month_sales(self.wide_df, year, month, cat)
                month_profit = get_month_sales(self.wide_profit_df, year, month, cat)

                # --- Константы для (year, month, cat) — не зависят от t ---
                lastyear_sales_s     = get_month_sales(self.wide_df, year - 1, month, cat)
                lastyear_month_total = float(lastyear_sales_s.sum())
                prev_month_total     = get_previous_month_total(self.wide_df, year, month, cat)

                lastyear_profit_s    = get_month_sales(self.wide_profit_df, year - 1, month, cat)
                lastyear_profit_total = float(lastyear_profit_s.sum())
                prev_profit_total    = get_previous_month_total(self.wide_profit_df, year, month, cat)

                # --- Cumulative sum: O(1) lookup вместо O(t) pandas slice ---
                sales_cs     = np.concatenate([[0.0], np.cumsum(month_sales.values)])
                profit_cs    = np.concatenate([[0.0], np.cumsum(month_profit.values)])
                ly_sales_cs  = np.concatenate([[0.0], np.cumsum(lastyear_sales_s.values)])
                ly_profit_cs = np.concatenate([[0.0], np.cumsum(lastyear_profit_s.values)])
                ly_s_len     = len(ly_sales_cs) - 1
                ly_p_len     = len(ly_profit_cs) - 1

                for t in range(1, n_days):  # t = 1..n_days-1 (последний день — только таргет)
                    day_date = pd.Timestamp(year=year, month=month, day=t)
                    cal_row  = cal.loc[day_date]

                    # Таргеты через cumsum: O(1)
                    target        = float(sales_cs[-1]  - sales_cs[t])
                    target_profit = float(profit_cs[-1] - profit_cs[t])

                    # Накопленные суммы текущего месяца: O(1)
                    cumulative        = float(sales_cs[t])
                    cumulative_profit = float(profit_cs[t])

                    # Rolling-окна: O(1) lookup в предвычисленных DataFrame
                    dd = day_date
                    s7   = float(roll7_s.at[dd, cat])  if dd in roll7_s.index  else 0.0
                    s14  = float(roll14_s.at[dd, cat]) if dd in roll14_s.index else 0.0
                    s28  = float(roll28_s.at[dd, cat]) if dd in roll28_s.index else 0.0
                    cs3  = float(roll3_c.at[dd, cat])  if dd in roll3_c.index  else 0.0
                    cs7  = float(roll7_c.at[dd, cat])  if dd in roll7_c.index  else 0.0
                    cs10 = float(roll10_c.at[dd, cat]) if dd in roll10_c.index else 0.0
                    cs14 = float(roll14_c.at[dd, cat]) if dd in roll14_c.index else 0.0
                    p7   = float(roll7_p.at[dd, cat])  if dd in roll7_p.index  else 0.0
                    p14  = float(roll14_p.at[dd, cat]) if dd in roll14_p.index else 0.0
                    p28  = float(roll28_p.at[dd, cat]) if dd in roll28_p.index else 0.0

                    # Накопленные суммы прошлого года: O(1)
                    lastyear_cumul        = float(ly_sales_cs[min(t, ly_s_len)])
                    lastyear_profit_cumul = float(ly_profit_cs[min(t, ly_p_len)])

                    self._raw_samples.append(
                        {
                            "features": [
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
                                month_idx,
                                float(day_date.toordinal()),
                                month_sin,
                                month_cos,
                                cumulative_profit,
                                p7, p14, p28,
                                lastyear_profit_cumul,
                                lastyear_profit_total,
                                prev_profit_total,
                            ],
                            "category_id": cat_idx,
                            "target": target,
                            "target_profit": target_profit,
                        }
                    )

    def _compute_features(
        self,
        year: int,
        month: int,
        cat: str,
        t: int,
        day_date: pd.Timestamp,
        cal_row: pd.Series,
        month_sales: pd.Series,
        month_profit: pd.Series,
    ) -> list[float]:
        """
        Возвращает числовые признаки в порядке NUMERICAL_FEATURES (28 штук).
        Используется только если вызывается напрямую (не из _build_samples).
        """
        cumulative = float(month_sales.iloc[:t].sum())
        sales_7  = get_rolling_sum(self.wide_df, day_date, cat, 7)
        sales_14 = get_rolling_sum(self.wide_df, day_date, cat, 14)
        sales_28 = get_rolling_sum(self.wide_df, day_date, cat, 28)
        cs_3  = get_rolling_sum(self.wide_create_df, day_date, cat, 3)
        cs_7  = get_rolling_sum(self.wide_create_df, day_date, cat, 7)
        cs_10 = get_rolling_sum(self.wide_create_df, day_date, cat, 10)
        cs_14 = get_rolling_sum(self.wide_create_df, day_date, cat, 14)
        lastyear_cumul       = get_cumulative_to_day(self.wide_df, year - 1, month, cat, t)
        lastyear_month_total = get_month_total(self.wide_df, year - 1, month, cat)
        prev_month_total     = get_previous_month_total(self.wide_df, year, month, cat)
        cumulative_profit        = float(month_profit.iloc[:t].sum())
        profit_7  = get_rolling_sum(self.wide_profit_df, day_date, cat, 7)
        profit_14 = get_rolling_sum(self.wide_profit_df, day_date, cat, 14)
        profit_28 = get_rolling_sum(self.wide_profit_df, day_date, cat, 28)
        lastyear_profit_cumul       = get_cumulative_to_day(self.wide_profit_df, year - 1, month, cat, t)
        lastyear_profit_month_total = get_month_total(self.wide_profit_df, year - 1, month, cat)
        prev_profit_month_total     = get_previous_month_total(self.wide_profit_df, year, month, cat)
        return [
            float(cal_row["days_left"]),
            float(cal_row["work_days_left"]),
            float(cal_row["days_passed"]),
            float(cal_row["work_days_passed"]),
            float(cal_row["is_weekend"]),
            cumulative,
            sales_7, sales_14, sales_28,
            cs_3, cs_7, cs_10, cs_14,
            lastyear_cumul, lastyear_month_total, prev_month_total,
            float(year), float(year * 12 + month), float(day_date.toordinal()),
            float(cal_row["month_sin"]), float(cal_row["month_cos"]),
            cumulative_profit,
            profit_7, profit_14, profit_28,
            lastyear_profit_cumul, lastyear_profit_month_total, prev_profit_month_total,
        ]

    # ------------------------------------------------------------------
    # Нормализация
    # ------------------------------------------------------------------

    def fit_scaler(self) -> "TabularDataset":
        """Фитирует StandardScaler на числовых признаках текущего датасета."""
        X = np.array([s["features"] for s in self._raw_samples], dtype=np.float32)
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        return self

    def _apply_scaler(self) -> None:
        """Применяет scaler к числовым признакам в self._raw_samples."""
        if not self._raw_samples:
            return
        X = np.array([s["features"] for s in self._raw_samples], dtype=np.float32)
        X_scaled = self.scaler.transform(X)
        for i, s in enumerate(self._raw_samples):
            s["features"] = X_scaled[i].tolist()

    def save_scaler(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)

    @staticmethod
    def load_scaler(path: str) -> StandardScaler:
        with open(path, "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # Финализация и PyTorch Dataset interface
    # ------------------------------------------------------------------

    def _finalize(self) -> None:
        n = len(self._raw_samples)
        self._X = np.zeros((n, NUM_NUMERICAL_FEATURES), dtype=np.float32)
        self._cat_ids = np.zeros(n, dtype=np.int64)
        self._y = np.zeros((n, 2), dtype=np.float32)  # [target_sales, target_profit]

        for i, s in enumerate(self._raw_samples):
            self._X[i] = s["features"]
            self._cat_ids[i] = s["category_id"]
            self._y[i, 0] = s["target"]
            self._y[i, 1] = s["target_profit"]

    def __len__(self) -> int:
        return len(self._raw_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Возвращает (features, category_id, targets):
          features     : FloatTensor (NUM_NUMERICAL_FEATURES,)
          category_id  : LongTensor  scalar
          targets      : FloatTensor (2,) — [target_sales, target_profit]
        """
        return (
            torch.from_numpy(self._X[idx]),
            torch.tensor(self._cat_ids[idx], dtype=torch.long),
            torch.from_numpy(self._y[idx]),
        )

    # ------------------------------------------------------------------
    # Утилиты
    # ------------------------------------------------------------------

    @staticmethod
    def train_val_split(
        wide_df: pd.DataFrame,
        categories: list[str],
        wide_create_df: Optional[pd.DataFrame] = None,
        wide_profit_df: Optional[pd.DataFrame] = None,
        val_months_count: int = 3,
        blind_test_months_count: int = 0,
    ) -> tuple["TabularDataset", "TabularDataset", Optional["TabularDataset"]]:
        """
        Разбивает данные по времени на train / val / blind_test.

        Порядок отсечения (от конца):
          1. blind_test_months_count последних завершённых месяцев → слепой тест
          2. val_months_count следующих (перед blind_test) → валидация
          3. остаток → обучение

        Скейлер фитируется ТОЛЬКО на тренировочных данных.
        Возвращает (train_ds, val_ds, blind_test_ds), где blind_test_ds=None если blind=0.
        """
        dates = wide_df.index
        ym_pairs = sorted(
            {(d.year, d.month) for d in dates},
            key=lambda x: (x[0], x[1]),
        )

        # Исключаем текущий месяц, если он незавершён
        today = date.today()
        current_ym = (today.year, today.month)
        _, last_day = monthrange(today.year, today.month)
        if ym_pairs and ym_pairs[-1] == current_ym and today.day < last_day:
            ym_pairs = ym_pairs[:-1]

        n_total = len(ym_pairs)

        # Слепой тест: последние n_blind месяцев
        n_blind = min(blind_test_months_count, max(0, n_total - 2))
        blind_ym = set(ym_pairs[-n_blind:]) if n_blind > 0 else set()
        ym_before_blind = ym_pairs[:-n_blind] if n_blind > 0 else ym_pairs

        # Валидация: n_val месяцев перед blind_test
        n_val = min(val_months_count, max(0, len(ym_before_blind) - 1))
        val_ym = set(ym_before_blind[-n_val:]) if n_val > 0 else set()
        train_ym = set(ym_before_blind[:-n_val]) if n_val > 0 else set(ym_before_blind)

        def _filter_wide(wdf, months_set):
            mask = [(d.year, d.month) in months_set for d in wdf.index]
            return wdf.loc[mask]

        train_wide = _filter_wide(wide_df, train_ym)
        val_wide = _filter_wide(wide_df, val_ym)

        train_create = _filter_wide(wide_create_df, train_ym) if wide_create_df is not None else None
        val_create = _filter_wide(wide_create_df, val_ym) if wide_create_df is not None else None

        train_profit = _filter_wide(wide_profit_df, train_ym) if wide_profit_df is not None else None
        val_profit = _filter_wide(wide_profit_df, val_ym) if wide_profit_df is not None else None

        # Обучаем скейлер только на train
        train_ds = TabularDataset(train_wide, categories, wide_create_df=train_create, wide_profit_df=train_profit, scaler=None, fit_on_init=True)
        val_ds = TabularDataset(val_wide, categories, wide_create_df=val_create, wide_profit_df=val_profit, scaler=train_ds.scaler, fit_on_init=False)

        blind_ds = None
        if n_blind > 0:
            blind_wide = _filter_wide(wide_df, blind_ym)
            blind_create = _filter_wide(wide_create_df, blind_ym) if wide_create_df is not None else None
            blind_profit = _filter_wide(wide_profit_df, blind_ym) if wide_profit_df is not None else None
            blind_ds = TabularDataset(blind_wide, categories, wide_create_df=blind_create, wide_profit_df=blind_profit, scaler=train_ds.scaler, fit_on_init=False)

        print(
            f"Dataset split: {len(train_ym)} train months ({len(train_ds)} samples), "
            f"{len(val_ym)} val months ({len(val_ds)} samples)"
            + (f", {n_blind} blind_test months ({len(blind_ds)} samples)" if blind_ds else "")
        )
        return train_ds, val_ds, blind_ds
