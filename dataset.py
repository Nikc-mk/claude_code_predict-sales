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
        scaler: Optional[StandardScaler] = None,
        fit_on_init: bool = True,
    ):
        self.wide_df = wide_df
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

        for year, month in ym_pairs:
            _, n_days = monthrange(year, month)
            cal = build_calendar_df(year, month)

            for cat in self.categories:
                cat_idx = self.cat2idx[cat]
                month_sales = get_month_sales(self.wide_df, year, month, cat)

                for t in range(1, n_days):  # t = 1..n_days-1 (последний день — только таргет)
                    day_date = pd.Timestamp(year=year, month=month, day=t)
                    cal_row = cal.loc[day_date]

                    # --- Таргет: сумма продаж с дня t+1 до конца месяца ---
                    target = float(month_sales.iloc[t:].sum())

                    # --- Числовые признаки ---
                    features = self._compute_features(
                        year=year,
                        month=month,
                        cat=cat,
                        t=t,
                        day_date=day_date,
                        cal_row=cal_row,
                        month_sales=month_sales,
                    )

                    self._raw_samples.append(
                        {
                            "features": features,       # list[float], len=NUM_NUMERICAL_FEATURES
                            "category_id": cat_idx,     # int
                            "target": target,           # float
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
    ) -> list[float]:
        """
        Возвращает числовые признаки в порядке NUMERICAL_FEATURES (17 штук):
          days_left, work_days_left, days_passed, work_days_passed, is_weekend,
          cumulative_sales, sales_last_7_days, sales_last_14_days, sales_last_28_days,
          sales_lastyear_1_to_t, sales_lastyear_month_total,
          sales_previous_month_total,
          year, month_idx, time_idx,
          month_sin, month_cos
        """
        # Накопленная сумма с 1 по t (включительно)
        cumulative = float(month_sales.iloc[:t].sum())

        # Скользящие окна (от day_date в сторону прошлого)
        sales_7 = get_rolling_sum(self.wide_df, day_date, cat, 7)
        sales_14 = get_rolling_sum(self.wide_df, day_date, cat, 14)
        sales_28 = get_rolling_sum(self.wide_df, day_date, cat, 28)

        # Прошлый год: накопленная сумма за 1..t и итог всего месяца
        lastyear_cumul = get_cumulative_to_day(self.wide_df, year - 1, month, cat, t)
        lastyear_month_total = get_month_total(self.wide_df, year - 1, month, cat)

        # Итог предыдущего месяца
        prev_month_total = get_previous_month_total(self.wide_df, year, month, cat)

        return [
            float(cal_row["days_left"]),
            float(cal_row["work_days_left"]),
            float(cal_row["days_passed"]),
            float(cal_row["work_days_passed"]),
            float(cal_row["is_weekend"]),
            cumulative,
            sales_7,
            sales_14,
            sales_28,
            lastyear_cumul,
            lastyear_month_total,
            prev_month_total,
            float(year),
            float(year * 12 + month),
            float(day_date.toordinal()),
            float(cal_row["month_sin"]),
            float(cal_row["month_cos"]),
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
        self._y = np.zeros(n, dtype=np.float32)

        for i, s in enumerate(self._raw_samples):
            self._X[i] = s["features"]
            self._cat_ids[i] = s["category_id"]
            self._y[i] = s["target"]

    def __len__(self) -> int:
        return len(self._raw_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Возвращает (features, category_id, target):
          features     : FloatTensor (NUM_NUMERICAL_FEATURES,)
          category_id  : LongTensor  scalar
          target       : FloatTensor scalar
        """
        return (
            torch.from_numpy(self._X[idx]),
            torch.tensor(self._cat_ids[idx], dtype=torch.long),
            torch.tensor(self._y[idx], dtype=torch.float32),
        )

    # ------------------------------------------------------------------
    # Утилиты
    # ------------------------------------------------------------------

    @staticmethod
    def train_val_split(
        wide_df: pd.DataFrame,
        categories: list[str],
        val_months_count: int = 3,
    ) -> tuple["TabularDataset", "TabularDataset"]:
        """
        Разбивает данные по времени: последние val_months_count месяцев — валидация.
        Скейлер фитируется ТОЛЬКО на тренировочных данных, затем применяется к обоим.
        """
        dates = wide_df.index
        ym_pairs = sorted(
            {(d.year, d.month) for d in dates},
            key=lambda x: (x[0], x[1]),
        )
        n_total = len(ym_pairs)
        n_val = min(val_months_count, n_total - 1)  # не менее 1 месяца на train
        n_train = n_total - n_val

        train_months = set(ym_pairs[:n_train])
        val_months = set(ym_pairs[n_train:])

        def _filter_wide(months_set):
            mask = [(d.year, d.month) in months_set for d in wide_df.index]
            return wide_df.loc[mask]

        train_wide = _filter_wide(train_months)
        val_wide = _filter_wide(val_months)

        # Обучаем скейлер только на train
        train_ds = TabularDataset(train_wide, categories, scaler=None, fit_on_init=True)
        val_ds = TabularDataset(val_wide, categories, scaler=train_ds.scaler, fit_on_init=False)

        print(
            f"Dataset split: {len(train_months)} train months ({len(train_ds)} samples), "
            f"{len(val_months)} val months ({len(val_ds)} samples)"
        )
        return train_ds, val_ds
