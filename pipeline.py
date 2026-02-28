"""
pipeline.py — ежедневный оркестратор пайплайна прогнозирования продаж.

Запуск (каждое утро, после получения данных за вчера):
    python pipeline.py --data data/sales.csv
    python pipeline.py --data data/sales.csv --retrain   # пересчитать модель
    python pipeline.py --data data/sales.csv --output forecasts/ --date 2026-02-22

Режимы:
  1. [По умолчанию] Только инференс (модель уже обучена):
     Загружает свежие данные → строит прогноз → сохраняет CSV.

  2. [--retrain] Переобучение + инференс:
     Обучает модель на свежих данных → строит прогноз.
     Рекомендуется запускать раз в месяц (или при поступлении нового месяца).

Выходной файл: forecasts/forecast_YYYYMM_YYYYMMDD.csv
"""

import argparse
import os
import sys
from datetime import datetime

from config import PATHS


def check_artifacts_exist() -> bool:
    """Проверяет, что все артефакты модели присутствуют."""
    required = [PATHS["model"], PATHS["scaler"], PATHS["category_encoder"]]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print("Артефакты модели не найдены:")
        for p in missing:
            print(f"  ✗ {p}")
        return False
    return True


def run_training(data_path: str, artifacts_dir: str) -> None:
    """Запускает полный цикл обучения."""
    print("\n" + "=" * 60)
    print("  ЭТАП 1: Обучение модели")
    print("=" * 60)
    from train import train
    train(data_path=data_path, artifacts_dir=artifacts_dir)


def run_inference(
    data_path: str,
    output_path: str,
    as_of_date: str | None = None,
    device: str = "cpu",
) -> None:
    """Запускает инференс и сохраняет прогноз."""
    print("\n" + "=" * 60)
    print("  ЭТАП 2: Прогнозирование")
    print("=" * 60)
    from predict import predict_current_month
    predict_current_month(
        data_path=data_path,
        model_path=PATHS["model"],
        scaler_path=PATHS["scaler"],
        cat_encoder_path=PATHS["category_encoder"],
        as_of_date=as_of_date,
        output_path=output_path,
        device=device,
    )


def run_daily_pipeline(
    data_path: str,
    output_path: str,
    retrain: bool = False,
    as_of_date: str | None = None,
    device: str = "cpu",
) -> None:
    """
    Основной метод ежедневного пайплайна.

    Параметры
    ---------
    data_path : str
        Путь к CSV с накопленными историческими данными (включая вчера).
    output_path : str
        Папка для сохранения прогнозов.
    retrain : bool
        Если True — переобучить модель перед прогнозом.
    as_of_date : str | None
        Дата расчёта (YYYY-MM-DD). None = сегодня.
    """
    start_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"  Sales Forecast Pipeline  |  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"  Data:    {data_path}")
    print(f"  Output:  {output_path}")
    print(f"  Retrain: {retrain}")

    artifacts_dir = PATHS["artifacts_dir"]
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    # --- Этап 1: Обучение (если нужно или модели нет) ---
    needs_train = retrain or not check_artifacts_exist()
    if needs_train:
        run_training(data_path=data_path, artifacts_dir=artifacts_dir)
    else:
        print("\nАртефакты модели найдены — пропускаем обучение (используйте --retrain для переобучения)")

    # --- Этап 2: Прогноз ---
    run_inference(
        data_path=data_path,
        output_path=output_path,
        as_of_date=as_of_date,
        device=device,
    )

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n  Пайплайн завершён за {elapsed:.1f} сек.")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Daily sales forecast pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Прогноз на сегодня (модель уже обучена):
  python pipeline.py --data data/sales.csv

  # Переобучить модель и построить прогноз:
  python pipeline.py --data data/sales.csv --retrain

  # Прогноз на конкретную дату:
  python pipeline.py --data data/sales.csv --date 2026-02-22

  # Сохранить прогноз в нестандартную папку:
  python pipeline.py --data data/sales.csv --output results/
        """,
    )
    p.add_argument("--data", type=str, default=PATHS["data"],
                   help="Путь к CSV с данными о продажах")
    p.add_argument("--output", type=str, default=PATHS["forecasts_dir"],
                   help="Папка для сохранения прогнозов")
    p.add_argument("--retrain", action="store_true",
                   help="Переобучить модель перед прогнозом")
    p.add_argument("--date", type=str, default=None,
                   help="Дата расчёта (YYYY-MM-DD). По умолчанию — сегодня")
    p.add_argument("--device", type=str, default="cpu",
                   help="Устройство PyTorch: cpu или cuda")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.data):
        print(f"Ошибка: файл данных не найден: {args.data}")
        print("Укажите путь через --data data/your_file.csv")
        sys.exit(1)

    run_daily_pipeline(
        data_path=args.data,
        output_path=args.output,
        retrain=args.retrain,
        as_of_date=args.date,
        device=args.device,
    )
