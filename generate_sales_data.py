"""Генератор тестовых данных о продажах в CSV формате."""

import csv
from datetime import date, timedelta
from pathlib import Path

CATEGORIES = [
    "MA002", "MA004", "MA005", "MA007", "MA009", "MA010", "MA011", "MA014", "MA015",
    "MA017", "MA019", "MA020", "MA021", "MA022", "MA025", "MA026", "MA029", "MA030",
    "MA031", "MA032", "MA033", "MA037", "MA039", "MA040", "MA041", "MA042", "MA043",
    "MA044", "MA045", "MA046", "MA047", "MA048", "MA049", "MA050", "MA051", "MA052",
    "MA053", "MA054", "MA055", "MA056", "MA057", "MA058", "MA059", "MA061", "MA062",
    "MA063", "MA064", "MA065", "MA066", "MA067", "MA068", "MA069", "MA070", "MA071",
    "MA072", "MA073", "MA076", "MA077", "MA078"
]

OUTPUT_FILE = "sales_data.csv"


def generate_sales_data():
    """Генерирует тестовые данные о продажах."""
    today = date.today()
    
    # Начало месяца от (текущая дата - 2 года)
    start_date = date(today.year - 4, today.month, 1)
    
    # Вчера (включительно)
    end_date = today - timedelta(days=1)
    
    print(f"Генерация данных с {start_date} по {end_date}")
    print(f"Количество категорий: {len(CATEGORIES)}")
    
    rows = []
    current_date = start_date
    
    while current_date <= end_date:
        for category in CATEGORIES:
            # Генерируем случайные продажи от 100 до 50000
            sales = 100 + (hash(f"{current_date}{category}") % 49900)
            # Генерация create_sale (больше sales, от sales + 1 до sales + 50000)
            create_sale = sales + 1 + (hash(f"{current_date}{category}_create") % 49900)
            # Profit: от -5% до 55% от sales (натуральное значение)
            profit_pct = -0.05 + (abs(hash(f"{current_date}{category}_profit")) % 10001) / 10000 * 0.60
            profit = round(sales * profit_pct)
            # create_sale_margin: от -0.10 до 0.70
            create_sale_margin = round(
                -0.10 + (abs(hash(f"{current_date}{category}_margin")) % 10001) / 10000 * 0.80,
                4
            )
            rows.append({
                "date": current_date.isoformat(),
                "category": category,
                "sales": sales,
                "create_sale": create_sale,
                "profit": profit,
                "create_sale_margin": create_sale_margin,
            })
        current_date += timedelta(days=1)
    
    print(f"Всего строк: {len(rows)}")
    
    # Запись в CSV
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "category", "sales", "create_sale", "profit", "create_sale_margin"], delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Данные записаны в файл: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_sales_data()