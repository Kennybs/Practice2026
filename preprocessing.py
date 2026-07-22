# Реализованы два этапа проверки:
# 1. Отбрасываются образцы, у которых максимум напряжения находится
#    в последней точке измерения.
# 2. После достижения максимального напряжения удаляется хвост кривой,
#    если напряжение опускается ниже 99% от максимального.
#
# Такой подход позволяет сохранить максимальное количество корректных
# образцов и убрать участок кривой после разрушения материала.
import pandas as pd
import numpy as np

# Загружаем подготовленный датасет.
df = pd.read_csv("data/PreparedData.csv", sep=";", low_memory=False)

# Удаляем пустой столбец, если он есть.
df = df.drop(columns=["Unnamed: 9"], errors="ignore")

# Убираем лишние пробелы.
df.columns = df.columns.str.strip()

# Порог — 1% от максимального напряжения.
THRESHOLD = 0.01

bad_samples = []

# Список частей DataFrame, которые затем объединим.
processed_samples = []

for sample_id, sample in df.groupby("sample_id"):

    sample = sample.reset_index(drop=True)

    # -------------------------
    # Первое условие
    # -------------------------

    peak_index = sample["Напряжение"].idxmax()

    # Если максимум оказался в последней точке,
    # образец считается некорректным.
    if peak_index == len(sample) - 1:
        bad_samples.append(sample_id)
        continue

    # -------------------------
    # Второе условие
    # -------------------------

    peak_stress = sample.loc[peak_index, "Напряжение"]

    # После пика рассматриваем только участок разрушения.
    post_peak = sample.iloc[peak_index + 1:].copy()

    # Ищем первую точку,
    # где напряжение отклоняется от максимума больше чем на 1%.
    cut_index = None

    for idx, row in post_peak.iterrows():

        relative_drop = (peak_stress - row["Напряжение"]) / peak_stress

        if relative_drop > THRESHOLD:
            cut_index = idx
            break

    # Если нашли такую точку —
    # оставляем только часть кривой ДО нее.
    if cut_index is not None:

        sample = sample.iloc[:cut_index]

    processed_samples.append(sample)

# Собираем итоговый DataFrame.
clean_df = pd.concat(processed_samples, ignore_index=True)

print("\nПлохие образцы:")
print(bad_samples)

print(f"\nВсего плохих образцов: {len(bad_samples)}")

print(f"\nДо очистки: {len(df)} строк")
print(f"После очистки: {len(clean_df)} строк")
print(f"Удалено строк: {len(df) - len(clean_df)}")

print(clean_df["sample_id"].isin(bad_samples).any())

# Сохраняем результат.
#clean_df.to_csv("data/CleanData.csv", sep=";", index=False)