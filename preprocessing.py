"""
ПОЛНЫЙ КОНВЕЙЕР ПОДГОТОВКИ ДАННЫХ
Объединяет DataConnect и preprocessing в один скрипт
"""

import pandas as pd
import numpy as np

print("ЗАПУСК ПОДГОТОВКИ ДАННЫХ")


# ==========================================================
# ЭТАП 1: ОБЪЕДИНЕНИЕ ДАННЫХ 
# ==========================================================

print("\n[1/3] Загрузка и объединение данных из Excel...")

# Загружаем метаданные
metadata_df = pd.read_excel("data/TargetData.xlsx", sheet_name="Результаты")
metadata_df.rename(columns={metadata_df.columns[0]: 'sample_id'}, inplace=True)

# Загружаем все листы с кривыми
all_sheets = pd.read_excel("data/TargetData.xlsx", sheet_name=None)
curve_sheets = {name: df for name, df in all_sheets.items() if name != 'Результаты'}

# Объединяем кривые с метаданными
processed_dfs = []
for name, df in curve_sheets.items():
    curve = pd.read_excel(
        "data/TargetData.xlsx", 
        sheet_name=name, 
        skiprows=3, 
        header=None, 
        names=["Деформация", "Напряжение"]
    )
    curve["sample_id"] = name
    merged = pd.merge(curve, metadata_df, on="sample_id", how="left")
    processed_dfs.append(merged)

full_df = pd.concat(processed_dfs, ignore_index=True)

print(f"Объединено: {len(full_df)} строк, {full_df['sample_id'].nunique()} образцов")

# ==========================================================
# ЭТАП 2: ОЧИСТКА ДАННЫХ 
# ==========================================================

print("\n[2/3] Очистка данных от аномалий...")

# Удаляем мусорные колонки
full_df = full_df.drop(columns=[col for col in full_df.columns if 'Unnamed' in str(col)], errors="ignore")
full_df.columns = full_df.columns.str.strip()

THRESHOLD = 0.01
bad_samples = []
processed_samples = []

for sample_id, sample in full_df.groupby("sample_id"):
    sample = sample.reset_index(drop=True)
    peak_index = sample["Напряжение"].idxmax()
    
    # Проверка 1: пик в последней точке
    if peak_index == len(sample) - 1:
        bad_samples.append(sample_id)
        continue
    
    # Проверка 2: обрезка хвоста
    peak_stress = sample.loc[peak_index, "Напряжение"]
    post_peak = sample.iloc[peak_index + 1:].copy()
    
    cut_index = None
    for idx, row in post_peak.iterrows():
        relative_drop = (peak_stress - row["Напряжение"]) / peak_stress
        if relative_drop > THRESHOLD:
            cut_index = idx
            break
    
    if cut_index is not None:
        sample = sample.iloc[:cut_index]
    
    processed_samples.append(sample)

clean_df = pd.concat(processed_samples, ignore_index=True)

print(f"❌ Удалено образцов: {len(bad_samples)}")
print(f" До очистки: {len(full_df)} строк")
print(f"📈 После очистки: {len(clean_df)} строк")
print(f"️ Удалено строк: {len(full_df) - len(clean_df)}")

# ==========================================================
# ЭТАП 3: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ==========================================================

print("\n[3/3] Сохранение результатов...")

# Опционально: сохраняем промежуточный файл (можно закомментировать если промежуточный не нужен)
full_df.to_csv("data/PreparedData.csv", sep=";", index=False)
print("💾 Сохранен: data/PreparedData.csv (промежуточный файл)")

# Сохраняем чистый датасет
clean_df.to_csv("data/CleanData.csv", sep=";", index=False)
print("💾 Сохранен: data/CleanData.csv (готовый для обучения)")

# Вывод статистики
print("Предобработка завершена успешно")
print(f"📦 Итого образцов: {clean_df['sample_id'].nunique()}")
print(f"📏 Итого строк: {len(clean_df)}")
print(f"📂 Файл: data/CleanData.csv")
