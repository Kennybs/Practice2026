import pandas as pd
import numpy as np

# Загружаем подготовленный датасет.
# low_memory=False убирает предупреждения Pandas о смешанных типах данных.
df = pd.read_csv("data/PreparedData.csv", sep=";", low_memory=False)

# Удаляем пустой столбец, если он присутствует.
df = df.drop(columns=["Unnamed: 9"], errors="ignore")

# Убираем лишние пробелы из названий столбцов.
df.columns = df.columns.str.strip()

# Отладочная информация (при необходимости можно раскомментировать).
# print(df.columns)
# print(df.head())
# print(df.info())

# Список sample_id образцов, признанных некорректными.
bad_samples = []

# Обрабатываем каждый образец отдельно.
for sample_id, sample in df.groupby("sample_id"):

    # Переиндексируем строки внутри образца.
    sample = sample.reset_index(drop=True)

    # Индекс (номер точки) максимального напряжения.
    peak_index = sample["Напряжение"].idxmax()

    # Деформация в момент максимального напряжения.
    peak_strain = sample.loc[peak_index, "Деформация"]

    # Деформация в последней точке испытания.
    final_strain = sample.iloc[-1]["Деформация"]

    # Индекс последней точки измерения.
    last_index = len(sample) - 1

    # Если максимум напряжения находится в последней точке,
    # значит испытание завершилось до разрушения образца.
    # Такой образец считаем некорректным.
    if peak_index == last_index:
        bad_samples.append(sample_id)

# Выводим список найденных некорректных образцов.
print("\nПлохие образцы:")
print(bad_samples)

print(f"\nВсего плохих образцов: {len(bad_samples)}")

# Удаляем из общего DataFrame все строки,
# относящиеся к некорректным образцам.
clean_df = df[~df["sample_id"].isin(bad_samples)]

# Проверяем, сколько строк было удалено.
print(f"До очистки: {len(df)} строк")
print(f"После очистки: {len(clean_df)} строк")

# Контрольная проверка.
# Должно вывести False, если плохие образцы полностью удалены.
print(clean_df["sample_id"].isin(bad_samples).any())

# После реализации второй проверки можно сохранить результат:
clean_df.to_csv("data/CleanData.csv", sep=";", index=False)