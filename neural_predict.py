import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================================
# Загрузка обученной модели
# ==========================================================

print("Загрузка модели...")

model = joblib.load("models/model.pkl")
scaler_X = joblib.load("models/scaler_X.pkl")
scaler_y = joblib.load("models/scaler_y.pkl")

print("Модель успешно загружена.")

# ==========================================================
# Загружаем очищенный датасет
# Он нужен только для определения диапазона деформации
# ==========================================================

df = pd.read_csv(
    "data/CleanData.csv",
    sep=";",
    low_memory=False
)

df.columns = df.columns.str.strip()

# ==========================================================
# Диапазон деформации
# ==========================================================

min_def = df["Деформация"].min()
max_def = df["Деформация"].max()

deformation = np.linspace(min_def, max_def, 300)

# ==========================================================
# Константы физической модели
# ==========================================================

fiber_E = 240000
matrix_E = 2700

# ==========================================================
# Базовые параметры материала
# От них будут создаваться новые виртуальные образцы
# ==========================================================

base_polymer = df["Раствор полимера,%"].median()
base_length = df["Длина, мм"].median()
base_mass = df["Масса, мг"].median()
base_fiber = df["Содержание волокна, %"].median()

# ==========================================================
# Сколько образцов сгенерировать
# ==========================================================

NUM_SAMPLES = 5

# ==========================================================
# Здесь будут храниться все результаты
# ==========================================================

all_samples = []

plt.figure(figsize=(10,6))

# ==========================================================
# Генерация новых образцов
# ==========================================================

for sample_number in range(NUM_SAMPLES):

    # ------------------------------------------------------
    # Немного изменяем параметры материала.
    # Это позволяет получать разные виртуальные образцы.
    # ------------------------------------------------------

    polymer = base_polymer + np.random.uniform(-1.0, 1.0)

    fiber = base_fiber + np.random.uniform(-2.0, 2.0)

    mass = base_mass + np.random.uniform(-5, 5)

    length = base_length + np.random.uniform(-3, 3)

    # ------------------------------------------------------
    # Эффективный модуль упругости
    # ------------------------------------------------------

    fiber_fraction = fiber / 100

    effective_E = (
        fiber_E * fiber_fraction
        + matrix_E * (1 - fiber_fraction)
    )

   # ------------------------------------------------------
# Формируем признаки для нейронной сети.
# Для каждой точки деформации используются
# одинаковые параметры материала.
# ------------------------------------------------------

    X = np.column_stack([
        deformation,
        np.full_like(deformation, polymer),
        np.full_like(deformation, length),
        np.full_like(deformation, mass),
        np.full_like(deformation, fiber)
    ])

    # Масштабируем признаки перед подачей в модель.
    X_scaled = scaler_X.transform(X)

    residual_scaled = model.predict(X_scaled)

    residual = scaler_y.inverse_transform(

        residual_scaled.reshape(-1,1)

    ).ravel()

    # ------------------------------------------------------
    # Физическая часть
    # ------------------------------------------------------

    physical_stress = effective_E * (deformation / 100)

    # ------------------------------------------------------
    # Итоговое напряжение
    # ------------------------------------------------------

    stress = physical_stress + residual

    # ------------------------------------------------------
    # Сохраняем образец
    # ------------------------------------------------------

    sample_df = pd.DataFrame({

        "sample_id": f"Generated_{sample_number+1}",

        "Деформация": deformation,

        "Напряжение": stress,

        "Раствор полимера,%": polymer,

        "Длина, мм": length,

        "Масса, мг": mass,

        "Содержание волокна, %": fiber

    })

    all_samples.append(sample_df)

    plt.plot(
        deformation,
        stress,
        linewidth=2,
        label=f"Generated {sample_number+1}"
    )

# Создаем папку output, если её нет
os.makedirs("output", exist_ok=True)

result = pd.concat(all_samples, ignore_index=True)

result.to_csv(
    "output/generated_samples.csv",
    sep=";",
    index=False
)

# Также сохраняем каждый образец на отдельный лист Excel
with pd.ExcelWriter(
    "output/generated_samples.xlsx",
    engine="openpyxl"
) as writer:
    for sample_number, sample_df in enumerate(all_samples, start=1):
        sample_df.to_excel(
            writer,
            sheet_name=f"Образец_{sample_number}",
            index=False
        )


# ==========================================================
# Информация о завершении работы
# ==========================================================

print()
print("===================================")
print("Генерация завершена.")
print("===================================")
print()

print(f"Сгенерировано образцов: {NUM_SAMPLES}")

print("Сохранены файлы:")
print(" - generated_samples.csv")
print(" - generated_samples.xlsx")

# ==========================================================
# Визуализация
# ==========================================================

plt.title("Сгенерированные виртуальные образцы")

plt.xlabel("Деформация")

plt.ylabel("Напряжение")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()