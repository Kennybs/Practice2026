import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ==========================================================
# Константы физической модели
# ==========================================================

# Модуль Юнга углеродного волокна (МПа)
FIBER_E = 240000

# Модуль Юнга полимерной матрицы (МПа)
MATRIX_E = 2700

# Квадратичный член физической модели
K2 = 0.0


# ==========================================================
# Загрузка очищенного датасета
# ==========================================================

print("Загрузка данных...")

df = pd.read_csv(
    "data/CleanData.csv",
    sep=";",
    low_memory=False
)

df.columns = df.columns.str.strip()

print(f"Количество строк: {len(df)}")
print(f"Количество образцов: {df['sample_id'].nunique()}")


# ==========================================================
# Формирование обучающей выборки
# ==========================================================

print("\nПодготовка обучающих данных...")

X_list = []
y_real = []
stress_phys_list = []

# Проходим по каждому образцу отдельно
for sample_id, sample in df.groupby("sample_id"):

    sample = sample.reset_index(drop=True)

    deformation = sample["Деформация"].values
    stress = sample["Напряжение"].values

    polymer = sample["Раствор полимера,%"].iloc[0]
    length = sample["Длина, мм"].iloc[0]
    mass = sample["Масса, мг"].iloc[0]
    fiber = sample["Содержание волокна, %"].iloc[0]

    # Объемная доля волокна
    vf = fiber / 100

    # Эффективный модуль Юнга
    e_eff = FIBER_E * vf + MATRIX_E * (1 - vf)

    # Физическая модель
    stress_phys = (
        e_eff * (deformation / 100)
        + K2 * (deformation / 100) ** 2
    )

    for i in range(len(sample)):

        X_list.append([
            deformation[i],
            polymer,
            length,
            mass,
            fiber
        ])

        y_real.append(stress[i])
        stress_phys_list.append(stress_phys[i])

X = np.array(X_list)

y_real = np.array(y_real)

stress_phys = np.array(stress_phys_list)

print("Размер матрицы признаков:", X.shape)


# ==========================================================
# Остатки физической модели
# ==========================================================

print("\nРасчет остатков...")

residual = y_real - stress_phys


# ==========================================================
# Масштабирование
# ==========================================================

print("Масштабирование данных...")

scaler_x = StandardScaler()

X_scaled = scaler_x.fit_transform(X)

scaler_y = StandardScaler()

residual_scaled = scaler_y.fit_transform(
    residual.reshape(-1, 1)
).ravel()


# ==========================================================
# Обучение нейронной сети
# ==========================================================

print("\nОбучение нейронной сети...")

model = MLPRegressor(

    hidden_layer_sizes=(100, 100),

    activation="relu",

    solver="adam",

    max_iter=5000,

    random_state=42

)

model.fit(X_scaled, residual_scaled)

print("Обучение завершено.")


# ==========================================================
# Проверка качества модели
# ==========================================================

print("\nОценка качества модели...")

residual_pred = scaler_y.inverse_transform(

    model.predict(X_scaled).reshape(-1, 1)

).ravel()

stress_pred = stress_phys + residual_pred

mse = mean_squared_error(y_real, stress_pred)

mae = mean_absolute_error(y_real, stress_pred)

r2 = r2_score(y_real, stress_pred)

print(f"MSE : {mse:.6f}")
print(f"MAE : {mae:.6f}")
print(f"R²  : {r2:.6f}")


# ==========================================================
# Сохранение модели
# ==========================================================

print("\nСохранение модели...")

joblib.dump(model, "model.pkl")

joblib.dump(scaler_x, "scaler_x.pkl")

joblib.dump(scaler_y, "scaler_y.pkl")

print("Модель сохранена.")


# ==========================================================
# Итоговая информация
# ==========================================================

print("\n===========================")
print("Обучение успешно завершено.")
print("===========================")
print(f"Образцов: {df['sample_id'].nunique()}")
print(f"Точек обучения: {len(X)}")
print(f"MSE = {mse:.6f}")
print(f"MAE = {mae:.6f}")
print(f"R²  = {r2:.6f}")