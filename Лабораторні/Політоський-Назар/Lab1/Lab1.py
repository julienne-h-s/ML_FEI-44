import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro, probplot

# =====================
# 2. Завантаження та EDA
# =====================
# Завантажуємо датасет
df = pd.read_csv("Student_Performance.csv")

# 🔹 Перетворення категоріальної змінної у числову одразу після завантаження
# Це потрібно для подальшого використання у регресії
if "Extracurricular Activities" in df.columns:
    df['Extracurricular Activities'] = (
        df['Extracurricular Activities'].map({'No': 0, 'Yes': 1}).astype(int)
    )

# Виводимо загальну інформацію про датасет (типи колонок, пропущені значення)
print("🔹 Інформація про датасет:")
print(df.info())
print(df.describe())

# Перевірка на наявність пропущених значень
print("\n🔹 Перевірка пропущених значень:")
print(df.isnull().sum())

# 🔹 Візуалізація кореляційної матриці
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Кореляція ознак")
plt.show()

# 🔹 Перевірка правильності кодування категоріальних змінних
print("\n🔹 Перевірка типів даних перед train/test:")
print(df.dtypes)

print("\n🔹 Унікальні значення Extracurricular Activities:")
print(df["Extracurricular Activities"].unique())

# =====================
# 3. Попередня обробка
# =====================
# Вибираємо цільову змінну та ознаки
y = df["Performance Index"]  # Змінна, яку будемо прогнозувати
X = df.drop("Performance Index", axis=1)  # Всі інші колонки — ознаки

# 🔹 Поділ на тренувальний та тестовий набори (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔹 Масштабування ознак (важливо для градієнтного спуску)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🔹 Перевірка на NaN після масштабування
print("\n🔹 Перевірка NaN після масштабування:")
print("NaN у X_train:", np.isnan(X_train_scaled).sum())
print("NaN у X_test:", np.isnan(X_test_scaled).sum())

# 🔹 Страховка: замінюємо NaN та inf на 0
X_train_scaled = np.nan_to_num(X_train_scaled)
X_test_scaled = np.nan_to_num(X_test_scaled)

# 🔹 Додаємо стовпець одиниць для β0 (вільного члена регресії)
X_train_final = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]
X_test_final = np.c_[np.ones(X_test_scaled.shape[0]), X_test_scaled]

# =====================
# 4. Градієнтний спуск
# =====================
# Функція для обчислення середньоквадратичної помилки
def compute_cost(X, y, b):
    predictions = X.dot(b)  # Обчислюємо прогноз
    errors = predictions - y  # Обчислюємо похибку
    mse = (1 / len(y)) * np.sum(errors ** 2)  # MSE
    return mse

# Основна функція градієнтного спуску
def gradient_descent(X, y, b, learning_rate=0.01, epochs=1000):
    cost_history = []  # Історія помилок для візуалізації збіжності
    m = len(y)  # Кількість спостережень
    for _ in range(epochs):
        predictions = X.dot(b)  # Поточні прогнози
        errors = predictions - y  # Розрахунок похибки
        gradient = (2 / m) * X.T.dot(errors)  # Градієнт MSE
        b = b - learning_rate * gradient  # Оновлюємо коефіцієнти
        cost_history.append(compute_cost(X, y, b))  # Зберігаємо поточну помилку
    return b, cost_history

# 🔹 Ініціалізація коефіцієнтів нулями
b_init = np.zeros(X_train_final.shape[1])

# 🔹 Навчання моделі методом градієнтного спуску
b_final, cost_history = gradient_descent(
    X_train_final, y_train.values, b_init, 0.01, 1000
)

# 🔹 Візуалізація збіжності градієнтного спуску
plt.plot(range(len(cost_history)), cost_history)
plt.xlabel("Ітерації")
plt.ylabel("MSE")
plt.title("Збіжність градієнтного спуску")
plt.show()

# 🔹 Виведення фінальних коефіцієнтів
print("🔹 Фінальні коефіцієнти:")
print(b_final)

# =====================
# 5. Оцінка моделі
# =====================
# Прогнозування на тестовому наборі
y_pred = X_test_final.dot(b_final)

# Обчислення MSE та R² на тесті
mse_test = np.mean((y_test - y_pred) ** 2)
r2 = 1 - (
    np.sum((y_test - y_pred) ** 2)
    / np.sum((y_test - np.mean(y_test)) ** 2)
)

print(f"\nMSE (тест): {mse_test:.4f}")
print(f"R² (тест): {r2:.4f}")

# =====================
# 6. Перевірка 6 припущень МНК
# =====================

# 1. Лінійність та гомоскедастичність
residuals = y_test - y_pred  # Залишки
plt.scatter(y_pred, residuals)
plt.axhline(0, color="red")
plt.xlabel("Прогнозовані значення")
plt.ylabel("Залишки")
plt.title("Залишки vs Прогнозовані значення")
plt.show()

# 2. Мультиколінеарність (VIF)
vif = pd.DataFrame()
vif["VIF"] = [
    variance_inflation_factor(X_train_scaled, i)
    for i in range(X_train_scaled.shape[1])
]
vif["feature"] = X.columns
print("\n🔹 VIF показники:")
print(vif)

# 3. Нормальність залишків
probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q plot залишків")
plt.show()

shapiro_test = shapiro(residuals)  # Shapiro-Wilk тест на нормальність
print("\n🔹 Shapiro-Wilk test:", shapiro_test)

# 4. Автокореляція залишків
dw = durbin_watson(residuals)
print("\n🔹 Durbin-Watson:", dw)

# 5. Викиди (Cook’s Distance)
model = sm.OLS(y_train, X_train_final).fit()  # OLS на тренувальних даних
influence = model.get_influence()  # Вплив кожного спостереження
cooks_d = influence.cooks_distance[0]

# Візуалізація Cook’s Distance
plt.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",")
plt.axhline(4 / len(y_train), color="red", linestyle="--")  # Порогове значення
plt.title("Cook's Distance")
plt.show()
