"""
Модуль для множинної лінійної регресії з нуля
Реалізація градієнтного спуску та аналізу моделі
"""

import numpy as np
import pandas as pd
from typing import Tuple, List


class LinearRegressionGD:
    """
    Множинна лінійна регресія з градієнтним спуском
    
    Параметри:
    -----------
    learning_rate : float, default=0.01
        Швидкість навчання (alpha)
    epochs : int, default=1000
        Кількість ітерацій навчання
    verbose : bool, default=False
        Виводити прогрес навчання
    """
    
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000, verbose: bool = False):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose = verbose
        self.weights = None
        self.cost_history = []
        
    def compute_cost(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        """
        Обчислює середню квадратичну помилку (MSE)
        
        Параметри:
        ----------
        X : numpy array, shape (n_samples, n_features)
            Матриця ознак
        y : numpy array, shape (n_samples,)
            Вектор цільових значень
        weights : numpy array, shape (n_features,)
            Вектор ваг
            
        Повертає:
        ---------
        mse : float
            Середня квадратична помилка
        """
        predictions = X.dot(weights)
        errors = predictions - y
        mse = (1 / len(y)) * np.sum(errors ** 2)
        return mse
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionGD':
        """
        Навчання моделі методом градієнтного спуску
        
        Параметри:
        ----------
        X : numpy array, shape (n_samples, n_features)
            Матриця ознак (вже з intercept якщо потрібно)
        y : numpy array, shape (n_samples,)
            Вектор цільових значень
            
        Повертає:
        ---------
        self : LinearRegressionGD
            Навчена модель
        """
        n_samples, n_features = X.shape
        
        # Ініціалізація ваг
        self.weights = np.zeros(n_features)
        
        # Градієнтний спуск
        for epoch in range(self.epochs):
            # Передбачення
            predictions = X.dot(self.weights)
            
            # Помилка
            errors = predictions - y
            
            # Градієнт
            gradient = (1 / n_samples) * X.T.dot(errors)
            
            # Оновлення ваг
            self.weights = self.weights - self.learning_rate * gradient
            
            # Збереження функції втрат
            cost = self.compute_cost(X, y, self.weights)
            self.cost_history.append(cost)
            
            # Вивід прогресу
            if self.verbose and (epoch + 1) % 100 == 0:
                print(f"Епоха {epoch + 1}/{self.epochs}, MSE: {cost:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Прогнозування за допомогою навченої моделі
        
        Параметри:
        ----------
        X : numpy array, shape (n_samples, n_features)
            Матриця ознак
            
        Повертає:
        ---------
        predictions : numpy array, shape (n_samples,)
            Передбачені значення
        """
        if self.weights is None:
            raise ValueError("Модель не навчена. Спочатку викличте fit().")
        
        return X.dot(self.weights)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Обчислює R² score
        
        Параметри:
        ----------
        X : numpy array, shape (n_samples, n_features)
            Матриця ознак
        y : numpy array, shape (n_samples,)
            Вектор цільових значень
            
        Повертає:
        ---------
        r2 : float
            R² score
        """
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2


def calculate_vif(X: np.ndarray, feature_names: List[str] = None) -> pd.DataFrame:
    """
    Обчислює Variance Inflation Factor для кожної ознаки
    
    Параметри:
    ----------
    X : numpy array, shape (n_samples, n_features)
        Матриця ознак (без intercept)
    feature_names : list, optional
        Назви ознак
        
    Повертає:
    ---------
    vif_df : pandas DataFrame
        Таблиця з VIF для кожної ознаки
    """
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    vif_data = []
    
    for i in range(X.shape[1]):
        # i-та ознака як цільова
        y_i = X[:, i]
        # Інші ознаки як предиктори
        X_i = np.delete(X, i, axis=1)
        
        # Додаємо intercept
        X_i_with_intercept = np.c_[np.ones(X_i.shape[0]), X_i]
        
        # Навчання простої регресії
        model = LinearRegressionGD(learning_rate=0.01, epochs=500, verbose=False)
        model.fit(X_i_with_intercept, y_i)
        
        # Обчислення R²
        r2 = model.score(X_i_with_intercept, y_i)
        
        # VIF
        vif = 1 / (1 - r2) if r2 < 0.9999 else float('inf')
        vif_data.append({'Feature': feature_names[i], 'VIF': vif})
    
    return pd.DataFrame(vif_data)


def cooks_distance(X: np.ndarray, y: np.ndarray, weights: np.ndarray, 
                  residuals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Обчислює Cook's Distance для виявлення впливових спостережень
    
    Параметри:
    ----------
    X : numpy array, shape (n_samples, n_features)
        Матриця ознак (з intercept)
    y : numpy array, shape (n_samples,)
        Вектор цільових значень
    weights : numpy array, shape (n_features,)
        Вектор ваг моделі
    residuals : numpy array, shape (n_samples,)
        Залишки моделі
        
    Повертає:
    ---------
    cooks_d : numpy array
        Cook's Distance для кожного спостереження
    leverage : numpy array
        Leverage (hat values) для кожного спостереження
    """
    n = len(y)
    p = X.shape[1]
    
    # Leverage (hat values)
    hat_matrix_diag = np.sum(X * np.linalg.pinv(X.T @ X) @ X.T, axis=1)
    
    # MSE
    mse = np.mean(residuals ** 2)
    
    # Стандартизовані залишки
    standardized_residuals = residuals / np.sqrt(mse * (1 - hat_matrix_diag))
    
    # Cook's Distance
    cooks_d = (standardized_residuals ** 2 / p) * (hat_matrix_diag / (1 - hat_matrix_diag) ** 2)
    
    return cooks_d, hat_matrix_diag


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Обчислює набір метрик для оцінки моделі
    
    Параметри:
    ----------
    y_true : numpy array
        Фактичні значення
    y_pred : numpy array
        Передбачені значення
        
    Повертає:
    ---------
    metrics : dict
        Словник з метриками (MSE, RMSE, MAE, R²)
    """
    # MSE
    mse = np.mean((y_true - y_pred) ** 2)
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }


# Приклад використання
if __name__ == "__main__":
    # Генерація тестових даних
    np.random.seed(42)
    X = np.random.randn(100, 3)
    X_with_intercept = np.c_[np.ones(100), X]
    true_weights = np.array([1.5, 2.0, -1.0, 0.5])
    y = X_with_intercept.dot(true_weights) + np.random.randn(100) * 0.1
    
    # Навчання моделі
    model = LinearRegressionGD(learning_rate=0.01, epochs=1000, verbose=True)
    model.fit(X_with_intercept, y)
    
    # Прогнозування
    y_pred = model.predict(X_with_intercept)
    
    # Оцінка
    metrics = evaluate_model(y, y_pred)
    print("\nМетрики:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nВаги моделі: {model.weights}")
    print(f"Справжні ваги: {true_weights}")
