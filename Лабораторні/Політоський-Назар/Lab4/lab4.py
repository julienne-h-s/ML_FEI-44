import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, matthews_corrcoef
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Завантаження та підготовка даних
# -------------------------------
df = pd.read_csv("synthetic_coffee_health_10000.csv")  # файл із Kaggle
df = df.drop(columns=['ID'])

y = df['Health_Issues']
X = df.drop(columns=['Health_Issues'])

# Перетворюємо мітки у числа
le = LabelEncoder()
y = le.fit_transform(y)

# ========================================================
# Інженерія Ознак та Feature Selection
# ========================================================
X['Coffee_Sleep_Interaction'] = X['Coffee_Intake'] * X['Sleep_Hours']
X['Caffeine_per_Hour'] = X['Caffeine_mg'] / (X['Sleep_Hours'] + 1)
X['Age_Binned'] = pd.cut(X['Age'], bins=[18, 30, 50, 80], labels=['Young', 'Middle', 'Old'])

cat_cols = ['Gender', 'Country', 'Sleep_Quality', 'Stress_Level',
            'Occupation', 'Smoking', 'Alcohol_Consumption', 'Age_Binned']
X = pd.get_dummies(X, columns=cat_cols)

scaler = StandardScaler()
num_cols = ['Age', 'Coffee_Intake', 'Caffeine_mg', 'Sleep_Hours',
            'BMI', 'Heart_Rate', 'Physical_Activity_Hours',
            'Coffee_Sleep_Interaction', 'Caffeine_per_Hour']
X[num_cols] = scaler.fit_transform(X[num_cols])

X = X.fillna(X.mean(numeric_only=True))
X = X.fillna(0)

# Початкове дерево для оцінки важливості ознак
dt_fs = DecisionTreeClassifier(criterion='gini', max_depth=8, random_state=42)
dt_fs.fit(X, y)
importances = pd.Series(dt_fs.feature_importances_, index=X.columns).sort_values(ascending=False)

selected_features = importances.head(10).index
print("\nТоп-10 відібраних ознак за важливістю:")
print(selected_features.tolist())

X = X[selected_features]

# -------------------------------
# Розбиття на train / validation / test
# -------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

X_train_np = X_train.values
X_val_np = X_val.values
X_test_np = X_test.values
y_train_np = y_train
y_val_np = y_val
y_test_np = y_test

# -------------------------------
# Власне дерево з неагресивним прунінгом
# -------------------------------
class MyDecisionTree:
    def __init__(self, max_depth=8, min_samples=2):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.tree = None

    def gini(self, y):
        m = len(y)
        if m == 0: return 0
        p = [np.sum(y == c) / m for c in np.unique(y)]
        return 1 - sum(pi**2 for pi in p)

    def best_split(self, X, y):
        best_gini = float('inf')
        best_feature, best_threshold = None, None
        m, n = X.shape
        for feature in range(n):
            for t in np.unique(X[:, feature]):
                left = y[X[:, feature] <= t]
                right = y[X[:, feature] > t]
                if len(left) == 0 or len(right) == 0:
                    continue
                g = (len(left)/m)*self.gini(left) + (len(right)/m)*self.gini(right)
                if g < best_gini:
                    best_gini = g
                    best_feature, best_threshold = feature, t
        return best_feature, best_threshold

    def build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples:
            from collections import Counter
            most_common = Counter(y).most_common(1)[0][0]
            return {'leaf': True, 'class': most_common}

        feature, threshold = self.best_split(X, y)
        if feature is None:
            from collections import Counter
            most_common = Counter(y).most_common(1)[0][0]
            return {'leaf': True, 'class': most_common}

        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold
        left = self.build_tree(X[left_mask], y[left_mask], depth+1)
        right = self.build_tree(X[right_mask], y[right_mask], depth+1)
        from collections import Counter
        most_common = Counter(y).most_common(1)[0][0]
        return {'leaf': False, 'feature': feature, 'threshold': threshold,
                'left': left, 'right': right,
                'class': most_common}

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)
        return self

    def predict_one(self, x, node=None):
        node = self.tree if node is None else node
        if node['leaf']:
            return node['class']
        if x[node['feature']] <= node['threshold']:
            return self.predict_one(x, node['left'])
        else:
            return self.predict_one(x, node['right'])

    def predict(self, X):
        return np.array([self.predict_one(sample) for sample in X])

    def calculate_error(self, X, y, node):
        preds = [self.predict_one(x, node) for x in X]
        return np.mean(preds != y)

    def count_leaves(self, node):
        if node['leaf']: return 1
        return self.count_leaves(node['left']) + self.count_leaves(node['right'])

    # --------- Неагресивний прунінг ----------
    def prune_tree(self, node, alpha, validation_X, validation_y):
        if 'leaf' in node:
            return self.calculate_error(validation_X, validation_y, node)
        
        left_error = self.prune_tree(node['left'], alpha, validation_X, validation_y)
        right_error = self.prune_tree(node['right'], alpha, validation_X, validation_y)
        subtree_error = left_error + right_error

        leaf_error = self.calculate_error(
            validation_X, validation_y,
            {'leaf': True, 'class': node['class']}
        )

        if leaf_error <= subtree_error + alpha:
            node['leaf'] = True
            del node['left'], node['right'], node['feature'], node['threshold']
            return leaf_error
        return subtree_error

# -------------------------------
# Навчання дерева
# -------------------------------
tree = MyDecisionTree(max_depth=8, min_samples=5)
tree.fit(X_train_np, y_train_np)

print("=== Власне дерево (до прунінгу) ===")
y_pred_my = tree.predict(X_test_np)
print("Accuracy:", accuracy_score(y_test_np, y_pred_my))
print("F1-score:", f1_score(y_test_np, y_pred_my, average='weighted'))
print("MCC:", matthews_corrcoef(y_test_np, y_pred_my))

# Прунінг
alpha = 0.001
tree.prune_tree(tree.tree, alpha, X_val_np, y_val_np)

print("\n=== Власне дерево (після прунінгу) ===")
y_pred_pruned = tree.predict(X_test_np)
print("Accuracy:", accuracy_score(y_test_np, y_pred_pruned))
print("F1-score:", f1_score(y_test_np, y_pred_pruned, average='weighted'))
print("MCC:", matthews_corrcoef(y_test_np, y_pred_pruned))

# -------------------------------
# sklearn DecisionTree
# -------------------------------
dt_sk = DecisionTreeClassifier(criterion='gini', max_depth=8, random_state=42)
dt_sk.fit(X_train_np, y_train_np)
y_pred_sk = dt_sk.predict(X_test_np)

print("\n=== sklearn DecisionTree ===")
print("Accuracy:", accuracy_score(y_test_np, y_pred_sk))
print("F1-score:", f1_score(y_test_np, y_pred_sk, average='weighted'))
print("MCC:", matthews_corrcoef(y_test_np, y_pred_sk))

# -------------------------------
# Random Forest (точніший)
# -------------------------------
rf = RandomForestClassifier(n_estimators=300, max_features='sqrt', random_state=42)
rf.fit(X_train_np, y_train_np)
y_pred_rf = rf.predict(X_test_np)

print("\n=== RandomForest ===")
print("Accuracy:", accuracy_score(y_test_np, y_pred_rf))
print("F1-score:", f1_score(y_test_np, y_pred_rf, average='weighted'))
print("MCC:", matthews_corrcoef(y_test_np, y_pred_rf))

# -------------------------------
# Confusion Matrix для всіх моделей
# -------------------------------
models = {
    "MyDecisionTree (pruned)": y_pred_pruned,
    "sklearn DecisionTree": y_pred_sk,
    "RandomForest": y_pred_rf
}

plt.figure(figsize=(18, 5))

for i, (name, y_pred) in enumerate(models.items(), 1):
    cm = confusion_matrix(y_test_np, y_pred)
    plt.subplot(1, 3, i)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix\n{name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")

plt.tight_layout()
plt.show()
