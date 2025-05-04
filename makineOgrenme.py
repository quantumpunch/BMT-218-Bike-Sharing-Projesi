import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error, f1_score
)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


# F1 skoru için sınıflandırma dönüşümü
def regression_f1(y_true, y_pred, threshold=4500):
    y_true_cls = (y_true >= threshold).astype(int)
    y_pred_cls = (y_pred >= threshold).astype(int)
    return f1_score(y_true_cls, y_pred_cls)

# Toleranslı doğruluk (%10 sapma içinde olan tahminler)
def regression_accuracy(y_true, y_pred, tolerance=0.1):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred) <= (tolerance * y_true)) * 100  # yüzde

# Veri hazırlığı
df = pd.read_csv("day.csv")
df = df.drop(['instant', 'dteday', 'casual', 'registered'], axis=1)
X = df.drop('cnt', axis=1)
y = df['cnt']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18)

# Ölçekleme (KNN için)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Lineer Regresyon": LinearRegression(),
    "Karar Ağaçları": DecisionTreeRegressor(random_state=18),
    "Rastgele Orman": RandomForestRegressor(n_estimators=720, random_state=18),
    "K-En Yakın Komşu": KNeighborsRegressor(n_neighbors=5),

}

results = []

for name, model in models.items():
    X_cv = X_train_scaled if name in ['KNN Regressor'] else X_train
    X_eval = X_test_scaled if name in ['KNN Regressor'] else X_test

    r2_scores = cross_val_score(model, X_cv, y_train, cv=5, scoring='r2')
    neg_mse_scores = cross_val_score(model, X_cv, y_train, cv=5, scoring='neg_mean_squared_error')

    start_time = time.time()
    model.fit(X_cv, y_train)
    y_pred = model.predict(X_eval)
    elapsed_time = time.time() - start_time

    r2 = np.mean(r2_scores)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    acc = regression_accuracy(y_test, y_pred)
    f1 = regression_f1(y_test, y_pred)

    results.append({
        "Model": name,
        "Doğruluk R²": r2,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE (%)": mape,
        "İsabetlilik (±10%)": acc,
        "F1 Skoru": f1,
        "Zaman (s)": elapsed_time
    })

results_df = pd.DataFrame(results).sort_values(by="İsabetlilik (±10%)", ascending=False)

# ---------- Görselleştirme (sadece matplotlib) ---------- #
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')
table = ax.table(cellText=results_df.round(4).values,
                 colLabels=results_df.columns,
                 cellLoc='center',
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
plt.title("Regresyon Model Karşılaştırma Tablosu", fontsize=14, pad=5)
plt.tight_layout()
plt.show()

# Ayrıca DataFrame olarak da çıktı
print(results_df)
