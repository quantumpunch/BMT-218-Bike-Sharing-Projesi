import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 🔹 1. Veriyi yükle
df = pd.read_csv('day.csv')

# 🔹 2. Gereksiz sütunları çıkar
df = df.drop(['instant', 'dteday', 'casual', 'registered'], axis=1)

# 🔹 3. Anlamlı kategori etiketlerini ata
df['season'] = df['season'].map({1: 'İlkbahar', 2: 'Yaz', 3: 'Sonbahar', 4: 'Kış'})
df['weathersit'] = df['weathersit'].map({1: 'Açık', 2: 'Bulutlu', 3: 'Yağmurlu', 4: 'Şiddetli'})
df['weekday'] = df['weekday'].map({
    0: 'Pazartesi', 1: 'Salı', 2: 'Çarşamba', 3: 'Perşembe',
    4: 'Cuma', 5: 'Cumartesi', 6: 'Pazar'
})
df['mnth'] = df['mnth'].map({
    1: 'Ocak', 2: 'Şubat', 3: 'Mart', 4: 'Nisan',
    5: 'Mayıs', 6: 'Haziran', 7: 'Temmuz', 8: 'Ağustos',
    9: 'Eylül', 10: 'Ekim', 11: 'Kasım', 12: 'Aralık'
})
df['holiday'] = df['holiday'].map({0: 'Hayır', 1: 'Evet'})
df['workingday'] = df['workingday'].map({0: 'Hayır', 1: 'Evet'})
df['yr'] = df['yr'].map({0: '2011', 1: '2012'})

# Değişkenleri orijinal değerlere dönüştür
df['temp'] = df['temp'] * 41
df['atemp'] = df['atemp'] * 50
df['hum'] = df['hum'] * 100
df['windspeed'] = df['windspeed'] * 67


# 4. Türkçe etiket sözlüğü
var_name_map = {
    'season': 'Mevsim',
    'yr': 'Yıl',
    'mnth': 'Ay',
    'holiday': 'Resmi Tatil',
    'weekday': 'Haftanın Günü',
    'workingday': 'İş Günü',
    'weathersit': 'Hava Durumu',
    'temp': 'Sıcaklık (°C)',
    'atemp': 'Hissedilen Sıcaklık (°C)',
    'hum': 'Nem (%)',
    'windspeed': 'Rüzgar Hızı (km/h)',
    'cnt': 'Kiralanan Bisiklet'
}

# 🔹 5. Temel istatistiksel bilgiler
print("🔎 Temel İstatistiksel Bilgiler:\n")
print(df.describe(include='all'))

# 🔹 6. Korelasyon matrisi
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f",
            xticklabels=[var_name_map.get(col, col) for col in df.select_dtypes(include=['float64', 'int64']).columns],
            yticklabels=[var_name_map.get(col, col) for col in df.select_dtypes(include=['float64', 'int64']).columns])
plt.title("Korelasyon Matrisi")
plt.show()

# 🔹 8. Kategorik değişkenler ve hedef değişken analizi
categorical_cols = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']

for col in categorical_cols:
    plt.figure(figsize=(10, 4))
    sns.barplot(x=col, y='cnt', data=df, ci=None, estimator='mean')
    plt.title(f"Ortalama {var_name_map.get('cnt')} ({var_name_map.get(col)})")
    plt.xlabel(var_name_map.get(col))
    plt.ylabel(var_name_map.get('cnt'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 🔹 9. Boxplot: Kategorik değişkenlere göre cnt dağılımı ve aykırı değerlerin gösterilmesi
for col in categorical_cols:
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=col, y='cnt', data=df)
    plt.title(f"{var_name_map.get('cnt')} Kutu Grafiği ({var_name_map.get(col)})")
    plt.xlabel(var_name_map.get(col))
    plt.ylabel(var_name_map.get('cnt'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 🔹 10. Sayısal değişkenlerin cnt ile korelasyonu
correlations = df.corr(numeric_only=True)['cnt'].drop('cnt').sort_values(ascending=False)
print("\n'Kiralanan Bisiklet Sayısı' ile Korelasyonlar (yüksekten düşüğe):\n")
for var, corr in correlations.items():
    print(f"{var_name_map.get(var, var)}: {corr:.2f}")

# 🔹 11. Korelasyon grafiği (Türkçe)
plt.figure(figsize=(10, 5))
sns.barplot(x=correlations.values, y=[var_name_map.get(v, v) for v in correlations.index])
plt.title("'Kiralanan Bisiklet Sayısı' ile Korelasyonu Olan Değişkenler")
plt.xlabel("Korelasyon Katsayısı")
plt.ylabel("Değişken")
plt.tight_layout()
plt.show()
