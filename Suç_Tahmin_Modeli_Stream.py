import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', '{:.2f}'.format)


# Başlık
st.title("Suç Tahmin Modeli")

# Veriyi yükle
data_path="C:/Users/almak/Desktop/yetkin/DSMLBC/16.DÖNEM/Suç Tahmin Modeli/Crime_Data_from_2020_to_Present_Proje.csv"
st.header("Veri Setininin Yüklenmesi")
df_ = pd.read_csv(data_path)
df = df_.copy()
st.success("Veri başarıyla yüklendi!")

# Veri Setini Gösterme
st.header("Veri Seti")
st.write("Yüklenen veri seti:")
st.dataframe(df.head())

# Veri Hakkında Özet Bilgi
st.header("Veri Seti Özeti")
st.write("Veri seti boyutu:")
st.write(df.shape)
st.write("İlk 5 Satır")
st.write(df.head())
st.write("Veri seti istatiksel özeti")
st.write(df.describe())
st.write("Veri setinin kolonları")
st.write(df.columns)
st.write("Veri setindeki eksik değer analizi")
st.write(df.isnull().sum().sort_values(ascending=False).to_frame(name='Eksik Değer Sayısı').assign(Oran=lambda x: round((x['Eksik Değer Sayısı'] / len(df)) * 100, 2)))
eksik_degerler = (df.isnull().sum().sort_values(ascending=False).to_frame(name='Eksik Değer Sayısı').assign(Oran=lambda x: round((x['Eksik Değer Sayısı'] / len(df)) * 100, 2)))
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(eksik_degerler.index, eksik_degerler['Eksik Değer Sayısı'], color='gray', label='Eksik Değer Sayısı')
ax.set_xticks([])
ax.set_title('Eksik Değer Sayısı ve Oranları', fontsize=16)
ax.set_xlabel('Sütunlar', fontsize=12)
ax.set_ylabel('Değer', fontsize=12)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Streamlit'te grafiği göster
st.subheader("Eksik Değer Grafiği")
st.pyplot(fig)

def baslik_duzenle(df):
    df.columns = df.columns.str.replace(" ", "_")
    df.columns = df.columns.str.upper()
    return df.head()

st.write("Veri seti kolon başlıklarının düzenlenmesi:")
st.write(baslik_duzenle(df))

def bos_sütun_ve_değerleri_sil(df):
    df.drop(["CRM_CD_4", "CRM_CD_3", "CRM_CD_2", "CROSS_STREET", "WEAPON_DESC", "WEAPON_USED_CD"], axis=1, inplace=True)
    return df.head()

st.write("Veri seti boş değerlerin silinmesi:")
st.write(bos_sütun_ve_değerleri_sil(df))
st.write("Veri setinin yeni boyutu:")
st.write(df.shape)


def tarihi_parcala(df):
    df['DATE_OCC'] = pd.to_datetime(df['DATE_OCC'], errors='coerce')
    df['DATE_OCC'] = pd.to_datetime(df['DATE_OCC'], format='%Y-%m-%d', errors='coerce')
    df['YEAR_OCC'] = df['DATE_OCC'].dt.year
    df['MONTH_OCC'] = df['DATE_OCC'].dt.month
    df['SEASON_OCC'] = ['Kış' if month in [12, 1, 2] else 'İlkbahar' if month in [3, 4, 5] else 'Yaz' if month in [6, 7,8] else 'Sonbahar' for month in df['MONTH_OCC']]
    df['DAY_OF_WEEK_OCC'] = df['DATE_OCC'].dt.dayofweek
    df['DAY_NAME_OCC'] = df['DATE_OCC'].dt.day_name()
    df['TIME_OCC_FORMATTED'] = df['TIME_OCC'].astype(str).str.zfill(4)
    df['TIME_OCC_FORMATTED'] = df['TIME_OCC_FORMATTED'].str[:2] + ':' + df['TIME_OCC_FORMATTED'].str[2:]
    #df['TIME_OCC_HOUR'] = df['TIME_OCC_FORMATTED'].str[:2]
    df['TIME_OCC_HOUR'] = df['TIME_OCC'].astype(str).str.zfill(4).str[:2].astype(int)
    return df.head()

st.write(tarihi_parcala(df))

def gece_gündüz_ekleme(df):
    df['TIME_OCC_HOUR_2'] = df['TIME_OCC_HOUR'].astype(int)
    df['DAY_NIGHT_OCC'] = ['Gündüz' if 6 <= hour < 18 else 'Gece' for hour in df['TIME_OCC_HOUR_2']]
    return df.head()

st.write("Tarih değişkenlerinin düzenlenmesi ve üretilmesi:")
#st.write(tarihi_parcala(df))
st.write(gece_gündüz_ekleme(df))

st.write("Suç türlerine göre adetleri:")
crime_counts = df['CRM_CD_DESC'].value_counts().reset_index()
crime_counts.index = crime_counts.index + 1
crime_counts.columns = ['SUÇ_TÜRÜ', 'OLAY_SAYISI']
st.write(crime_counts)

top_10_crimes_index = df['CRM_CD_DESC'].value_counts().head(10)
st.title("Suç Analizi")
st.write("En sık görülen ilk 10 suç:")
fig, ax = plt.subplots(figsize=(10, 6))
top_10_crimes_index.plot(kind='barh', ax=ax, color='gray')
ax.set_title("En Sık Görülen İlk 10 Suç", fontsize=16)
ax.set_xlabel("Olay Sayısı", fontsize=12)
ax.set_ylabel("Suç Türü", fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.gca().invert_yaxis()
st.pyplot(fig)


top_10_crimes_index = df['CRM_CD_DESC'].value_counts().head(10).index

# Her suç için ayrı grafik oluştur

st.title("Özet Grafikler")
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(30, 20))
axes = axes.flatten()


for i, crime in enumerate(top_10_crimes_index):
    # Suç filtresi ve yıllık olay sayısı
    yearly_data = df[df['CRM_CD_DESC'] == crime].groupby('YEAR_OCC').size()
    # Grafik oluşturma
    yearly_data.plot(kind='bar', color='gray', ax=axes[i], width=0.5)
    axes[i].set_title(f"{crime} Suçunun Yıllara Göre Dağılımı", fontsize=12)
    axes[i].set_xlabel("Yıl", fontsize=10)
    axes[i].set_ylabel("Olay Sayısı", fontsize=10)
    axes[i].tick_params(axis='x', rotation=45)
# Boş grafik hücrelerini temizleme (eğer 10'dan az suç varsa)
for j in range(len(top_10_crimes_index), len(axes)):
    fig.delaxes(axes[j])

    fig.subplots_adjust(hspace=0.6, wspace=0.6)  # Dikey (hspace) ve yatay (wspace) boşlukları artır
# Streamlit'te başlık ve grafik gösterimi
st.subheader("İlk 10 Suçun Yıllara Göre Dağılımı")
st.pyplot(fig)





# Suçların mevsimlere göre dağılım grafiği
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(30, 20))
axes = axes.flatten()

for i, crime in enumerate(top_10_crimes_index):
    # Suç filtresi ve mevsimlik olay sayısı
    seasonal_data = df[df['CRM_CD_DESC'] == crime].groupby('SEASON_OCC').size().reindex(
        ['Kış', 'İlkbahar', 'Yaz', 'Sonbahar'], fill_value=0)

    # Grafik oluşturma
    seasonal_data.plot(kind='bar', color='gray', ax=axes[i], width=0.5)
    axes[i].set_title(f"{crime} Suçunun Mevsimlere Göre Dağılımı", fontsize=12)
    axes[i].set_xlabel("Mevsim", fontsize=10)
    axes[i].set_ylabel("Olay Sayısı", fontsize=10)
    axes[i].tick_params(axis='x', rotation=45)

# Boş grafik hücrelerini temizleme (eğer 10'dan az suç varsa)
for j in range(len(top_10_crimes_index), len(axes)):
    fig.delaxes(axes[j])

# Alt grafikler arasındaki boşlukları ayarlama
fig.subplots_adjust(hspace=0.6, wspace=0.6)

# Streamlit'te başlık ve grafik gösterimi
st.subheader("İlk 10 Suçun Mevsimlere Göre Dağılımı")
st.pyplot(fig)





fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(30, 20))
axes = axes.flatten()

# Her bir suç için grafikleri oluşturma
for i, crime in enumerate(top_10_crimes_index):
    # Suç filtresi ve aylık olay sayısı
    monthly_data = df[df['CRM_CD_DESC'] == crime].groupby('MONTH_OCC').size()

    # Grafik oluşturma
    monthly_data.plot(kind='bar', color='gray', ax=axes[i], width=0.5)
    axes[i].set_title(f"{crime} Suçunun Aylara Göre Dağılımı", fontsize=12)
    axes[i].set_xlabel("Ay", fontsize=10)
    axes[i].set_ylabel("Olay Sayısı", fontsize=10)
    axes[i].tick_params(axis='x', rotation=45)

# Boş grafik hücrelerini temizleme (eğer 10'dan az suç varsa)
for j in range(len(top_10_crimes_index), len(axes)):
    fig.delaxes(axes[j])

# Alt grafikler arasındaki boşlukları ayarlama
fig.subplots_adjust(hspace=0.6, wspace=0.6)

# Streamlit'te başlık ve grafik gösterimi
st.subheader("İlk 10 Suçun Aylara Göre Dağılımı")
st.pyplot(fig)



fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(30, 20))
axes = axes.flatten()

for i, crime in enumerate(top_10_crimes_index):
    # Suç filtresi ve günlük olay sayısı
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily_data = df[df['CRM_CD_DESC'] == crime].groupby('DAY_NAME_OCC').size().reindex(day_order)

    # Grafik oluşturma
    daily_data.plot(kind='bar', color='gray', ax=axes[i], width=0.5)
    axes[i].set_title(f"{crime} Suçunun Günlere Göre Dağılımı", fontsize=12)
    axes[i].set_xlabel("Gün", fontsize=10)
    axes[i].set_ylabel("Olay Sayısı", fontsize=10)
    axes[i].tick_params(axis='x', rotation=45)

# Boş grafik hücrelerini temizleme (eğer 10'dan az suç varsa)
for j in range(len(top_10_crimes_index), len(axes)):
    fig.delaxes(axes[j])

# Alt grafikler arasındaki boşlukları ayarlama
fig.subplots_adjust(hspace=0.6, wspace=0.6)

# Streamlit'te başlık ve grafik gösterimi
st.subheader("İlk 10 Suçun Günlere Göre Dağılımı")
st.pyplot(fig)


fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(30, 20))
axes = axes.flatten()

for i, crime in enumerate(top_10_crimes_index):
    # Suç filtresi ve gece-gündüz olay sayısı
    day_night_data = df[df['CRM_CD_DESC'] == crime].groupby('DAY_NIGHT_OCC').size().reindex(
        ['Gündüz', 'Gece'], fill_value=0)

    # Grafik oluşturma
    day_night_data.plot(kind='bar', color='gray', ax=axes[i], width=0.5)
    axes[i].set_title(f"{crime} Suçunun Gece-Gündüz Dağılımı", fontsize=12)
    axes[i].set_xlabel("Gece-Gündüz", fontsize=10)
    axes[i].set_ylabel("Olay Sayısı", fontsize=10)
    axes[i].tick_params(axis='x', rotation=45)

# Boş grafik hücrelerini temizleme (eğer 10'dan az suç varsa)
for j in range(len(top_10_crimes_index), len(axes)):
    fig.delaxes(axes[j])

# Alt grafikler arasındaki boşlukları ayarlama
fig.subplots_adjust(hspace=0.6, wspace=0.6)

# Streamlit'te başlık ve grafik gösterimi
st.subheader("İlk 10 Suçun Gece-Gündüz Dağılımı")
st.pyplot(fig)



# Grafik düzeni: 3 satır ve 4 sütun
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(30, 20))
axes = axes.flatten()

# Her suç için grafik oluşturma
for i, crime in enumerate(top_10_crimes_index):
    # Suç filtresi ve saatlik olay sayısı
    hourly_data = df[df['CRM_CD_DESC'] == crime].groupby('TIME_OCC_HOUR').size().reindex(
        range(24), fill_value=0)  # 0-23 aralığında saatleri zorunlu hale getir

    # Grafik oluşturma
    hourly_data.plot(kind='bar', color='gray', ax=axes[i], width=0.6)
    axes[i].set_title(f"{crime} Suçunun Saatlere Göre Dağılımı", fontsize=14)
    axes[i].set_xlabel("Saat", fontsize=12)
    axes[i].set_ylabel("Olay Sayısı", fontsize=12)
    axes[i].tick_params(axis='x', rotation=45)

# Boş grafik hücrelerini temizleme (eğer 10'dan az suç varsa)
for j in range(len(top_10_crimes_index), len(axes)):
    fig.delaxes(axes[j])

# Alt grafikler arasındaki boşlukları ayarlama
fig.subplots_adjust(hspace=0.6, wspace=0.6)

# Streamlit'te başlık ve grafik gösterimi
st.subheader("İlk 10 Suçun Saatlere Göre Dağılımı")
st.pyplot(fig)





st.subheader("Aykırı Değer Kontrolü")
num_col = df.select_dtypes(include=['float64', 'int64']).columns
st.write("Sayısal Kolonlar:", num_col)


# Boxplot grafiği oluşturma
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x=df["VICT_AGE"], ax=ax)
ax.set_title("Age için Boxplot Grafiği", fontsize=14)
# Boxplot grafiğini Streamlit'te gösterme
st.pyplot(fig)

#############################################################################################################

import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# 1. Veri Hazırlığı (Örnek için basit veri seti)
from sklearn.datasets import make_classification

# Veri seti oluşturma
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Modellerin Eğitimi ve Değerlendirilmesi
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrikler
    metrics = {
        "Doğruluk (Eğitim)": accuracy_score(y_train, y_train_pred),
        "Doğruluk (Test)": accuracy_score(y_test, y_test_pred),
        "F1 Skoru (Eğitim)": f1_score(y_train, y_train_pred),
        "F1 Skoru (Test)": f1_score(y_test, y_test_pred),
        "ROC AUC (Eğitim)": roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]),
        "ROC AUC (Test)": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    }
    return metrics

# Modelleri tanımlama
models = {
    "LightGBM": LGBMClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
}

# 3. Streamlit Arayüzü
st.title("Model Başarı Değerlendirme Tablosu")

results = {}
for model_name, model in models.items():
    st.subheader(f"Model: {model_name}")
    metrics = train_and_evaluate_model(model, X_train, X_test, y_train, y_test)
    results[model_name] = metrics

    # Sonuçları tablo formatında gösterme
    st.table(metrics)

# Tüm sonuçları karşılaştırmalı tablo olarak gösterme
st.title("Karşılaştırmalı Model Değerlendirmesi")
comparison_table = {
    "Model": [],
    "Doğruluk (Eğitim)": [],
    "Doğruluk (Test)": [],
    "F1 Skoru (Eğitim)": [],
    "F1 Skoru (Test)": [],
    "ROC AUC (Eğitim)": [],
    "ROC AUC (Test)": []
}

for model_name, metrics in results.items():
    comparison_table["Model"].append(model_name)
    for key, value in metrics.items():
        comparison_table[key].append(value)

st.dataframe(pd.DataFrame(comparison_table))


########################################################################################


import streamlit as st
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Veri yükleme ve hazırlık (örnek veri setiyle çalıştığınızı varsayıyoruz)
@st.cache_data
def load_and_prepare_data():

    # Kategorik verileri dönüştürme
    le_day = LabelEncoder()
    le_crime = LabelEncoder()
    df['DAY_NAME_OCC'] = le_day.fit_transform(df['DAY_NAME_OCC'])
    df['CRM_CD_DESC'] = le_crime.fit_transform(df['CRM_CD_DESC'])

    X = df[['DAY_NAME_OCC', 'TIME_OCC_HOUR', 'CRM_CD_DESC']]
    y = df[['LAT', 'LON']]

    return df, X, y, le_day, le_crime

df, X, y, le_day, le_crime = load_and_prepare_data()

# Model eğitimi
def train_model(X, y):
    model_lat = LGBMRegressor(random_state=42)
    model_lon = LGBMRegressor(random_state=42)

    X_train, X_test, y_train_lat, y_test_lat = train_test_split(X, y['LAT'], test_size=0.2, random_state=42)
    _, _, y_train_lon, y_test_lon = train_test_split(X, y['LON'], test_size=0.2, random_state=42)

    model_lat.fit(X_train, y_train_lat)
    model_lon.fit(X_train, y_train_lon)

    return model_lat, model_lon

model_lat, model_lon = train_model(X, y)

# Streamlit arayüzü
st.title("Suç Lokasyonu Tahmin Uygulaması")
st.write("Gün, saat ve suç türünü seçerek tahmini çalıştırabilirsiniz.")

# Kullanıcıdan giriş alma
day = st.selectbox("Gün Seçin", options=list(le_day.classes_))
time_hour = st.slider("Saat Seçin (24 saat formatında)", min_value=0, max_value=23, value=12)
crime = st.selectbox("Suç Türü Seçin", options=list(le_crime.classes_))

# Tahmin yap butonu
if st.button("Tahmini Çalıştır"):
    # Kullanıcı girdilerini dönüştürme
    day_encoded = le_day.transform([day])[0]
    crime_encoded = le_crime.transform([crime])[0]
    example = [[day_encoded, time_hour, crime_encoded]]

    # Tahminler
    predicted_lat = model_lat.predict(example)
    predicted_lon = model_lon.predict(example)

    # Sonuçları gösterme
    st.success(f"Tahmin edilen lokasyon: LAT = {predicted_lat[0]:.6f}, LON = {predicted_lon[0]:.6f}")

    # Harita üzerinde gösterim
    st.map(pd.DataFrame({'lat': [predicted_lat[0]], 'lon': [predicted_lon[0]]}))