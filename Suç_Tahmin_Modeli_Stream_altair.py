import pandas as pd
import streamlit as st
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Streamlit ayarları
st.set_page_config(layout="wide")

# Başlık
st.title("Suç Tahmin Modeli")

# Veri Yükleme
data_path = "C:/Users/almak/Desktop/yetkin/DSMLBC/16.DÖNEM/Suç Tahmin Modeli/Crime_Data_from_2020_to_Present_Proje.csv"
st.header("Veri Setinin Yüklenmesi")
df = pd.read_csv(data_path)
st.success("Veri başarıyla yüklendi!")

# Veri Seti Görüntüleme
st.header("Veri Seti")
st.dataframe(df.head())

# Veri Hakkında Özet Bilgi
st.header("Veri Seti Özeti")
st.write("Veri seti boyutu:", df.shape)
st.write("İstatistiksel özet:", df.describe())
st.write("Eksik değer analizi:")
eksik_degerler = df.isnull().sum().sort_values(ascending=False).to_frame(name='Eksik Değer Sayısı').assign(
    Oran=lambda x: round((x['Eksik Değer Sayısı'] / len(df)) * 100, 2)
)
st.dataframe(eksik_degerler)

# Eksik Değer Grafiği
eksik_degerler.reset_index(inplace=True)
eksik_degerler.columns = ['Kolon', 'Eksik Değer Sayısı', 'Oran']
chart = alt.Chart(eksik_degerler).mark_bar(color='gray').encode(
    x=alt.X('Kolon', sort=None),
    y='Eksik Değer Sayısı',
    tooltip=['Kolon', 'Eksik Değer Sayısı', 'Oran']
).properties(
    title='Eksik Değer Sayısı ve Oranları',
    width=800,
    height=400
)
st.subheader("Eksik Değer Grafiği")
st.altair_chart(chart)

# Kolon Başlıklarını Düzenleme
st.write("Kolon başlıklarının düzenlenmesi:")
df.columns = df.columns.str.replace(" ", "_").str.upper()
st.dataframe(df.head())

# Tarih Değişkenlerini Düzenleme
df['DATE_OCC'] = pd.to_datetime(df['DATE_OCC'], errors='coerce')
df['YEAR_OCC'] = df['DATE_OCC'].dt.year
df['MONTH_OCC'] = df['DATE_OCC'].dt.month
df['SEASON_OCC'] = df['MONTH_OCC'].map({
    12: 'Kış', 1: 'Kış', 2: 'Kış',
    3: 'İlkbahar', 4: 'İlkbahar', 5: 'İlkbahar',
    6: 'Yaz', 7: 'Yaz', 8: 'Yaz',
    9: 'Sonbahar', 10: 'Sonbahar', 11: 'Sonbahar'
})
df['DAY_NAME_OCC'] = df['DATE_OCC'].dt.day_name()
df['TIME_OCC_HOUR'] = df['TIME_OCC'].astype(str).str.zfill(4).str[:2].astype(int)
df['DAY_NIGHT_OCC'] = df['TIME_OCC_HOUR'].apply(lambda x: 'Gündüz' if 6 <= x < 18 else 'Gece')

st.write("Tarih değişkenlerinin düzenlenmiş hali:")
st.dataframe(df.head())

# En Sık Görülen İlk 10 Suç
top_10_crimes = df['CRM_CD_DESC'].value_counts().head(10).reset_index()
top_10_crimes.columns = ['Suç Türü', 'Olay Sayısı']
chart = alt.Chart(top_10_crimes).mark_bar(color='gray').encode(
    y=alt.Y('Suç Türü', sort='-x'),
    x='Olay Sayısı',
    tooltip=['Suç Türü', 'Olay Sayısı']
).properties(
    title='En Sık Görülen İlk 10 Suç',
    width=800,
    height=400
)
st.subheader("En Sık Görülen İlk 10 Suç")
st.altair_chart(chart)

# İlk 10 Suçun Yıllara Göre Dağılımı
top_10_crimes_index = top_10_crimes['Suç Türü'].tolist()
yearly_data = df[df['CRM_CD_DESC'].isin(top_10_crimes_index)].groupby(['CRM_CD_DESC', 'YEAR_OCC']).size().reset_index(name='Olay Sayısı')
chart = alt.Chart(yearly_data).mark_bar().encode(
    x='YEAR_OCC:O',
    y='Olay Sayısı:Q',
    color='CRM_CD_DESC:N',
    column=alt.Column('CRM_CD_DESC:N', header=alt.Header(title='Suç Türü')),
    tooltip=['YEAR_OCC', 'Olay Sayısı']
).properties(
    width=120,
    height=300
)
st.subheader("İlk 10 Suçun Yıllara Göre Dağılımı")
st.altair_chart(chart)

# İlk 10 Suçun Mevsimlere Göre Dağılımı
seasonal_data = df[df['CRM_CD_DESC'].isin(top_10_crimes_index)].groupby(['CRM_CD_DESC', 'SEASON_OCC']).size().reset_index(name='Olay Sayısı')
chart = alt.Chart(seasonal_data).mark_bar().encode(
    x='SEASON_OCC:O',
    y='Olay Sayısı:Q',
    color='CRM_CD_DESC:N',
    column=alt.Column('CRM_CD_DESC:N', header=alt.Header(title='Suç Türü')),
    tooltip=['SEASON_OCC', 'Olay Sayısı']
).properties(
    width=120,
    height=300
)
st.subheader("İlk 10 Suçun Mevsimlere Göre Dağılımı")
st.altair_chart(chart)

# Model Eğitimi ve Değerlendirme
st.title("Model Eğitimi ve Değerlendirme")
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "LightGBM": LGBMClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
}

results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[model_name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    }

st.write("Model Performansı:")
st.dataframe(pd.DataFrame(results).T)

# Suç Lokasyonu Tahmin Uygulaması
st.title("Suç Lokasyonu Tahmin Uygulaması")
le_day = LabelEncoder()
le_crime = LabelEncoder()
df['DAY_NAME_OCC'] = le_day.fit_transform(df['DAY_NAME_OCC'])
df['CRM_CD_DESC'] = le_crime.fit_transform(df['CRM_CD_DESC'])

X = df[['DAY_NAME_OCC', 'TIME_OCC_HOUR', 'CRM_CD_DESC']]
y = df[['LAT', 'LON']]

model_lat = LGBMRegressor(random_state=42)
model_lon = LGBMRegressor(random_state=42)
model_lat.fit(X, y['LAT'])
model_lon.fit(X, y['LON'])

day = st.selectbox("Gün Seçin", options=le_day.classes_)
time_hour = st.slider("Saat Seçin", 0, 23, 12)
crime = st.selectbox("Suç Türü Seçin", options=le_crime.classes_)

if st.button("Tahmini Çalıştır"):
    input_data = [[le_day.transform([day])[0], time_hour, le_crime.transform([crime])[0]]]
    predicted_lat = model_lat.predict(input_data)[0]
    predicted_lon = model_lon.predict(input_data)[0]
    st.success(f"Tahmin edilen lokasyon: LAT={predicted_lat:.6f}, LON={predicted_lon:.6f}")
    st.map(pd.DataFrame({'lat': [predicted_lat], 'lon': [predicted_lon]}))
