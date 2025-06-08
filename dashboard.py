import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

st.set_page_config(page_title="Disney Princess Dashboard", layout="wide")
st.title("👑 Disney Princess Popularity Dashboard")

# Sidebar navigation
page = st.sidebar.selectbox("Pilih Halaman", ["Dashboard Umum", "Unsupervised Learning (K-Means)", "Supervised Learning (Logistic Regression)"])

@st.cache_data
def load_data():
    return pd.read_csv("disney_princess_popularity_dataset_300_rows.csv")

df = load_data()

# ---------------------------- DASHBOARD UMUM ----------------------------
if menu == "Dashboard Umum":
st.title("👑 Disney Princess Popularity Dashboard")
st.header("📊 Ringkasan Dataset")
st.dataframe(df.head())

bash
Copy
Edit
st.subheader("📌 Statistik Umum")
st.dataframe(df.describe())

st.subheader("📊 Distribusi Status IsIconic")
fig0, ax0 = plt.subplots()
df['IsIconic_Biner'] = df['IsIconic'].map({'Yes': 1, 'No': 0})
df['IsIconic_Biner'].dropna().map({1: 'Ikonik', 0: 'Tidak Ikonik'}).value_counts().sort_index().plot(
    kind='bar', color=['gray', 'gold'], ax=ax0)
ax0.set_title('Distribusi Putri Disney Ikonik vs Tidak')
ax0.set_ylabel('Jumlah')
st.pyplot(fig0)


# ---------------------------- UNSUPERVISED LEARNING (K-MEANS) ----------------------------
elif menu == "Unsupervised Learning":
st.title("🔍 Unsupervised Learning - K-Means Clustering")

python
Copy
Edit
selected_cols = [
    'PopularityScore',
    'GoogleSearchIndex2024',
    'RottenTomatoesScore',
    'BoxOfficeMillions'
]

data_cluster = df[selected_cols].dropna()
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_cluster)

df_scaled = pd.DataFrame(data_scaled, columns=selected_cols)

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
labels = kmeans.fit_predict(data_scaled)

df_result = df.copy()
df_result['Cluster'] = labels

cluster_names = {
    0: "Viral Sensation",
    1: "Classic Icons",
    2: "Underrated Gems"
}

df_result['Cluster Label'] = df_result['Cluster'].map(cluster_names)

st.subheader("📁 Hasil Clustering")
st.dataframe(df_result[['PrincessName'] + selected_cols + ['Cluster', 'Cluster Label']])

st.subheader("📌 Statistik Tiap Cluster")
st.write(df_result.groupby('Cluster Label')[selected_cols].mean())

st.subheader("📈 Elbow Method")
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, random_state=42, n_init='auto')
    km.fit(data_scaled)
    wcss.append(km.inertia_)

fig_elbow, ax_elbow = plt.subplots()
ax_elbow.plot(range(1, 11), wcss, marker='o')
ax_elbow.set_title('Metode Elbow')
ax_elbow.set_xlabel('Jumlah Cluster')
ax_elbow.set_ylabel('WCSS')
st.pyplot(fig_elbow)

st.subheader("🧭 Visualisasi Clustering")
fig_vis, ax_vis = plt.subplots(figsize=(10, 6))
scatter = ax_vis.scatter(
    data_scaled[:, 0],  # PopularityScore scaled
    data_scaled[:, 1],  # GoogleSearchIndex2024 scaled
    c=labels,
    cmap='viridis',
    marker='o'
)
ax_vis.set_title('K-Means Clustering of Disney Princesses')
ax_vis.set_xlabel('Popularity Score (scaled)')
ax_vis.set_ylabel('Google Search Index 2024 (scaled)')
colorbar = fig_vis.colorbar(scatter, ax=ax_vis)
colorbar.set_label('Cluster')
st.pyplot(fig_vis)

st.subheader("📝 Detail Tiap Cluster")
for cid in sorted(cluster_names):
    st.markdown(f"**Cluster {cid} - {cluster_names[cid]}**")
    st.dataframe(df_result[df_result['Cluster'] == cid][['PrincessName'] + selected_cols])


# ---------------------------- SUPERVISED LEARNING (LOGISTIC REGRESSION) ----------------------------
elif menu == "Supervised Learning":
st.title("📉 Supervised Learning - Logistic Regression")

python
Copy
Edit
st.subheader("🔧 Persiapan Data")
df['IsIconic_Biner'] = df['IsIconic'].map({'Yes': 1, 'No': 0})
features = [
    'PopularityScore', 'GoogleSearchIndex2024', 'RottenTomatoesScore',
    'BoxOfficeMillions', 'IMDB_Rating', 'AvgScreenTimeMinutes',
    'NumMerchItemsOnAmazon', 'InstagramFanPages', 'TikTokHashtagViewsMillions'
]
X = df[features]
y = df['IsIconic_Biner']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

st.subheader("📋 Laporan Klasifikasi")
st.text(classification_report(y_test, y_pred))

st.markdown("""
**Penjelasan Metrik:**
- Precision: Dari semua prediksi "ikonik", berapa yang benar.
- Recall: Dari semua yang seharusnya ikonik, berapa yang ditemukan.
- F1-score: Gabungan Precision dan Recall
- Support: Jumlah data per kelas

Interpretasi penting dalam konteks Disney Princess.
""")

st.subheader("🧮 Confussion Matrix")
conf = confusion_matrix(y_test, y_pred)
fig_conf, ax_conf = plt.subplots()
sns.heatmap(conf, annot=True, fmt='d', cmap='Blues', xticklabels=['Tidak Ikonik', 'Ikonik'], yticklabels=['Tidak Ikonik', 'Ikonik'], ax=ax_conf)
ax_conf.set_title('Matriks Kebingungan')
ax_conf.set_xlabel('Prediksi')
ax_conf.set_ylabel('Aktual')
st.pyplot(fig_conf)

st.markdown("""
Matriks kebingungan digunakan untuk mengevaluasi performa klasifikasi:
- True Positive (TP): Prediksi Ikonik & Benar
- False Positive (FP): Prediksi Ikonik, tapi salah
- False Negative (FN): Tidak diprediksi ikonik, padahal iya
- True Negative (TN): Prediksi Tidak Ikonik & Benar

Metrik seperti Akurasi, Precision, Recall berasal dari sini.
""")

st.subheader("📊 Kurva ROC dan AUC")
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)
fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, label=f"AUC = {auc:.2f}", color="darkorange")
ax_roc.plot([0,1],[0,1],'--', color='gray')
ax_roc.set_title("ROC Curve")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.legend()
st.pyplot(fig_roc)

st.subheader("📌 Pentingnya Fitur")
coef = model.coef_[0]
feature_imp = pd.DataFrame({'Fitur': features, 'Koefisien': coef})
feature_imp['Rasio Odds'] = np.exp(feature_imp['Koefisien'])
feature_imp_sorted = feature_imp.sort_values('Koefisien', ascending=False)

fig_feat, ax_feat = plt.subplots()
sns.barplot(data=feature_imp_sorted, y='Fitur', x='Koefisien', palette='viridis', ax=ax_feat)
ax_feat.set_title("Koefisien Fitur dalam Model")
st.pyplot(fig_feat)

st.subheader("📈 Rasio Odds (e^Koefisien)")
st.dataframe(feature_imp_sorted)

st.subheader("📍 Prediksi vs Aktual IsIconic")
hasil_pred = pd.DataFrame({
    'Princess': df.loc[y_test.index, 'PrincessName'],
    'Aktual IsIconic': y_test.map({1: 'Ikonik', 0: 'Tidak'}),
    'Prediksi IsIconic': pd.Series(y_pred, index=y_test.index).map({1: 'Ikonik', 0: 'Tidak'})
})
st.dataframe(hasil_pred.reset_index(drop=True))
