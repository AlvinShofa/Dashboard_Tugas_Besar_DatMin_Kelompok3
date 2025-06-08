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

@st.cache_data
def load_data():
    return pd.read_csv("disney_princess_popularity_dataset_300_rows.csv")

df = load_data()

# ---------------------------- DASHBOARD UMUM ----------------------------
st.header("📊 Ringkasan Dataset")
st.dataframe(df.head())

# ---------------------------- UNSUPERVISED LEARNING (K-MEANS) ----------------------------
st.header("🔍 Unsupervised Learning - K-Means Clustering")

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
# Plot Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init='auto')
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

fig, ax = plt.subplots()
ax.plot(range(1, 11), wcss, marker='o')
ax.set_title('Metode Elbow untuk Menentukan Jumlah Cluster')
ax.set_xlabel('Jumlah Cluster')
ax.set_ylabel('WCSS')
st.pyplot(fig)

st.subheader("🧭 Visualisasi Clustering")

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.scatter(
    data_scaled[:, 0],  # Popularity Score (scaled)
    data_scaled[:, 1],  # Google Search Index 2024 (scaled)
    c=labels,
    cmap='viridis',
    marker='o'
)
ax2.set_title('K-Means Clustering of Disney Princesses')
ax2.set_xlabel('Popularity Score (scaled)')
ax2.set_ylabel('Google Search Index 2024 (scaled)')
colorbar = fig2.colorbar(ax2.collections[0], ax=ax2)
colorbar.set_label('Cluster')
st.pyplot(fig2)

st.subheader("📝 Detail Tiap Cluster")
for cluster_id in sorted(cluster_names):
    st.markdown(f"**Cluster {cluster_id} - {cluster_names[cluster_id]}**")
    st.dataframe(df_result[df_result['Cluster'] == cluster_id][['PrincessName'] + selected_cols].reset_index(drop=True))

# ---------------------------- SUPERVISED LEARNING (LOGISTIC REGRESSION) ----------------------------
st.header("📉 Supervised Learning - Logistic Regression")

# Preprocessing
st.subheader("🔧 Persiapan Data")
df['IsIconic'] = df['IsIconic'].map({'Yes': 1, 'No': 0})
features_for_regression = [
    'PopularityScore', 'GoogleSearchIndex2024', 'RottenTomatoesScore', 'BoxOfficeMillions',
    'IMDB_Rating', 'AvgScreenTimeMinutes', 'NumMerchItemsOnAmazon', 'InstagramFanPages',
    'TikTokHashtagViewsMillions']
target = 'IsIconic'
X = df[features_for_regression]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modeling
logreg = LogisticRegression(random_state=42, max_iter=1000)
logreg.fit(X_train_scaled, y_train)
y_pred = logreg.predict(X_test_scaled)
y_pred_proba = logreg.predict_proba(X_test_scaled)[:, 1]

# Evaluation
st.subheader("📋 Laporan Klasifikasi")
report = classification_report(y_test, y_pred, output_dict=True)
st.text(classification_report(y_test, y_pred))

st.markdown("""
**Penjelasan:**
- **Precision**: Seberapa akurat model saat memprediksi kelas tersebut. Misalnya, dari semua prediksi "ikonik", berapa yang benar-benar ikon.
- **Recall**: Seberapa baik model menemukan semua contoh dari kelas tersebut. Misalnya, dari semua putri yang memang "ikonik", berapa yang berhasil ditemukan oleh model.
- **F1-score**: Harmonik rata-rata precision dan recall, berguna untuk dataset tidak seimbang antara kelas.
- **Support**: Jumlah data aktual dari masing-masing kelas di test set.

Model ini mencoba membedakan antara putri yang ikonik dan tidak dengan mempertimbangkan faktor-faktor seperti:
- Skor popularitas (Popularity Score, Google Search Index, dll)
- Keberhasilan film (Box Office, IMDB)
- Aktivitas penggemar (jumlah fan page, views TikTok)

Hasil klasifikasi ini membantu dalam memahami fitur mana yang paling mempengaruhi kemungkinan seorang putri dianggap ikonik.
""")

st.subheader("🧮 Matriks Kebingungan")
conf_matrix = confusion_matrix(y_test, y_pred)
fig3, ax3 = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Tidak Ikonik', 'Ikonik'],
            yticklabels=['Tidak Ikonik', 'Ikonik'], ax=ax3)
ax3.set_xlabel('Prediksi')
ax3.set_ylabel('Aktual')
ax3.set_title('Matriks Kebingungan')
st.pyplot(fig3)

st.subheader("📊 Kurva ROC dan Nilai AUC")
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
fig4, ax4 = plt.subplots()
ax4.plot(fpr, tpr, color='darkorange', lw=2, label=f'Kurva ROC (AUC = {roc_auc:.2f})')
ax4.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax4.set_xlim([0.0, 1.0])
ax4.set_ylim([0.0, 1.05])
ax4.set_xlabel('Tingkat Positif Palsu')
ax4.set_ylabel('Tingkat Positif Benar')
ax4.set_title('Kurva ROC')
ax4.legend(loc="lower right")
st.pyplot(fig4)

st.subheader("📌 Pentingnya Fitur")
feature_importance = pd.DataFrame({
    'Fitur': features_for_regression,
    'Koefisien': logreg.coef_[0]
}).sort_values('Koefisien', ascending=False)

fig5, ax5 = plt.subplots()
sns.barplot(data=feature_importance, x='Koefisien', y='Fitur', palette='viridis', ax=ax5)
ax5.set_title('Pentingnya Fitur dalam Memprediksi Status Ikonik')
ax5.set_xlabel('Nilai Koefisien')
ax5.set_ylabel('Fitur')
st.pyplot(fig5)

# Odds Ratio
feature_importance['Rasio_Odds'] = np.exp(feature_importance['Koefisien'])
st.subheader("📈 Rasio Odds (Koefisien Eksponensial)")
st.dataframe(feature_importance.sort_values('Rasio_Odds', ascending=False))
