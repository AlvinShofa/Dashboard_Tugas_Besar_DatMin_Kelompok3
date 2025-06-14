import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, silhouette_score

st.set_page_config(page_title="Disney Princess Dashboard", layout="wide")
st.title("👑 Disney Princess Popularity Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv("disney_princess_popularity_dataset_300_rows.csv")

df = load_data()

# ---------------------------- DATA UNDERSTANDING ----------------------------
st.header("📘 Data Understanding")

with st.expander("🔍 Informasi Umum Dataset"):
    buffer = df.info(buf=None)
    st.write(df.dtypes)
    st.write(f"Jumlah baris: {df.shape[0]}")
    st.write(f"Jumlah kolom: {df.shape[1]}")
    st.write("Jumlah data null:")
    st.dataframe(df.isnull().sum())    

with st.expander("📈 Statistik Deskriptif"):
    st.dataframe(df.describe())    

with st.expander("📊 Korelasi Antar Variabel Numerik"):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ---------------------------- BOXPLOT OUTLIERS ----------------------------
st.header("📦 Deteksi Outlier - Boxplot")
selected_outlier_features = st.multiselect(
    "Pilih fitur untuk visualisasi boxplot outlier!",
    numeric_cols,
    default=['PopularityScore', 'GoogleSearchIndex2024', 'BoxOfficeMillions']
)

if selected_outlier_features:
    fig, axes = plt.subplots(nrows=1, ncols=len(selected_outlier_features), figsize=(4*len(selected_outlier_features), 4))
    if len(selected_outlier_features) == 1:
        axes = [axes]
    for i, col in enumerate(selected_outlier_features):
        sns.boxplot(y=df[col], ax=axes[i], color='lightblue')
        axes[i].set_title(f'Boxplot: {col}')
    st.pyplot(fig)
else:
    st.info("Pilih minimal satu fitur untuk ditampilkan sebagai boxplot.")

# ---------------------------- ELBOW METHOD ----------------------------
st.header("📐 Elbow Method")
clustering_features = ['PopularityScore', 'GoogleSearchIndex2024', 'RottenTomatoesScore', 'BoxOfficeMillions']

data_cluster = df[clustering_features].dropna()
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_cluster)

inertias = []

K = range(1, 11)  # Mengecek k dari 1 sampai 10
for k_elbow in K:
    kmeans_elbow = KMeans(n_clusters=k_elbow, n_init='auto', random_state=42)
    kmeans_elbow.fit(data_scaled)
    inertias.append(kmeans_elbow.inertia_)

fig_elbow, ax_elbow = plt.subplots(figsize=(4, 3))
ax_elbow.plot(K, inertias, marker='o')
ax_elbow.set_xlabel('Jumlah Cluster (k)')
ax_elbow.set_ylabel('Inertia')
ax_elbow.set_title('Metode Elbow')

st.pyplot(fig_elbow)

# ---------------------------- K-MEANS CLUSTERING ----------------------------
st.header("🔍 Unsupervised Learning - K-Means Clustering")

clustering_features = st.multiselect(
    "Pilih fitur untuk proses clustering (gunakan yang sama seperti Elbow di atas ya):",
    ['PopularityScore', 'GoogleSearchIndex2024', 'RottenTomatoesScore', 'BoxOfficeMillions'],
    default=['PopularityScore', 'GoogleSearchIndex2024', 'RottenTomatoesScore', 'BoxOfficeMillions']
)

if clustering_features:
    data_cluster = df[clustering_features].dropna()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_cluster)

    k = st.slider("Pilih jumlah klaster (k):", min_value=2, max_value=10, value=3)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(data_scaled)
    df_clustered = data_cluster.copy()
    df_clustered["Cluster"] = cluster_labels

    silhouette = silhouette_score(data_scaled, cluster_labels)
    st.success(f"Silhouette Score untuk k={k}: *{silhouette:.4f}*")

    st.subheader("📊 Visualisasi Clustering (PCA 2D + Region Klaster)")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(data_scaled)

    kmeans_2d = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels_2d = kmeans_2d.fit_predict(X_pca)

    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    Z = kmeans_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(4, 3))
    cmap_light = ListedColormap(sns.color_palette("pastel", n_colors=k).as_hex()) 
    cmap_bold = ListedColormap(sns.color_palette("Set1", n_colors=k).as_hex())  

    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.5)
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_2d, cmap=cmap_bold, edgecolor='k')
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("Visualisasi Cluster (PCA) + Region")
    legend_labels = [f"Cluster {i}" for i in range(k)]

    ax.legend(handles=scatter.legend_elements()[0], labels=legend_labels)

    st.pyplot(fig)

    st.subheader("📌 Statistik Tiap Cluster")
    st.dataframe(df_clustered.groupby("Cluster")[clustering_features].mean())

    st.subheader("📁 Data dengan Label Klaster")
    df["Cluster"] = cluster_labels
    st.dataframe(df[['PrincessName'] + clustering_features + ['Cluster']])
else:
    st.warning("Silakan pilih fitur untuk melanjutkan proses clustering.")
    
# ---------------------------- LOGISTIC REGRESSION ----------------------------
st.header("📉 Supervised Learning - Logistic Regression")

# Preprocessing
df['IsIconic'] = df['IsIconic'].map({'Yes': 1, 'No': 0})
regression_features = [
    'PopularityScore', 'GoogleSearchIndex2024', 'RottenTomatoesScore', 'BoxOfficeMillions',
    'IMDB_Rating', 'AvgScreenTimeMinutes', 'NumMerchItemsOnAmazon', 'InstagramFanPages', 'TikTokHashtagViewsMillions'
]
X = df[regression_features]
y = df['IsIconic']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
logreg = LogisticRegression(random_state=42, max_iter=1000)
logreg.fit(X_train_scaled, y_train)
y_pred = logreg.predict(X_test_scaled)
y_proba = logreg.predict_proba(X_test_scaled)[:, 1]

# Evaluation
st.subheader("🎯 Evaluasi Model")
st.write(f"Akurasi: *{logreg.score(X_test_scaled, y_test):.2f}*")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
fig2, ax2 = plt.subplots(figsize=(4, 3))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Bukan Ikonik', 'Ikonik'],
            yticklabels=['Bukan Ikonik', 'Ikonik'],
            ax=ax2)
ax2.set_xlabel('Prediksi')
ax2.set_ylabel('Aktual')
ax2.set_title('Confusion Matrix')
st.pyplot(fig2)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)
fig3, ax3 = plt.subplots(figsize=(4, 3))
ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('ROC Curve')
ax3.legend(loc="lower right")
st.pyplot(fig3)

# Feature Importance
coef_df = pd.DataFrame({ 
    'Fitur': regression_features,
    'Koefisien': logreg.coef_[0],
    'Odds Ratio': np.exp(logreg.coef_[0]) 
}).sort_values('Odds Ratio', ascending=False)

st.subheader("📈 Pentingnya Fitur & Odds Ratio")
st.dataframe(coef_df)

fig4, ax4 = plt.subplots(figsize=(4, 3))
sns.barplot(x='Koefisien', y='Fitur', data=coef_df, palette='viridis', ax=ax4)
ax4.set_title("Kontribusi Fitur dalam Prediksi")
st.pyplot(fig4)
