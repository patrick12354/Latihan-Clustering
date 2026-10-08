## Semua kode dalam 1 cell (copy)

```python
# Contoh Clustering Text
# CSV -> bersihin teks -> TF-IDF -> KMeans -> evaluasi -> plot -> insight

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

df = pd.read_csv("dataset.csv")                        # GANTI nama file
df = df.rename(columns={"nama_kolom_teks": "corpus"})  # GANTI nama kolom teks
# df = df.sample(5000, random_state=42)   # kalau datanya gede, ambil sampel acak
df = df.dropna(subset="corpus").drop_duplicates(subset="corpus").reset_index(drop=True)
print(df.shape)
df.head()

# Plot deskriptif

df["jumlah_kata"] = df["corpus"].str.split().str.len()
df["jumlah_kata"].plot(kind="hist", bins=30, title="Panjang teks")
plt.show()

# kalau ada kolom kategori
# df["kategori"].value_counts().plot(kind="bar")
# plt.show()

# Preprocessing

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = set(stopwords.words("indonesian"))   # GANTI "english" kalau bahasa inggris
stop.update(["yg", "ga", "gak", "aja", "banget", "nya"])

def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [w for w in text.split() if w not in stop]
    return " ".join(words)

df["cleaned"] = df["corpus"].apply(clean)
df[["corpus", "cleaned"]].head()

from wordcloud import WordCloud

wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(df["cleaned"]))
plt.imshow(wc)
plt.axis("off")
plt.show()

# TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)
X_tfidf = vectorizer.fit_transform(df["cleaned"])   # dipakai lagi buat top keywords
X = X_tfidf

# opsional: SVD (PCA-nya teks), bandingin dulu sebelum dipakai
# kalau dipakai, X jadi array biasa -> ikuti catatan "kalau pakai SVD" di cell bawah
# from sklearn.decomposition import TruncatedSVD
# from sklearn.preprocessing import Normalizer
# X = Normalizer().fit_transform(TruncatedSVD(n_components=10, random_state=42).fit_transform(X_tfidf))

X.shape

# Tentukan k

from tqdm.auto import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

ks = range(2, 11)
inertia, sil = [], []
for k in tqdm(ks):
    km = KMeans(n_clusters=k, random_state=42).fit(X)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X, km.labels_))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(ks, inertia, "o-"); ax[0].set_title("Elbow")
ax[1].plot(ks, sil, "o-"); ax[1].set_title("Silhouette")
plt.show()

# k dipilih = ... karena ... (isi sendiri dari grafik)

# KMeans + evaluasi

k = 4   # GANTI
kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
df["cluster"] = kmeans.labels_

print("Silhouette    :", silhouette_score(X, df["cluster"]))
print("Davies-Bouldin:", davies_bouldin_score(X.toarray(), df["cluster"]))
# kalau pakai SVD: ganti X.toarray() jadi X
# print("Davies-Bouldin:", davies_bouldin_score(X, df["cluster"]))
print(df["cluster"].value_counts())

# Plot cluster (PCA)

from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
pc = pca.fit_transform(X.toarray())
# kalau pakai SVD: pc = pca.fit_transform(X)
df["x0"], df["x1"] = pc[:, 0], pc[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="x0", y="x1", hue="cluster", palette="viridis")
plt.title("TF-IDF + KMeans")
plt.show()

# Insight tiap cluster

terms = vectorizer.get_feature_names_out()
centers = pd.DataFrame(X_tfidf.todense()).groupby(df["cluster"]).mean()

for c, row in centers.iterrows():
    top = [terms[i] for i in np.argsort(row)[-10:][::-1]]
    print(f"Cluster {c}: {', '.join(top)}")
    for t in df[df["cluster"] == c]["corpus"].head(3):
        print("   -", t)

# kalau ada kolom label
# pd.crosstab(df["cluster"], df["kategori"])

# Kesimpulan
# - Evaluasi: silhouette = ..., DBI = ...
# - Cluster 0: ...
# - Cluster 1: ...
# - Cluster 2: ...
```

---

## Per bagian

# Contoh Clustering Text
CSV -> bersihin teks -> TF-IDF -> KMeans -> evaluasi -> plot -> insight

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

df = pd.read_csv("dataset.csv")                        # GANTI nama file
df = df.rename(columns={"nama_kolom_teks": "corpus"})  # GANTI nama kolom teks
# df = df.sample(5000, random_state=42)   # kalau datanya gede, ambil sampel acak
df = df.dropna(subset="corpus").drop_duplicates(subset="corpus").reset_index(drop=True)
print(df.shape)
df.head()
```

## Plot deskriptif

```python
df["jumlah_kata"] = df["corpus"].str.split().str.len()
df["jumlah_kata"].plot(kind="hist", bins=30, title="Panjang teks")
plt.show()

# kalau ada kolom kategori
# df["kategori"].value_counts().plot(kind="bar")
# plt.show()
```

## Preprocessing

```python
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = set(stopwords.words("indonesian"))   # GANTI "english" kalau bahasa inggris
stop.update(["yg", "ga", "gak", "aja", "banget", "nya"])

def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [w for w in text.split() if w not in stop]
    return " ".join(words)

df["cleaned"] = df["corpus"].apply(clean)
df[["corpus", "cleaned"]].head()
```

```python
from wordcloud import WordCloud

wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(df["cleaned"]))
plt.imshow(wc)
plt.axis("off")
plt.show()
```

## TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)
X_tfidf = vectorizer.fit_transform(df["cleaned"])   # dipakai lagi buat top keywords
X = X_tfidf

# opsional: SVD (PCA-nya teks), bandingin dulu sebelum dipakai
# kalau dipakai, X jadi array biasa -> ikuti catatan "kalau pakai SVD" di cell bawah
# from sklearn.decomposition import TruncatedSVD
# from sklearn.preprocessing import Normalizer
# X = Normalizer().fit_transform(TruncatedSVD(n_components=10, random_state=42).fit_transform(X_tfidf))

X.shape
```

## Tentukan k

```python
from tqdm.auto import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

ks = range(2, 11)
inertia, sil = [], []
for k in tqdm(ks):
    km = KMeans(n_clusters=k, random_state=42).fit(X)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X, km.labels_))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(ks, inertia, "o-"); ax[0].set_title("Elbow")
ax[1].plot(ks, sil, "o-"); ax[1].set_title("Silhouette")
plt.show()
```

k dipilih = ... karena ... (isi sendiri dari grafik)

## KMeans + evaluasi

```python
k = 4   # GANTI
kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
df["cluster"] = kmeans.labels_

print("Silhouette    :", silhouette_score(X, df["cluster"]))
print("Davies-Bouldin:", davies_bouldin_score(X.toarray(), df["cluster"]))
# kalau pakai SVD: ganti X.toarray() jadi X
# print("Davies-Bouldin:", davies_bouldin_score(X, df["cluster"]))
print(df["cluster"].value_counts())
```

## Plot cluster (PCA)

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
pc = pca.fit_transform(X.toarray())
# kalau pakai SVD: pc = pca.fit_transform(X)
df["x0"], df["x1"] = pc[:, 0], pc[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="x0", y="x1", hue="cluster", palette="viridis")
plt.title("TF-IDF + KMeans")
plt.show()
```

## Insight tiap cluster

```python
terms = vectorizer.get_feature_names_out()
centers = pd.DataFrame(X_tfidf.todense()).groupby(df["cluster"]).mean()

for c, row in centers.iterrows():
    top = [terms[i] for i in np.argsort(row)[-10:][::-1]]
    print(f"Cluster {c}: {', '.join(top)}")
    for t in df[df["cluster"] == c]["corpus"].head(3):
        print("   -", t)

# kalau ada kolom label
# pd.crosstab(df["cluster"], df["kategori"])
```

## Kesimpulan
- Evaluasi: silhouette = ..., DBI = ...
- Cluster 0: ...
- Cluster 1: ...
- Cluster 2: ...
