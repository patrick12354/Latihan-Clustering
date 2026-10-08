## Semua kode dalam 1 cell (copy)

```python
# Contoh Clustering Image
# load gambar -> scale + PCA -> KMeans -> evaluasi -> plot -> insight

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Load data

# # dataset contoh (digits). Kalau soal kasih folder gambar, pakai cell di bawahnya
# from sklearn.datasets import load_digits
# d = load_digits()
# X = d.data
# images = d.images
# y_true = d.target      # label asli, cuma buat ngecek di akhir
# print(X.shape)

# kalau gambar di folder: data/nama_kelas/gambar.jpg
# from pathlib import Path
# from PIL import Image
# paths = sorted(Path("data").rglob("*.jpg"))
# images = np.array([np.array(Image.open(p).convert("RGB").resize((64, 64))) for p in paths])
# X = images.reshape(len(images), -1) / 255.0
# y_true = np.array([p.parent.name for p in paths])

# kalau data .npz (MNIST 100rb): ambil sampel dulu biar ga berat
data = np.load("Image/MNIST_100000_28x28.npz")
images = data["x_train"][:5000]
X = images.reshape(len(images), -1) / 255.0
y_true = data["y_train"][:5000]

fig, ax = plt.subplots(2, 8, figsize=(10, 3))
for i, a in enumerate(ax.flat):
    a.imshow(images[i], cmap="gray")
    a.axis("off")
plt.show()

# Preprocessing

# from sklearn.preprocessing import RobustScaler

# X_scaled = RobustScaler().fit_transform(X)  # kalau ada outlier

X_scaled = StandardScaler().fit_transform(X)


pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(pca.explained_variance_ratio_.sum())

# Tentukan k

from tqdm.auto import tqdm
ks = range(2, 15)
inertia, sil = [], []
for k in tqdm(ks):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_pca)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X_pca, km.labels_))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(ks, inertia, "o-"); ax[0].set_title("Elbow")
ax[1].plot(ks, sil, "o-"); ax[1].set_title("Silhouette")
plt.show()

# k dipilih = ... karena ... (isi sendiri dari grafik)

# KMeans + evaluasi

k = 10   # GANTI
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_pca)

print("Silhouette    :", silhouette_score(X_pca, labels))
print("Davies-Bouldin:", davies_bouldin_score(X_pca, labels))
print(pd.Series(labels).value_counts().sort_index())

# Plot cluster

p2 = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

plt.figure(figsize=(7, 6))
sc = plt.scatter(p2[:, 0], p2[:, 1], c=labels, cmap="tab10", s=12)
plt.legend(*sc.legend_elements(), title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.title("KMeans (PCA 2D)")
plt.show()

# Insight tiap cluster

fig, ax = plt.subplots(1, k, figsize=(2 * k, 2.5))
for c in range(k):
    mean_img = images[labels == c].mean(axis=0).astype(images.dtype)
    ax[c].imshow(mean_img, cmap="gray")
    ax[c].set_title(f"C{c} (n={np.sum(labels == c)})", fontsize=8)
    ax[c].axis("off")
plt.show()

# bandingin ke label asli kalau ada
pd.crosstab(labels, y_true)

# contoh gambar asli tiap cluster
fig, ax = plt.subplots(k, 6, figsize=(9, 1.6 * k))
for c in range(k):
    idx = np.random.choice(np.where(labels == c)[0], 6, replace=False)
    for a, i in zip(ax[c], idx):
        a.imshow(images[i], cmap="gray"); a.axis("off")
    ax[c][0].set_title(f"C{c}", fontsize=8, loc="left")
plt.show()

# Kesimpulan
# - Evaluasi: silhouette = ..., DBI = ...
# - Cluster 0: ...
# - Cluster 1: ...
# - Cluster yang campur: ...
```

---

## Per bagian

# Contoh Clustering Image
load gambar -> scale + PCA -> KMeans -> evaluasi -> plot -> insight

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
```

## Load data

```python
# # dataset contoh (digits). Kalau soal kasih folder gambar, pakai cell di bawahnya
# from sklearn.datasets import load_digits
# d = load_digits()
# X = d.data
# images = d.images
# y_true = d.target      # label asli, cuma buat ngecek di akhir
# print(X.shape)
```

```python
# kalau gambar di folder: data/nama_kelas/gambar.jpg
# from pathlib import Path
# from PIL import Image
# paths = sorted(Path("data").rglob("*.jpg"))
# images = np.array([np.array(Image.open(p).convert("RGB").resize((64, 64))) for p in paths])
# X = images.reshape(len(images), -1) / 255.0
# y_true = np.array([p.parent.name for p in paths])
```

```python
# kalau data .npz (MNIST 100rb): ambil sampel dulu biar ga berat
data = np.load("Image/MNIST_100000_28x28.npz")
images = data["x_train"][:5000]
X = images.reshape(len(images), -1) / 255.0
y_true = data["y_train"][:5000]
```

```python
fig, ax = plt.subplots(2, 8, figsize=(10, 3))
for i, a in enumerate(ax.flat):
    a.imshow(images[i], cmap="gray")
    a.axis("off")
plt.show()
```

## Preprocessing

```python
# from sklearn.preprocessing import RobustScaler

# X_scaled = RobustScaler().fit_transform(X)  # kalau ada outlier

X_scaled = StandardScaler().fit_transform(X)


pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(pca.explained_variance_ratio_.sum())
```

## Tentukan k

```python
from tqdm.auto import tqdm
ks = range(2, 15)
inertia, sil = [], []
for k in tqdm(ks):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_pca)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X_pca, km.labels_))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(ks, inertia, "o-"); ax[0].set_title("Elbow")
ax[1].plot(ks, sil, "o-"); ax[1].set_title("Silhouette")
plt.show()
```

k dipilih = ... karena ... (isi sendiri dari grafik)

## KMeans + evaluasi

```python
k = 10   # GANTI
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_pca)

print("Silhouette    :", silhouette_score(X_pca, labels))
print("Davies-Bouldin:", davies_bouldin_score(X_pca, labels))
print(pd.Series(labels).value_counts().sort_index())
```

## Plot cluster

```python
p2 = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

plt.figure(figsize=(7, 6))
sc = plt.scatter(p2[:, 0], p2[:, 1], c=labels, cmap="tab10", s=12)
plt.legend(*sc.legend_elements(), title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.title("KMeans (PCA 2D)")
plt.show()
```

## Insight tiap cluster

```python
fig, ax = plt.subplots(1, k, figsize=(2 * k, 2.5))
for c in range(k):
    mean_img = images[labels == c].mean(axis=0).astype(images.dtype)
    ax[c].imshow(mean_img, cmap="gray")
    ax[c].set_title(f"C{c} (n={np.sum(labels == c)})", fontsize=8)
    ax[c].axis("off")
plt.show()

# bandingin ke label asli kalau ada
pd.crosstab(labels, y_true)
```

```python
# contoh gambar asli tiap cluster
fig, ax = plt.subplots(k, 6, figsize=(9, 1.6 * k))
for c in range(k):
    idx = np.random.choice(np.where(labels == c)[0], 6, replace=False)
    for a, i in zip(ax[c], idx):
        a.imshow(images[i], cmap="gray"); a.axis("off")
    ax[c][0].set_title(f"C{c}", fontsize=8, loc="left")
plt.show()
```

## Kesimpulan
- Evaluasi: silhouette = ..., DBI = ...
- Cluster 0: ...
- Cluster 1: ...
- Cluster yang campur: ...
