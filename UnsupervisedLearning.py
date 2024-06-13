import ax as ax
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, silhouette_score, \
    homogeneity_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns

df = pd.read_csv('penguins.csv')
df = pd.DataFrame(df)

# features = df['bill_length_mm', 'flipper_length_mm']

species_bill_length_avg = df.groupby('species')['bill_length_mm'].transform('mean')
species_flipper_length_avg = df.groupby('species')['flipper_length_mm'].transform('mean')

species_bill_length_avg = df.groupby('species')['bill_length_mm'].transform('mean')
species_flipper_length_avg = df.groupby('species')['flipper_length_mm'].transform('mean')
# print(df['species'].unique())
# print(df['island'].unique())
# print(species_bill_length_avg.values)
# 用相应species的平均值填充bill_length_mm的缺失值
df['bill_length_mm'] = df['bill_length_mm'].fillna(species_bill_length_avg)

# 用相应species的平均值填充flipper_length_mm的缺失值
df['flipper_length_mm'] = df['flipper_length_mm'].fillna(species_flipper_length_avg)

X = df.drop(['species', 'island', 'sex', 'bill_depth_mm', 'body_mass_g', 'year', 'rowid'], axis=1)
y = df['species']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow Method
inertia = []

K = 10
for k in range(1, K + 1):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot the elbow
plt.figure()
plt.plot(range(1, K + 1), inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('The elbow method showing the optimal k')
plt.show()
plt.savefig('cluster_k_fold parameter.png', format='png', dpi=300)


# Choosing K=3 based on the Elbow Method
k_mean = KMeans(n_clusters=3, random_state=14)
k_mean.fit(X)
y_kmeans = k_mean.predict(X)

species_labels = ['Adelie', 'Chinstrap', 'Gentoo']

np.random.seed(42)  # 为了结果可复现，设置固定的随机种子
num_clusters = 3  # KMeans使用的簇的数量

# 为每个数据点生成一个随机簇标签
random_labels = np.random.randint(0, num_clusters, len(X))

# 计算并打印随机聚类的轮廓系数，与KMeans聚类的轮廓系数进行比较
silhouette_random = silhouette_score(X, random_labels)
silhouette_kmeans = silhouette_score(X, k_mean.labels_)
homogeneity_random = homogeneity_score(y, random_labels)
homogeneity_kmeans = homogeneity_score(y, y_kmeans)

print('Silhouette Score for KMeans:', silhouette_kmeans)
print('Silhouette Score for Random Clustering:', silhouette_random)
print('Homogeneity Score for KMeans:', homogeneity_kmeans)
print('Homogeneity Score for Random Clustering:', homogeneity_random)

centroids = k_mean.cluster_centers_
plt.figure()

# 绘制随机聚类的散点图
colors = plt.scatter(X['flipper_length_mm'], X['bill_length_mm'], c=random_labels, cmap='viridis', alpha=0.5,
                     edgecolor='k')
species_labels = ['Adelie', 'Chinstrap', 'Gentoo']
plt.legend(handles=colors.legend_elements()[0], labels=species_labels, title="Species")
plt.title('Random Clustering Baseline')
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Bill Length (mm)')
plt.show()

plt.scatter(x=df['bill_length_mm'], y=df['flipper_length_mm'], c=k_mean.labels_, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
# Visualizing the clusters
plt.title('K-Means Clustering of Penguins')
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Bill Length (mm)')
plt.show()
