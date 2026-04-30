import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans

image = mpimg.imread('example_grayscale.png')


h, w = image.shape


X = image.reshape(-1, 1)


K = 100

kmeans = KMeans(n_clusters=K, n_init=10)
labels = kmeans.fit_predict(X)
centri = kmeans.cluster_centers_


X_kvant = centri[labels]
image_kvant = X_kvant.reshape(h, w)


plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title(f"Kvantizirana (K={K})")
plt.imshow(image_kvant, cmap='gray')
plt.axis('off')

plt.show()