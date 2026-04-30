import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image


image = Image.open("example.png")
image_np = np.array(image)


h, w, c = image_np.shape
pixels = image_np.reshape(-1, 3)


k = 8
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(pixels)

# Zamijeni boje centroidima
new_colors = kmeans.cluster_centers_[kmeans.labels_]
quantized_image = new_colors.reshape(h, w, 3).astype(np.uint8)


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_np)
plt.title("Originalna slika")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(quantized_image)
plt.title(f"Kvantizirana slika (k={k})")
plt.axis("off")

plt.show()