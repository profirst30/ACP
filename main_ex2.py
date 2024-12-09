# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA

# %%
# Load data
imgtmp = loadmat("Indian_pines_corrected.mat")
img = np.float32(imgtmp['indian_pines_corrected'])
maptmp = loadmat("Indian_pines_gt.mat")
map = maptmp['indian_pines_gt']

# %%
# Display a component of the image
res = img[:, :, 18]
plt.imshow((res - np.min(res)) / (np.max(res) - np.min(res)))
plt.title("Component 18 of the image")
plt.show()

plt.imshow(map)
plt.title("Ground truth map")
plt.show()

# %%
# Apply PCA on the multispectral image
img_reshaped = img.reshape(-1, img.shape[2])
pca = PCA()
img_pca = pca.fit_transform(img_reshaped)

# %%
# Evaluate the number of principal components needed to retain most of the information
explained_variance = np.cumsum(pca.explained_variance_ratio_)
plt.plot(explained_variance)
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Principal Components')
plt.show()

# %%
# Visualize the projection on the principal axis
principal_axis_projection = img_pca[:, 0].reshape(img.shape[0], img.shape[1])
plt.imshow(principal_axis_projection, cmap='gray')
plt.title('Projection on the Principal Axis')
plt.show()

# %%
# Visualize the projection on the first three principal axes as a color image
color_projection = img_pca[:, :3].reshape(img.shape[0], img.shape[1], 3)
plt.imshow((color_projection - np.min(color_projection)) / (np.max(color_projection) - np.min(color_projection)))
plt.title('Projection on the First Three Principal Axes')
plt.show()

# %%
# Compare with ground truth
plt.imshow(map)
plt.title("Ground truth map")
plt.show()

# %%
# Spectral signature of the principal components
for k in range(3):
    plt.plot(pca.components_[k, :])
    plt.title(f'Spectral Signature of Principal Component {k+1}')
    plt.xlabel('Wavelength Index')
    plt.ylabel('Component Value')
    plt.show()