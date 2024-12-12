# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA

# %%
# Charger les données
imgtmp = loadmat("Indian_pines_corrected.mat")
img = np.float32(imgtmp['indian_pines_corrected'])
maptmp = loadmat("Indian_pines_gt.mat")
map = maptmp['indian_pines_gt']

# %%
# Afficher une bande spectrale de l'image
res = img[:, :, 18]
plt.imshow((res - np.min(res)) / (np.max(res) - np.min(res)))
plt.title("Component 18 of the image")
plt.savefig("./resultats/Affichage_canal.svg")
plt.show()

plt.imshow(map)
plt.title("Ground truth map")
plt.savefig("./resultats/carte_verite_terrain.svg")
plt.show()

# %%
# Application de l'ACP
img_reshaped = img.reshape(-1, img.shape[2])
pca = PCA()
img_pca = pca.fit_transform(img_reshaped)

# %%
# Evaluate the number of principal components needed to retain most of the information
explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Définition des seuils standards
seuils = [0.8, 0.9, 0.95, 0.99]

# Analyse du nombre de composantes nécessaires pour chaque seuil
for seuil in seuils:
    n_composantes = np.where(explained_variance >= seuil)[0][0] + 1
    print(f"Seuil {seuil*100}% : {n_composantes} composantes")

# Visualisation des seuils
plt.figure(figsize=(10, 6))
plt.plot(explained_variance)
for seuil in seuils:
    plt.axhline(y=seuil, color='r', linestyle='--', alpha=0.3)
plt.xlabel('Nombre de composantes principales')
plt.ylabel('Variance expliquée cumulée')
plt.title('Seuils de variance expliquée')
plt.grid(True)
plt.savefig("./resultats/evaluation_composantes.svg")
plt.show()
""" explained_variance = np.cumsum(pca.explained_variance_ratio_)
plt.plot(explained_variance)
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Principal Components')
plt.show() """

# %%

# Projection sur le premier axe (niveaux de gris)
projection_1axe = img_pca[:, 0].reshape(img.shape[0], img.shape[1])
plt.figure(figsize=(8, 8))
plt.imshow(projection_1axe, cmap='gray')
plt.title('Projection sur le premier axe factoriel')
plt.colorbar()
plt.savefig("./resultats/projection_1axe.svg")
plt.show()

# Projection sur les 3 premiers axes (image couleur)
projection_3axes = img_pca[:, :3].reshape(img.shape[0], img.shape[1], 3)
# Normalisation entre 0 et 1
projection_3axes_norm = (projection_3axes - projection_3axes.min()) / (projection_3axes.max() - projection_3axes.min())

plt.figure(figsize=(8, 8))
plt.imshow(projection_3axes_norm)
plt.title('Projection sur les 3 premiers axes factoriels')
plt.colorbar()
plt.savefig("./resultats/projection_3axes.svg")
plt.show()

""" principal_axis_projection = img_pca[:, 0].reshape(img.shape[0], img.shape[1])
plt.imshow(principal_axis_projection, cmap='gray')
plt.title('Projection on the Principal Axis')
plt.show() """

# %%
# Compare with ground truth
plt.imshow(map)
plt.title("Ground truth map")
plt.show()

# %%
# Obtenir les vecteurs propres (loadings)
loadings = pca.components_

# Créer un tableau de longueurs d'onde (à ajuster selon vos données)
wavelengths = np.linspace(400, 2400, img.shape[2])  # exemple pour 220 bandes

# Visualiser les coefficients des 3 premières composantes
plt.figure(figsize=(12, 6))
for i in range(3):
    plt.plot(wavelengths, loadings[i], label=f'CP{i+1}')
plt.xlabel('Longueur d\'onde (nm)')
plt.ylabel('Coefficient de corrélation')
plt.title('Signature spectrale des composantes principales')
plt.legend()
plt.grid(True)
plt.savefig("./resultats/signatures_spectrales.svg")
plt.show()

# Identifier les longueurs d'onde les plus importantes pour CP1
importance_cp1 = np.abs(loadings[0])
top_wavelengths = wavelengths[np.argsort(importance_cp1)[-5:]]
print(f"Longueurs d'onde les plus importantes pour CP1: {top_wavelengths} nm")
# %%