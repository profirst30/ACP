# %%
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os

# Création du dossier resultats s'il n'existe pas
if not os.path.exists("resultats"):
    os.makedirs("resultats")

# %%
# Chargement des données
X = np.transpose(pd.read_csv("./notes.csv", sep=";", header=0, index_col=0))
nomi = list(X.index)
nomv = list(X.columns)
print(nomi)
print(nomv)
print(X)

# %%
# Histogramme des notes en français
plt.hist(X["fran"], bins=20, label='français')
plt.legend(loc='upper right')
plt.xlabel("Notes")
plt.ylabel("Effectif")

plt.savefig("./resultats/histogramme_francais.svg")
plt.title("Histogramme des notes en français")
plt.show()

# Histogramme des notes en latin
plt.hist(X["lati"], bins=20, label="latin")
plt.legend(loc='upper right')
plt.xlabel("Notes")
plt.ylabel("Effectif")

plt.savefig("./resultats/histogramme_latin.svg")
plt.title("Histogramme des notes en latin")
plt.show()

# %%
# Nuage de points des notes en mathématiques en fonction des notes en sciences
for i in range(len(nomi)):
    plt.scatter(X["math"][i], X["scie"][i])
    plt.xlabel("Notes en mathématiques")
    plt.ylabel("Notes en sciences")
    plt.text(X["math"][i], X["scie"][i], nomi[i])

plt.savefig("./resultats/nuage_de_points_math_sciences.svg")
plt.title("Nuage de points des notes en mathématiques et en sciences")
plt.show()

# Nuage de points des notes en mathématiques en fonction des notes en dessin
for i in range(len(nomi)):
    plt.scatter(X["math"][i], X["d-m "][i])
    plt.xlabel("Notes en mathématiques")
    plt.ylabel("Notes en dessin")
    plt.text(X["math"][i], X["d-m "][i], nomi[i])

plt.savefig("./resultats/nuage_de_points_math_dessin.svg")
plt.title("Nuage de points des notes en mathématiques et en dessin")
plt.show()

# %%
# Calcul de l'ACP
p = len(nomv)
acp = PCA(n_components=p)
cc = acp.fit_transform(X)  # cc contient les projections
print("Composantes principales:\n", acp.components_)
print("Variance expliquée par chaque composante:\n", acp.explained_variance_)
print("Ratio de variance expliquée par chaque composante:\n", acp.explained_variance_ratio_)
var_cumul = np.cumsum(acp.explained_variance_ratio_)

# Tracer la courbe de la variance expliquée
plt.figure(figsize=(10, 6))
plt.plot(range(1, p + 1), var_cumul, marker='o', linestyle='--')

plt.xlabel('Composante principale')
plt.ylabel('Ratio de variance expliquée')
plt.xticks(range(1, p + 1))
plt.grid(True)
plt.savefig("./resultats/variance_cumulee_.svg")
plt.title('Variance expliquée par chaque composante principale')
plt.show()

# Cercle des corrélations
data2_proj = acp.transform(X)
print("Nouvelles projections:\n", data2_proj)

# Représentation des individus dans les plans factoriels
plt.figure(figsize=(12, 5))

# Plan E1∪E2
plt.subplot(121)
plt.scatter(cc[:, 0], cc[:, 1])
plt.xlabel('Première composante principale')
plt.ylabel('Deuxième composante principale')
plt.title('Projection dans E1∪E2')
for i in range(len(nomi)):
    plt.annotate(nomi[i], (cc[i, 0], cc[i, 1]))

# Plan E1∪E3
plt.subplot(122)
plt.scatter(cc[:, 0], cc[:, 2])
plt.xlabel('Première composante principale')
plt.ylabel('Troisième composante principale')
plt.title('Projection dans E1∪E3')
for i in range(len(nomi)):
    plt.annotate(nomi[i], (cc[i, 0], cc[i, 2]))
    
plt.savefig("./resultats/projection_des_eleves.svg")
plt.suptitle("Projection des élèves dans les plans factoriels")
plt.tight_layout()
# ajoute un titre general à la figure

plt.show()

# Calcul des corrélations
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

correlations = np.zeros((p, p))
for k in range(p):
    for j in range(p):
        correlations[k, j] = acp.components_[j, k] * np.sqrt(acp.explained_variance_[j])

print("\nCorrélations entre variables initiales et composantes principales:")
corr_df = pd.DataFrame(correlations, 
                      columns=[f'CP{i+1}' for i in range(p)],
                      index=nomv)
print(corr_df)

# %%
# Représentation biplot
from affichage_acp import my_biplot

my_biplot(
    score=cc[:, 0:2],                        # Coordonnées des élèves sur les deux premières composantes
    coeff=np.transpose(acp.components_[0:2, :]), # Coefficients des variables sur les axes
    coeff_labels=nomv,                       # Noms des matières
    score_labels=nomi,                       # Noms des élèves
    nomx="PC1",                              # Nom de l'axe X (première composante principale)
    nomy="PC2"                               # Nom de l'axe Y (deuxième composante principale)
)

