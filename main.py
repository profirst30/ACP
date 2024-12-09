# %%
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm
#import seaborn as sns

# creation du dossier resultats s'il n'existe pas
import os
if not os.path.exists("resultats"):
    os.makedirs("resultats")

# %%
X = np.transpose(pd.read_csv("./notes.csv",sep=";",header=0,index_col=0))
nomi = list(X.index)
nomv = list(X.columns)
print(nomi)
print(nomv)
print(X)
# %%
# francais 
plt.hist(X["fran"], bins=20, label='français')
plt.legend(loc='upper right')
plt.xlabel("Notes")
plt.ylabel("Effectif")
plt.savefig("./resultats/histogramme_francais.svg")
plt.title("Histogramme des notes en français")
plt.show()

# latin
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


# graphiquement on remarque que les notes en mathématiques et en sciences semblent corrélées, alors que les notes en mathématiques et en dessin ne semblent pas corrélées.


# %%
# Calcul de l'ACP
from sklearn.decomposition import PCA


p = len(nomv)

acp = PCA(n_components=p)
cc = acp.fit_transform(X) # cc contient les projections