# %%
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm
#import seaborn as sns


X = np.transpose(pd.read_csv("./notes.csv",sep=";",header=0,index_col=0))
nomi = list(X.index)
nomv = list(X.columns)
print(X)
print(nomi)
print(nomv)
# %%
#representation de la matiere "français"
plt.hist(X["fran"], bins=10, alpha=0.5, label='français')
plt.legend(loc='upper right')
plt.show()
# %%
plt.hist(X["lati"], bins=10, alpha=0.5, label='latin')
plt.legend(loc='upper right')
plt.show()
# %%
#Afficher les variables "mathematiques" et "français" dans un nuage de points
for i in range(len(nomi)):
    plt.scatter(X["math"][i], X["scie"][i])
    plt.text(X["math"][i], X["scie"][i],nomi[i])
    plt.xlabel("mathematiques")
    plt.ylabel("sciences")
    plt.title("mathematiques vs sciences")
plt.show()

# %%
#Afficher les variables "mathematiques" et "français" dans un nuage de points
for i in range(len(nomi)):
    plt.scatter(X["math"][i], X["d-m "][i])
    plt.text(X["math"][i], X["d-m "][i],nomi[i])
    plt.xlabel("mathematiques")
    plt.ylabel("dessins")
    plt.title("mathematiques vs dessins")
plt.show()
# %%
#1.2 Calcul de l’ACP
#Soit X les donn´ees de dimension p rang´ees dans un tableau de taille n × p. On peut effectuer l’ACP
#en Python en utilisant, par exemple, la commande PCA de scikit-learn :

from sklearn.decomposition import PCA

acp = PCA(n_components=p)