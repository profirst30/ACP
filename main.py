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
# %%
