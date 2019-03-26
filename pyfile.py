import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import seaborn as sns
import scipy.stats as st
import missingno as msno
import pandas_profiling as pp
from sklearn.model_selection import KFold
from IPython.display import HTML, display
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis, skew
from sklearn import metrics, datasets, linear_model
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split, KFold
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier

sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')

def log_transform(feature):
    train[feature] = np.log1p(train[feature].values)

def quadratic(feature):
    train[feature+'2'] = train[feature]**2

def error(actual, predicted):
    actual = np.log(actual)
    predicted = np.log(predicted)
    return np.sqrt(np.sum(np.square(actual-predicted))/len(actual))

def bootstrap(sample, stat_func, iters=5000):
    boots = []
    for i in range(iters):
        random_sample = np.random.choice(sample, replace=True, size=len(sample))
        stat = stat_func(random_sample)
        boots.append(stat)
    return boots

def tupler(e):
    count=0
    for i in e:
        i="{0:.0f}".format(i)
        j=e.index[count]
        k=(j,int(i))
        slist.append(k)
        count+=1