import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.metrics import accuracy_score as acc
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.base import TransformerMixin,BaseEstimator
import scipy.sparse as sps
from scipy.linalg import pinv
from collections import defaultdict
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from time import clock
from sklearn.metrics import adjusted_mutual_info_score as ami, homogeneity_score as hs, silhouette_score as ss, completeness_score as cs


nn_layers = [(100,), (50,), (50, 50)]
nn_reg = [10**-x for x in range(1,5)]
nn_iter = 1500

# clusters =  [2,5,10,15,20,25,30,35,40,50, 60, 70]
clusters =  [2,5,10,15,20,25,30,35,40,50, 60, 70]
dims = [2,3, 4, 5, 6, 7,] # 8, 9, 10,15,20,25,30,35,40,45,50,55,60]
dims_big = [2,3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def load_data():
    np.random.seed(0)

    cancer = pd.read_csv('../data/breast.csv');
    cancer_Y_VAL = 'diagnosis'
    cancer.drop(cancer.columns[[-1, 0]], axis=1, inplace=True)
    cancer.info()
    diag_map = {'M': 1, 'B': 0}
    cancer['diagnosis'] = cancer['diagnosis'].map(diag_map)

    # features_mean = list(cancer_data.columns[1:11])
    # cancer_x = cancer_data.loc[:, features_mean]
    # cancer_y = cancer_data.loc[:, 'diagnosis']

    housing = pd.read_csv('../data/housing.csv')
    housing_Y_VAL = 'price_bracket'

    cancer_y = cancer[cancer_Y_VAL].copy().values
    cancer_x = cancer.drop(cancer_Y_VAL, 1).copy().values

    housing_y = housing[housing_Y_VAL].copy().values
    housing_x = housing.drop(housing_Y_VAL, 1).copy().values

    cancer_x = StandardScaler().fit_transform(cancer_x)
    housing_x = StandardScaler().fit_transform(housing_x)

    return (cancer_x, cancer_y, housing_x, housing_y)


def cluster_acc(Y,clusterLabels):
    assert (Y.shape == clusterLabels.shape)
    pred = np.empty_like(Y)
    for label in set(clusterLabels):
        mask = clusterLabels == label
        sub = Y[mask]
        target = Counter(sub).most_common(1)[0][0]
        pred[mask] = target
    #    assert max(pred) == max(Y)
    #    assert min(pred) == min(Y)
    return acc(Y,pred)


class myGMM(GMM):
    def transform(self,X):
        return self.predict_proba(X)


def run_clustering(out, cancer_x, cancer_y, housing_x, housing_y):
    SSE = defaultdict(dict)
    ll = defaultdict(dict)
    acc = defaultdict(lambda: defaultdict(dict))
    adjMI = defaultdict(lambda: defaultdict(dict))
    km = kmeans(random_state=5)
    gmm = GMM(random_state=5)

    silhouette = defaultdict(lambda: defaultdict(dict))
    completeness = defaultdict(lambda: defaultdict(dict))
    homogeniety = defaultdict(lambda: defaultdict(dict))

    st = clock()
    for k in range(2, 20, 1):
        km.set_params(n_clusters=k)
        gmm.set_params(n_components=k)
        km.fit(cancer_x)
        gmm.fit(cancer_x)

        SSE[k]['cancer'] = km.score(cancer_x)
        ll[k]['cancer'] = gmm.score(cancer_x)

        acc[k]['cancer']['Kmeans'] = cluster_acc(cancer_y, km.predict(cancer_x))
        acc[k]['cancer']['GMM'] = cluster_acc(cancer_y, gmm.predict(cancer_x))

        adjMI[k]['cancer']['Kmeans'] = ami(cancer_y, km.predict(cancer_x))
        adjMI[k]['cancer']['GMM'] = ami(cancer_y, gmm.predict(cancer_x))

        silhouette[k]['cancer']['Kmeans Silhouette'] = ss(cancer_x, km.predict(cancer_x))
        silhouette[k]['cancer']['GMM Silhouette'] = ss(cancer_x, gmm.predict(cancer_x))

        completeness[k]['cancer']['Kmeans Completeness'] = cs(cancer_y, km.predict(cancer_x))
        completeness[k]['cancer']['GMM Completeness'] = cs(cancer_y, gmm.predict(cancer_x))

        homogeniety[k]['cancer']['Kmeans Homogeniety'] = hs(cancer_y, km.predict(cancer_x))
        homogeniety[k]['cancer']['GMM Homogeniety'] = hs(cancer_y, gmm.predict(cancer_x))

        km.fit(housing_x)
        gmm.fit(housing_x)
        SSE[k]['housing'] = km.score(housing_x)
        ll[k]['housing'] = gmm.score(housing_x)

        acc[k]['housing']['Kmeans'] = cluster_acc(housing_y, km.predict(housing_x))
        acc[k]['housing']['GMM'] = cluster_acc(housing_y, gmm.predict(housing_x))

        adjMI[k]['housing']['Kmeans'] = ami(housing_y, km.predict(housing_x))
        adjMI[k]['housing']['GMM'] = ami(housing_y, gmm.predict(housing_x))

        silhouette[k]['housing']['Kmeans Silhouette'] = ss(housing_x, km.predict(housing_x))
        silhouette[k]['housing']['GMM Silhouette'] = ss(housing_x, gmm.predict(housing_x))

        completeness[k]['housing']['Kmeans Completeness'] = cs(housing_y, km.predict(housing_x))
        completeness[k]['housing']['GMM Completeness'] = cs(housing_y, gmm.predict(housing_x))

        homogeniety[k]['housing']['Kmeans Homogeniety'] = hs(housing_y, km.predict(housing_x))
        homogeniety[k]['housing']['GMM Homogeniety'] = hs(housing_y, gmm.predict(housing_x))

        print(k, clock() - st)
    SSE = (-pd.DataFrame(SSE)).T
    SSE.rename(columns = lambda x: x+' SSE (left)',inplace=True)
    ll = pd.DataFrame(ll).T
    ll.rename(columns = lambda x: x+' log-likelihood',inplace=True)
    acc = pd.Panel(acc)

    adjMI = pd.Panel(adjMI)

    silhouette = pd.Panel(silhouette)
    completeness = pd.Panel(completeness)
    homogeniety = pd.Panel(homogeniety)


    SSE.to_csv(out+'SSE.csv')
    ll.to_csv(out+'logliklihood.csv')
    acc.ix[:,:,'housing'].to_csv(out+'Housing acc.csv')
    acc.ix[:,:,'cancer'].to_csv(out+'Perm acc.csv')

    adjMI.ix[:,:,'housing'].to_csv(out+'Housing adjMI.csv')
    adjMI.ix[:,:,'cancer'].to_csv(out+'Perm adjMI.csv')


    silhouette.ix[:,:,'cancer'].to_csv(out+'Perm silhouette.csv')
    completeness.ix[:,:,'cancer'].to_csv(out+'Perm completeness.csv')
    homogeniety.ix[:,:,'cancer'].to_csv(out+'Perm homogeniety.csv')

    silhouette.ix[:,:,'housing'].to_csv(out+'housing silhouette.csv')
    completeness.ix[:,:,'housing'].to_csv(out+'housing completeness.csv')
    homogeniety.ix[:,:,'housing'].to_csv(out+'housing homogeniety.csv')

def pairwiseDistCorr(X1,X2):
    assert X1.shape[0] == X2.shape[0]
    d1 = pairwise_distances(X1)
    d2 = pairwise_distances(X2)
    return np.corrcoef(d1.ravel(),d2.ravel())[0,1]

def aveMI(X,Y):
    MI = MIC(X,Y)
    return np.nanmean(MI)

def reconstructionError(projections,X):
    W = projections.components_
    if sps.issparse(W):
        W = W.todense()
    p = pinv(W)
    reconstructed = ((p@W)@(X.T)).T # Unproject projected data
    errors = np.square(X-reconstructed)
    return np.nanmean(errors)

class ImportanceSelect(BaseEstimator, TransformerMixin):
    def __init__(self, model, n=1):
        self.model = model
        self.n = n
    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self
    def transform(self, X):
        return X[:,self.model.feature_importances_.argsort()[::-1][:self.n]]