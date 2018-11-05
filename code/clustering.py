import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from time import clock
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
from helpers import load_data, nn_layers, nn_reg, nn_iter, cluster_acc, myGMM
from sklearn.metrics import adjusted_mutual_info_score as ami, homogeneity_score as hs, silhouette_score as ss, completeness_score as cs
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import sys
out = '../results/clustering/'
cancer_x, cancer_y, house_prices_x, house_prices_y = load_data() # cancer, house_prices

def part1():
	SSE = defaultdict(dict) # some of squared errors
	ll = defaultdict(dict) # log likelihood
	acc = defaultdict(lambda: defaultdict(dict))
	adjMI = defaultdict(lambda: defaultdict(dict))

	silhouette = defaultdict(lambda: defaultdict(dict))
	completeness = defaultdict(lambda: defaultdict(dict))
	homogeniety = defaultdict(lambda: defaultdict(dict))

	km = kmeans(random_state=5)
	gmm = GMM(random_state=5)

	st = clock()
	for k in range(2, 20, 1):
		km.set_params(n_clusters=k)
		gmm.set_params(n_components=k)
		km.fit(cancer_x)
		gmm.fit(cancer_x)

		SSE[k]['cancer'] = km.score(cancer_x)
		ll[k]['cancer'] = gmm.score(cancer_x)

		acc[k]['cancer']['Kmeans'] = cluster_acc(cancer_y,km.predict(cancer_x))
		acc[k]['cancer']['GMM'] = cluster_acc(cancer_y,gmm.predict(cancer_x))

		adjMI[k]['cancer']['Kmeans'] = ami(cancer_y,km.predict(cancer_x))
		adjMI[k]['cancer']['GMM'] = ami(cancer_y,gmm.predict(cancer_x))

		silhouette[k]['cancer']['Kmeans'] = ss(cancer_x, km.predict(cancer_x))
		silhouette[k]['cancer']['GMM'] = ss(cancer_x, gmm.predict(cancer_x))

		completeness[k]['cancer']['Kmeans'] = cs(cancer_y, km.predict(cancer_x))
		completeness[k]['cancer']['GMM'] = cs(cancer_y, gmm.predict(cancer_x))

		homogeniety[k]['cancer']['Kmeans'] = hs(cancer_y, km.predict(cancer_x))
		homogeniety[k]['cancer']['GMM'] = hs(cancer_y, gmm.predict(cancer_x))

		km.fit(house_prices_x)
		gmm.fit(house_prices_x)
		SSE[k]['house_prices'] = km.score(house_prices_x)
		ll[k]['house_prices'] = gmm.score(house_prices_x)

		acc[k]['house_prices']['Kmeans'] = cluster_acc(house_prices_y,km.predict(house_prices_x))
		acc[k]['house_prices']['GMM'] = cluster_acc(house_prices_y,gmm.predict(house_prices_x))

		adjMI[k]['house_prices']['Kmeans'] = ami(house_prices_y,km.predict(house_prices_x))
		adjMI[k]['house_prices']['GMM'] = ami(house_prices_y,gmm.predict(house_prices_x))

		silhouette[k]['house_prices']['Kmeans'] = ss(house_prices_x, km.predict(house_prices_x))
		silhouette[k]['house_prices']['GMM'] = ss(house_prices_x, gmm.predict(house_prices_x))

		completeness[k]['house_prices']['Kmeans'] = cs(house_prices_y, km.predict(house_prices_x))
		completeness[k]['house_prices']['GMM'] = cs(house_prices_y, gmm.predict(house_prices_x))

		homogeniety[k]['house_prices']['Kmeans'] = hs(house_prices_y, km.predict(house_prices_x))
		homogeniety[k]['house_prices']['GMM'] = hs(house_prices_y, gmm.predict(house_prices_x))

		print(k, clock()-st)


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
	acc.ix[:,:,'house_prices'].to_csv(out+'Housing acc.csv')
	acc.ix[:,:,'cancer'].to_csv(out+'Cancer acc.csv')

	adjMI.ix[:,:,'house_prices'].to_csv(out+'Housing adjMI.csv')
	adjMI.ix[:,:,'cancer'].to_csv(out+'Cancer adjMI.csv')


	silhouette.ix[:,:,'cancer'].to_csv(out+'Cancer silhouette.csv')
	completeness.ix[:,:,'cancer'].to_csv(out+'Cancer completeness.csv')
	homogeniety.ix[:,:,'cancer'].to_csv(out+'Cancer homogeniety.csv')

	silhouette.ix[:,:,'house_prices'].to_csv(out+'house_prices silhouette.csv')
	completeness.ix[:,:,'house_prices'].to_csv(out+'house_prices completeness.csv')
	homogeniety.ix[:,:,'house_prices'].to_csv(out+'house_prices homogeniety.csv')



def part4():
	clusters = list(range(2, 20, 1))
	grid ={'km__n_clusters':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_layers}
	mlp = MLPClassifier(activation='relu',max_iter=nn_iter,early_stopping=True,random_state=5)
	km = kmeans(random_state=5)
	pipe = Pipeline([('km',km),('NN',mlp)])
	gs = GridSearchCV(pipe,grid,verbose=10)

	gs.fit(cancer_x,cancer_y)
	tmp = pd.DataFrame(gs.cv_results_)
	tmp.to_csv(out+'Cancer cluster Kmeans.csv')

	grid ={'gmm__n_components':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_layers}
	mlp = MLPClassifier(activation='relu',max_iter=nn_iter,early_stopping=True,random_state=5)
	gmm = myGMM(random_state=5)
	pipe = Pipeline([('gmm',gmm),('NN',mlp)])
	gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

	gs.fit(cancer_x,cancer_y)
	tmp = pd.DataFrame(gs.cv_results_)
	tmp.to_csv(out+'Cancer cluster GMM.csv')

	grid ={'km__n_clusters':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_layers}
	mlp = MLPClassifier(activation='relu',max_iter=nn_iter,early_stopping=True,random_state=5)
	km = kmeans(random_state=5)
	pipe = Pipeline([('km',km),('NN',mlp)])
	gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

	gs.fit(house_prices_x,house_prices_y)
	tmp = pd.DataFrame(gs.cv_results_)
	tmp.to_csv(out+'Housing cluster Kmeans.csv')

	grid ={'gmm__n_components':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_layers}
	mlp = MLPClassifier(activation='relu',max_iter=nn_iter,early_stopping=True,random_state=5)
	gmm = myGMM(random_state=5)
	pipe = Pipeline([('gmm',gmm),('NN',mlp)])
	gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

	gs.fit(house_prices_x,house_prices_y)
	tmp = pd.DataFrame(gs.cv_results_)
	tmp.to_csv(out+'Housing cluster GMM.csv')

	# %% For chart 4/5
	cancer_x2D = TSNE(verbose=10,random_state=5).fit_transform(cancer_x)
	house_prices_x2D = TSNE(verbose=10,random_state=5).fit_transform(house_prices_x)

	Cancer2D = pd.DataFrame(np.hstack((cancer_x2D,np.atleast_2d(cancer_y).T)),columns=['x','y','target'])
	Housing2D = pd.DataFrame(np.hstack((house_prices_x2D,np.atleast_2d(house_prices_y).T)),columns=['x','y','target'])
	Cancer2D.to_csv(out+'Cancer2D.csv')
	Housing2D.to_csv(out+'Housing2D.csv')

part1()
part4()
