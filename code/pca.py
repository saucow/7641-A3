import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import load_data, nn_layers, nn_reg, nn_iter, cluster_acc, myGMM, clusters, dims, dims_big, run_clustering
# from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

import pandas as pd

out = '../results/pca/'

cancer_x, cancer_y, housing_x, housing_y = load_data() # cancer, housing
# raise Exception('Remove this line to run code')

def part2():
	pca = PCA(random_state=5)
	pca.fit(cancer_x)
	tmp = pd.Series(data = pca.explained_variance_,index = range(1,31))
	tmp1 = pd.Series(data = pca.explained_variance_ratio_,index = range(1,31))
	tmp2 = pd.Series(data = pca.explained_variance_ratio_.cumsum(),index = range(1,31))
	tmp3 = pd.Series(data = pca.singular_values_,index = range(1,31))
	pd.concat([tmp, tmp1, tmp2, tmp3], axis=1).to_csv(out+'cancer scree.csv')


	pca = PCA(random_state=5)
	pca.fit(housing_x)
	tmp = pd.Series(data = pca.explained_variance_,index = range(1,13))
	tmp1 = pd.Series(data = pca.explained_variance_ratio_,index = range(1,13))
	tmp2 = pd.Series(data = pca.explained_variance_ratio_.cumsum(),index = range(1,13))
	tmp3 = pd.Series(data = pca.singular_values_,index = range(1,13))
	pd.concat([tmp, tmp1, tmp2, tmp3], axis=1).to_csv(out+'housing scree.csv')

def part3():
	dim = 6
	pca = PCA(n_components=dim,random_state=10)
	cancer_x2 = pca.fit_transform(cancer_x)

	dim = 9
	pca = PCA(n_components=dim,random_state=10)
	housing_x2 = pca.fit_transform(housing_x)

	run_clustering(out, cancer_x2, cancer_y, housing_x2, housing_y)

def part4():
	dims = list(range(1, 31))
	grid ={'pca__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_layers}
	pca = PCA(random_state=5)
	mlp = MLPClassifier(activation='relu',max_iter=nn_iter,early_stopping=True,random_state=5)
	pipe = Pipeline([('pca',pca),('NN',mlp)])
	gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

	gs.fit(cancer_x,cancer_y)
	tmp = pd.DataFrame(gs.cv_results_)
	tmp.to_csv(out+'cancer part 4.csv')


	dims = dims_big
	grid ={'pca__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_layers}
	pca = PCA(random_state=5)
	mlp = MLPClassifier(activation='relu',max_iter=nn_iter,early_stopping=True,random_state=5)
	pipe = Pipeline([('pca',pca),('NN',mlp)])
	gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

	gs.fit(housing_x,housing_y)
	tmp = pd.DataFrame(gs.cv_results_)
	tmp.to_csv(out+'housing part 4.csv')

part2()
part3()
part4()
