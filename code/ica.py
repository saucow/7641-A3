import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import load_data, nn_layers, nn_reg, nn_iter, cluster_acc, myGMM, clusters, dims, dims_big, run_clustering
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import FastICA

out = '../results/ica/'

cancer_x, cancer_y, housing_x, housing_y = load_data() # cancer, housing

def part2():
	ica = FastICA(random_state=5, max_iter=1000, tol=0.75)
	kurt = {}
	for dim in range(1,31):
		ica.set_params(n_components=dim)
		tmp = ica.fit_transform(cancer_x)
		tmp = pd.DataFrame(tmp)
		tmp = tmp.kurt(axis=0)
		kurt[dim] = tmp.abs().mean()

	kurt = pd.Series(kurt)
	kurt.to_csv(out+'cancer part 2.csv')

	ica = FastICA(random_state=5)
	kurt = {}
	for dim in dims_big:
		ica.set_params(n_components=dim)
		tmp = ica.fit_transform(housing_x)
		tmp = pd.DataFrame(tmp)
		tmp = tmp.kurt(axis=0)
		kurt[dim] = tmp.abs().mean()

	kurt = pd.Series(kurt)
	kurt.to_csv(out+'housing part 2.csv')

def part3():
	dim = 11
	ica = FastICA(n_components=dim,random_state=10)
	cancer_x2 = ica.fit_transform(cancer_x)


	dim = 9
	ica = FastICA(n_components=dim,random_state=10)
	housing_x2 = ica.fit_transform(housing_x)

	run_clustering(out, cancer_x2, cancer_y, housing_x2, housing_y)

def part4():
	dims = list(range(1, 31))
	grid ={'ica__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_layers}
	ica = FastICA(random_state=5)
	mlp = MLPClassifier(activation='relu',max_iter=nn_iter,early_stopping=True,random_state=5)
	pipe = Pipeline([('ica',ica),('NN',mlp)])
	gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

	gs.fit(cancer_x,cancer_y)
	tmp = pd.DataFrame(gs.cv_results_)
	tmp.to_csv(out+'cancer part 4.csv')


	grid ={'ica__n_components':dims_big,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_layers}
	ica = FastICA(random_state=5)
	mlp = MLPClassifier(activation='relu',max_iter=nn_iter,early_stopping=True,random_state=5)
	pipe = Pipeline([('ica',ica),('NN',mlp)])
	gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

	gs.fit(housing_x,housing_y)
	tmp = pd.DataFrame(gs.cv_results_)
	tmp.to_csv(out+'housing part 4.csv')

part2()
part3()
part4()
