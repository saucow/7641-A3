import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import load_data, nn_layers, nn_reg, nn_iter, cluster_acc, myGMM, clusters, dims, dims_big, run_clustering, pairwiseDistCorr, reconstructionError, ImportanceSelect
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

out = '../results/random_forest/'

cancer_x, cancer_y, housing_x, housing_y = load_data() # cancer, housing

rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5,n_jobs=7)
fs_cancer = rfc.fit(cancer_x,cancer_y).feature_importances_
fs_housing = rfc.fit(housing_x,housing_y).feature_importances_

tmp = pd.Series(np.sort(fs_cancer)[::-1])
tmp.to_csv(out+'cancer part 2.csv')

tmp = pd.Series(np.sort(fs_housing)[::-1])
tmp.to_csv(out+'housing part 2.csv')

dims = list(range(1, 31))
filtr = ImportanceSelect(rfc)
grid ={'filter__n':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_layers}
mlp = MLPClassifier(activation='relu',max_iter=nn_iter,early_stopping=True,random_state=5)
pipe = Pipeline([('filter',filtr),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(cancer_x,cancer_y)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'cancer part 4.csv')


grid ={'filter__n':dims_big,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_layers}  
mlp = MLPClassifier(activation='relu',max_iter=nn_iter,early_stopping=True,random_state=5)
pipe = Pipeline([('filter',filtr),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(housing_x,housing_y)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'housing part 4.csv')

#3
dim = 7
filtr = ImportanceSelect(rfc,dim)
cancer_x2 = filtr.fit_transform(cancer_x,cancer_y)


dim = 7
filtr = ImportanceSelect(rfc,dim)
housing_x2 = filtr.fit_transform(housing_x,housing_y)

run_clustering(out, cancer_x2, cancer_y, housing_x2, housing_y)
