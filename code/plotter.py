import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt
import pandas

colors = ['y', 'c', 'r', 'b', 'g', 'c', 'y']
all_folders = ['clustering', 'ica', 'pca', 'random_forest', 'randomized_projections']
all_folders_name = ['Original', 'ICA', 'PCA', 'Random Forest', 'Randomized Projections']
nn_items = ['', 'param_ica__n_components', 'param_pca__n_components', 'param_filter__n', 'param_rp__n_components']
nn_items_cluster = ['param_gmm__n_components', 'param_km__n_clusters']
nn_cols = ['mean_fit_time', 'mean_test_score', 'mean_train_score', 'param_NN__alpha', 'param_NN__hidden_layer_sizes', 'params']

def save_plot(plot, title):
    plot.savefig('../analysis/plots/' + title.replace(' ', '_').replace('.', 'pt').lower() + '.png')

def get_df(path, headerVal=0):
    return pandas.read_csv('../results/' + path, header=headerVal)

def plot_nn(fileBase, extra, xLabel, title):
    title = title + ' Scoring'
    plt.figure()
    plt.title(title) 

    plt.xlabel(xLabel)
    plt.ylabel('Score')
    
    plt.grid()

    index = 0
    for folder in all_folders:
        fileName = fileBase + extra

        if index == 0:
            folder = 'base'
            fileName = fileBase
            index += 1
            continue

        line = 'o-'
        if index == 0:
            line = 'o-'

        color = colors[index]

        name = all_folders_name[index] 

        df = get_df(folder + '/' + fileName + '.csv').sort_values('rank_test_score')
        best = df.head(1)
        # print(best)

        best_data = best[nn_cols]

        best_data = map(lambda x: str(x), list(best_data.values[0, :]))
        # print(best_data)
        print(name + ' & ' + ' & '.join(best_data) + ' \\\\ \\hline')

        xVar = nn_items[index]

        nnAlpha = list(best['param_NN__alpha'].values)[0]
        nnLayers = list(best['param_NN__hidden_layer_sizes'].values)[0]

        dfLayers = df['param_NN__hidden_layer_sizes'] == nnLayers
        dfAlpha = df['param_NN__alpha'] == nnAlpha 

        best_series = df[dfLayers & dfAlpha].sort_values(xVar)

        xVals = list(set(df[xVar].values))

        vals = best_series['mean_test_score'].values

        plt.plot(xVals, vals, line, color=color,
             label=name)

        index += 1

    plt.legend(loc="best")

    save_plot(plt, title)
    plt.close()

def plot_nn_cluster(fileBase, extra, xLabel, title):
    print('')
    title = title + ' Scoring'
    plt.figure()
    plt.title(title) 

    plt.xlabel(xLabel)
    plt.ylabel('Score')
    
    plt.grid() 

    df = get_df('clustering/Housing cluster GMM.csv').sort_values('rank_test_score')
    best = df.head(1)
    # print(best)

    best_data = best[nn_cols]

    best_data = map(lambda x: str(x), list(best_data.values[0, :]))
    # print(best_data)
    print('Expectation Maximization' + ' & ' + ' & '.join(best_data) + ' \\\\ \\hline')

    xVar = nn_items_cluster[0]

    nnAlpha = list(best['param_NN__alpha'].values)[0]
    nnLayers = list(best['param_NN__hidden_layer_sizes'].values)[0]

    dfLayers = df['param_NN__hidden_layer_sizes'] == nnLayers
    dfAlpha = df['param_NN__alpha'] == nnAlpha 

    best_series = df[dfLayers & dfAlpha].sort_values(xVar)
    # print(best_series)

    xVals = list(set(df[xVar].values))
    xVals.sort() 

    vals = best_series['mean_test_score'].values

    # print(xVals)
    # print(vals)

    plt.plot(xVals, vals, 'o-', color='c',
         label='Expectation Maximization (GMM)')



    df = get_df('clustering/Housing cluster Kmeans.csv').sort_values('rank_test_score')
    best = df.head(1)
    # print(best)

    best_data = best[nn_cols]

    best_data = map(lambda x: str(x), list(best_data.values[0, :]))
    # print(best_data)
    print('k-means' + ' & ' + ' & '.join(best_data) + ' \\\\ \\hline')

    xVar = nn_items_cluster[1]

    nnAlpha = list(best['param_NN__alpha'].values)[0]
    nnLayers = list(best['param_NN__hidden_layer_sizes'].values)[0]

    dfLayers = df['param_NN__hidden_layer_sizes'] == nnLayers
    dfAlpha = df['param_NN__alpha'] == nnAlpha 

    best_series = df[dfLayers & dfAlpha].sort_values(xVar)
    # print(best_series)

    xVals = list(set(df[xVar].values))
    xVals.sort()

    vals = best_series['mean_test_score'].values

    # print(xVals)
    # print(vals)

    plt.plot(xVals, vals, 'o-', color='r',
         label='K Means')

    plt.legend(loc="best")

    save_plot(plt, title)
    plt.close()

plot_nn('cancer', ' dim red', 'Number of Components', '')
plot_nn_cluster('cancer', ' dim red', 'Number of Clusters', '')
