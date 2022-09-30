from __future__ import absolute_import
from __future__ import print_function
from cmath import inf

from mimic3benchmark.readers import LOSReader
from mimic3models import common_utils
from mimic3models.metrics import print_metrics_binary
from mimic3models.los3days.utils import save_results
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

import os
import numpy as np
import argparse
import json

# Clustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.manifold import TSNE
import seaborn as sns

def read_and_extract_features(reader, period, features):
    ret = common_utils.read_chunk(reader, reader.get_number_of_examples())
    # ret = common_utils.read_chunk(reader, 100)
    X = common_utils.extract_features_from_rawdata(ret['X'], ret['header'], period, features)
    return (X, ret['y'], ret['name'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--C', type=float, default=1.0, help='inverse of L1 / L2 regularization')
    parser.add_argument('--l1', dest='l2', action='store_false')
    parser.add_argument('--l2', dest='l2', action='store_true')
    parser.set_defaults(l2=True)
    parser.add_argument('--period', type=str, default='all', help='specifies which period extract features from',
                        choices=['first4days', 'first8days', 'last12hours', 'first25percent', 'first50percent', 'all'])
    parser.add_argument('--features', type=str, default='all', help='specifies what features to extract',
                        choices=['all', 'len', 'all_but_len'])
    parser.add_argument('--data', type=str, help='Path to the data of length of stay task',
                        default=os.path.join(os.path.dirname(__file__), '../../../data/los3days/'))
    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                        default='.')
    args = parser.parse_args()
    print(args)

    train_reader = LOSReader(dataset_dir=os.path.join(args.data, 'train'),
                                             listfile=os.path.join(args.data, 'train_listfile.csv'),
                                             period_length=24.0)

    val_reader = LOSReader(dataset_dir=os.path.join(args.data, 'train'),
                                           listfile=os.path.join(args.data, 'val_listfile.csv'),
                                           period_length=24.0)

    test_reader = LOSReader(dataset_dir=os.path.join(args.data, 'test'),
                                            listfile=os.path.join(args.data, 'test_listfile.csv'),
                                            period_length=24.0)

    print('Reading data and extracting features ...')
    (train_X, train_y, train_names) = read_and_extract_features(train_reader, args.period, args.features)
    (val_X, val_y, val_names) = read_and_extract_features(val_reader, args.period, args.features)
    (test_X, test_y, test_names) = read_and_extract_features(test_reader, args.period, args.features)
    print('  train data shape = {}'.format(train_X.shape))
    print('  validation data shape = {}'.format(val_X.shape))
    print('  test data shape = {}'.format(test_X.shape))

    print('Imputing missing values ...')
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(train_X)
    train_X = np.array(imputer.transform(train_X), dtype=np.float32)
    val_X = np.array(imputer.transform(val_X), dtype=np.float32)
    test_X = np.array(imputer.transform(test_X), dtype=np.float32)

    print('Normalizing the data to have zero mean and unit variance ...')
    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    val_X = scaler.transform(val_X)
    test_X = scaler.transform(test_X)

    # Plotting the Clusters using matplotlib
    n=4
    pca = PCA(n_components=n, random_state=42)
    print("Train shape:", train_X.shape)
    pca_results = pca.fit_transform(train_X)
    print("Nº components:", pca_results.shape[1])
    print(pca.explained_variance_ratio_)
    print("Cumsum:", np.sum(pca.explained_variance_ratio_))

    labels = {
        str(i): f"PC {i+1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }

    # Plot 2d
    df = pd.DataFrame()
    df["y"] = train_y
    df['pca-one'] = pca_results[:,0]
    df['pca-two'] = pca_results[:,1]

    #plt.figure(figsize=(13, 10))
    sns.scatterplot(x="pca-one", y="pca-two", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 2),
                    data=df).set(title="PCA-2d projection for LOS>3")
    plt.show()

    # T-SNE
    tsne = TSNE(n_components=2, verbose=1, random_state=42)
    tsne_results = tsne.fit_transform(train_X)
    print(tsne_results.shape)

    df = pd.DataFrame()
    df["y"] = train_y
    df['tsne-one'] = tsne_results[:,0]
    df['tsne-two'] = tsne_results[:,1] 

    sns.scatterplot(x="tsne-one", y="tsne-two", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 2),
                    data=df).set(title="T-SNE projection for LOS>3") 
    plt.show()
    
    # PCA(50) + TSNE
    n=50
    pca = PCA(n_components=n, random_state=42)
    print("Train shape:", train_X.shape)
    pca_50 = pca.fit_transform(train_X)
    print("Nº components:", pca_50.shape[1])
    print(pca.explained_variance_ratio_)
    print("Cumsum:", np.sum(pca.explained_variance_ratio_))

    labels = {
        str(i): f"PC {i+1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }

    tsne = TSNE(n_components=2,verbose=1,random_state=42)
    tsne_pca_results = tsne.fit_transform(pca_50)

    df = pd.DataFrame()
    df["y"] = train_y
    df['tsne-one'] = tsne_pca_results[:,0]
    df['tsne-two'] = tsne_pca_results[:,1] 

    sns.scatterplot(x="tsne-one", y="tsne-two", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 2),
                    data=df).set(title="PCA + T-SNE projection for LOS>3") 
    plt.show()

    '''
    # Silhouette
    silhouette_coefficients = []
    kmeans_kwargs= {
        "init":"random",
        "n_init":10,
        "max_iter":300,
        "random_state":42
    }
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(train_X)
        score = silhouette_score(train_X, kmeans.labels_)
        silhouette_coefficients.append(score)
        
    # Plotting graph to choose the best number of clusters
    # with the most Silhouette Coefficient score
    print(silhouette_coefficients)
    plt.style.use("fivethirtyeight")
    plt.plot(range(2, 10), silhouette_coefficients)
    plt.xticks(range(2, 10))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    plt.show()'''

    print("K-means Training...")
    # KMeans K=3,4
    n_clusters = 3
    kmeans = KMeans(
        init="random",
        n_clusters=n_clusters,
        n_init=10,
        max_iter=300,
        random_state=42
    )
    # Fit to the training data
    kmeans.fit(train_X)
    train_df = pd.DataFrame(train_X)
    # Generate out clusters
    train_cluster = kmeans.predict(train_X)
    n_columns = train_X.shape[1]

    # Add predicted cluster and y regression label to our training DataFrame
    train_df.insert(n_columns,'cluster', train_cluster)
    train_df.insert(n_columns+1,'y', train_y) 
    train_clusters_df = []
    for i in range(n_clusters):
        train_clusters_df.append(train_df[train_df['cluster']==i])

    # Logistic Regression
    penalty = ('l2' if args.l2 else 'l1')
    file_name = '{}.{}.{}.C{}'.format(args.period, args.features, penalty, args.C)
    logreg = LogisticRegression(penalty=penalty, C=args.C, random_state=42)
    k_logregs = []
    
    # One model for each cluster
    print("-------- Logistic Regression fitting for each cluster --------")
    for cluster_df in train_clusters_df:
        cluster_X = cluster_df.iloc[:,:n_columns] # X values
        cluster_Y = cluster_df.iloc[:,n_columns+1] # Y value
        logreg.fit(cluster_X, cluster_Y)
        k_logregs.append(logreg)
        print("Training cluster shape: ", cluster_df.shape)
    
    # Test sample es asignado al cluster correspondiente mediante Distancia euclidiana y se aplica el modelo correspondiente
    print("Assigning each test sample to the closest cluster centroid...")
    test_df = pd.DataFrame(test_X)
    test_clusters_df = test_df.copy()
    test_clusters_df["cluster"] = None
    test_clusters_df.insert(n_columns+1,'y', test_y) 
    i=0
    for row in test_df.itertuples(index=False):
        min_distance = float('inf')
        closest_cluster = None
        for k in range(kmeans.cluster_centers_.shape[0]):
            # Check if the assigned cluster has more than 100 samples
            # if train_clusters_df[k].shape[0] > 100: # Probar sin limite
            distance = np.linalg.norm(kmeans.cluster_centers_[k]-row)
            if distance < min_distance:
                min_distance = distance
                closest_cluster = k
        # Assign cluster to test sample
        test_clusters_df.iloc[i, n_columns] = closest_cluster
        i += 1

    test_clusters = []
    for i in range(n_clusters):
        test_cluster_i = test_clusters_df[test_clusters_df['cluster']==i]
        if np.size(test_cluster_i) != 0:
            test_clusters.append(test_cluster_i)
            print("Test cluster shape: ", test_cluster_i.shape)
    
    # Results
    result_dir = os.path.join(args.output_dir, 'results')
    common_utils.create_directory(result_dir)

    # Train metrics
    print("-------- Train metrics ---------")
    i = 0
    # For each cluster, predict probabilities of class labels
    for cluster_df in train_clusters_df:
        cluster_X = cluster_df.iloc[:,:n_columns] # X values
        cluster_Y = cluster_df.iloc[:,n_columns+1] # Y value
        classes_prob = k_logregs[i].predict_proba(cluster_X)
        if i == 0:
            train_X_probs = np.array(classes_prob)
            train_y = np.array(cluster_Y)
        else: 
            train_X_probs = np.concatenate((train_X_probs, classes_prob))
            train_y = np.concatenate((train_y, cluster_Y))
        i += 1
    with open(os.path.join(result_dir, 'train_{}.json'.format(file_name)), 'w') as res_file:
        # Print and save the results
        ret = print_metrics_binary(train_y, train_X_probs)        
        ret = {k : float(v) for k, v in ret.items()}
        json.dump(ret, res_file)

    # FALTA VALIDATION
    # Validation metrics
    print("-------- Validation metrics ---------")
    with open(os.path.join(result_dir, 'val_{}.json'.format(file_name)), 'w') as res_file:
        ret = print_metrics_binary(val_y, logreg.predict_proba(val_X))
        ret = {k: float(v) for k, v in ret.items()}
        json.dump(ret, res_file)

    # Test metrics
    print("-------- Test metrics ---------")
    i = 0
    # For each cluster, predict probabilities of class labels
    for cluster_df in test_clusters:
        cluster_X = cluster_df.iloc[:,:n_columns] # X values
        cluster_Y = cluster_df.iloc[:,n_columns+1] # Y value
        classes_prob = k_logregs[i].predict_proba(cluster_X)[:,1]
        if i == 0:
            test_X_probs = np.array(classes_prob)
            test_y = np.array(cluster_Y)
        else: 
            test_X_probs = np.concatenate((test_X_probs, classes_prob))
            test_y = np.concatenate((test_y, cluster_Y))
        i += 1
    with open(os.path.join(result_dir, 'test_{}.json'.format(file_name)), 'w') as res_file:
        ret = print_metrics_binary(test_y, test_X_probs)
        ret = {k: float(v) for k, v in ret.items()}
        json.dump(ret, res_file)

    save_results(test_names, test_X_probs, test_y,
                 os.path.join(args.output_dir, 'predictions', file_name + '.csv'))


if __name__ == '__main__':
    main()