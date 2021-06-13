import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from sklearn.utils import resample
from collections import defaultdict
pd.options.mode.chained_assignment = None  # default='warn'


def spearman_coeff(df, features, target):
    """ComputeSpearman's rank correlation coefficient 
       between each feature and target;
       Return a dictionary with key as feature name and 
       value as importance value;
       """
    coeff = {}
    for feature in features:
        coeff[feature] = spearmanr(df[feature], df[target]).correlation

    return coeff


def visualize_importances(feature_importances, figsize=(6, 4)):
    """Visualize the featured importances;
       Most important feature will be shown on the top."""
    n = len(feature_importances)
    feature_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))     #   Sorted by value in descending order.
    importances = list(feature_importances.values())
    features = list(feature_importances.keys())
    
    fig, ax = plt.subplots(figsize=figsize)
    # set grindlines
    ax.yaxis.grid(color='grey', linestyle='--', linewidth=1.5)

    for i in range(n):
        p1 = ax.scatter(importances[i], n-i, c='blue', s=30)

    # hide the spines
    ax.spines['left'].set_color('gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # set ticklabels of y-axis
    ax.set_yticks(np.arange(1, n+1))
    ax.set_yticklabels(reversed(features), fontsize=12, backgroundcolor='w')

    # hide the ticks
    ax.tick_params(axis='x', color='white')
    
    return ax


def dropcol_importances(model, X_train, y_train, X_valid, y_valid, score):
    """Drop column importance;
       Return a dictionary with key as feature name and 
       value as importance value;"""
    model.fit(X_train, y_train)
    baseline = score(y_valid, model.predict(X_valid))
    imp = {}
    for col in X_train.columns:
        X_train_ = X_train.drop(col, axis=1)
        X_valid_ = X_valid.drop(col, axis=1)
        model_ = model
        model_.fit(X_train_, y_train)
        m = score(y_valid, model_.predict(X_valid_))
        imp[col] = (baseline - m)
    return imp


def permutation_importances(model, X_valid, y_valid, score):
    """Permutation importance;
       Return a dictionary with key as feature name and 
       value as importance value;"""
    baseline = score(y_valid, model.predict(X_valid))
    imp = {}
    for col in X_valid.columns:
        save = X_valid[col].copy()
        X_valid[col] = np.random.permutation(X_valid[col])
        m = score(y_valid, model.predict(X_valid))
        X_valid[col] = save.values
        imp[col] = (baseline - m)
    return imp


def model_w_top_k_features(df, y, sorted_importances, model, max_k):
    """return a dictionary with key as k and value as 5-fold CV mae;
       different k value corresponds to different features in X"""
    k_mae = {}
    for k in range(1, max_k+1):
        # get the top k most important features 
        top_k_features = list(sorted_importances.keys())[:k]

        # set the columns of X as the k features
        X = df[top_k_features].values

        mae = []
        # run cross validation
        kfold = KFold(n_splits=5, random_state=1, shuffle=True)

        for train_index, test_index in kfold.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            mae.append(mean_absolute_error(y_test, y_test_pred))

        k_mae[k] = np.mean(mae)
    return k_mae


def model_w_pca(X, y, model, max_k):
    """return a dictionary with key as k and value as 5-fold CV mae;
       different k value corresponds to different number of components for PCA;
       input X is supposed to be standardized already"""
    k_mae = {}

    for k in range(1, max_k+1):

        # PCA Projection to k-dimensional
        pca = PCA(n_components=k)
        principal_comp = pca.fit_transform(X)

        mae = []
        # run cross validation
        kfold = KFold(n_splits=5, random_state=1, shuffle=True)
        for train_index, test_index in kfold.split(principal_comp):
            X_train, X_test = principal_comp[train_index], principal_comp[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            mae.append(mean_absolute_error(y_test, y_test_pred))

        k_mae[k] = np.mean(mae)

    return k_mae


def automatic_feature_selection(model, df, features, target, score):
    # y and original X 
    X = df[features]
    y = df[target].values
    
    # split the data into train and val
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    
    # fit a baseline model
    model.fit(X_train[features], y_train)

    # get a baseline validation metric
    baseline = score(y_valid, model.predict(X_valid[features]))
    print('Baseline validation metric:', baseline)
    print()
    
    drop = True
    while drop:
        print(f'k={len(features)}')
        # compute feature importance by Spearman rank correlation coefficient
        imp = spearman_coeff(df, features, target)
        sorted_imp = dict(sorted(imp.items(), key=lambda item: item[1], reverse=True))

        # drop the lowest importance feature
        lowest_imp_feature = list(sorted_imp.keys())[-1]
        features.remove(lowest_imp_feature)
        print('Drop the lowest importance feature: ', lowest_imp_feature)

        # retrain the model
        model.fit(X_train[features], y_train)

        # re-computing the validation metric
        new_score = score(y_valid, model.predict(X_valid[features]))
        print('The validation metric of re-trained model:', new_score)
        if new_score < baseline: # we have dropped one too many features
            features.append(lowest_imp_feature)
            drop = False
            print('The validation metric is worse, stop dropping.')
        print()
    print(f'number of features selected: {len(features)}')
    return features


def importances_w_errorbars(df, n, features, target, model, score):
    features_imps = defaultdict(list)

    # bootstrapping and record importance
    for _ in range(n):
        df_bootstrapping = resample(df, replace=True)

        # y and original X 
        X = df_bootstrapping[features]
        y = df_bootstrapping[target].values

        # split the data into train and val
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

        model.fit(X_train, y_train)
        feature_imp = permutation_importances(model, X_valid, y_valid, score)
        for feature, imp in feature_imp.items():
            features_imps[feature].append(imp)

    # get the mean and standard deviation of each feature importance
    feature_imp_mean_std = defaultdict(tuple)
    for feature, imps in features_imps.items():
        imp_mean = np.mean(imps)
        imp_std = np.std(imps)
        feature_imp_mean_std[feature] = (imp_mean, imp_std)
        
    # visualize
    n = len(feature_imp_mean_std)
    sorted_imp_mean_std = dict(sorted(feature_imp_mean_std.items(), key=lambda item: item[1], reverse=True))  
    features = list(sorted_imp_mean_std.keys())
    means = []
    stds = []
    for mean, std in list(sorted_imp_mean_std.values()):
        means.append(mean)
        stds.append(std)
        
    fig, ax = plt.subplots(figsize=(8, 6))
    # set grindlines
    ax.yaxis.grid(color='grey', linestyle='--', linewidth=0.4)

    for i in range(n):
        mid = ax.scatter(means[i], n-i, c='blue', s=30)
        lower = ax.scatter(means[i]-2*stds[i], n-i, c='green', s=30, marker='|')
        upper = ax.scatter(means[i]+2*stds[i], n-i, c='green', s=30, marker='|')
        ax.hlines(n-i, means[i]-2*stds[i], means[i]+2*stds[i], color='green')

    # hide the spines
    ax.spines['left'].set_color('gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # set ticklabels of y-axis
    ax.set_yticks(np.arange(1, n+1))
    ax.set_yticklabels(reversed(features), fontsize=12, backgroundcolor='w')

    # hide the ticks
    ax.tick_params(axis='x', color='white')
    ax.tick_params(axis='y', color='white')
    
    return ax
    

def single_feature_importance(col, X, y, score, model):
    # split the data into train and val
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

    # choose the model and metric
    model.fit(X_train, y_train)

    # compute the feature importance for col using permutation importance
    baseline = score(y_valid, model.predict(X_valid))

    save = X_valid[col].copy()
    X_valid[col] = np.random.permutation(X_valid[col])
    m = score(y_valid, model.predict(X_valid))
    X_valid[col] = save.values
    imp_col = (baseline - m)
    return imp_col


def importance_w_empirical_p(X, y, col, model, score, n):
    # true feature importance computed as a baseline
    imp_col_baseline = single_feature_importance(col, X, y, score, model)
    
    # store importances
    col_imps = []
    
    for _ in range(n):
        # shuffle the target variable y
        shuffled_y = np.random.permutation(y)

        # compute the feature importance for col using permutation importance
        imp_col = single_feature_importance(col, X, shuffled_y, score, model)
        col_imps.append(imp_col)
        
    return col_imps, imp_col_baseline


