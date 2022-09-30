import pandas as pd
import numpy as np
from collections import defaultdict as dd
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, normalized_mutual_info_score, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import csv
import re
import math
import scipy
import matplotlib.pyplot as plt

# Columns that will not be considered for feature selection
FILTERED_COLS = ['d_spell', 'f_spell', 'champion', 'side', 'damage_turrets']
# Columns to be evaluated using MI after discretisation
COLS = ['kills', 'assists', 'deaths', 'kda', 'level', 'time_cc', 'vision_score', 'gold_earned', 'turret_kills',
        'damage_taken', 'damage_objectives', 'damage_building', 'damage_total']
# Columns to be evaluated using MI without discretisation
COLS_DISC = ['minions_killed', 'region']
# Features that have been selected to use for supervised learning
X_COLS = ['damage_building', 'damage_taken', 'vision_score', 'minions_killed', 'assists']
# Target Variable
Y_COL = 'role'
# Performance metrics used for model evaluation
METRICS = ['Accuracy', 'Recall', 'Precision', 'F1']
# K for K-fold Cross Validation
K_FOLD = 5
# Line break
LINE = '======================================================================'


def kda_filler(row):
    '''
    Helper function that is used to compute 'kda' for a row if the row has
    'kills', 'deaths', and 'assists', but no 'kda'
    '''
    if (np.isnan(row['kda']) and not np.isnan(row['kills']) and not np.isnan(row['deaths']) and not np.isnan(
            row['assists'])):
        if (row['deaths'] == 0):
            return (row['kills'] + row['assists'])
        return (row['kills'] + row['assists']) / row['deaths']
    else:
        return row['kda']


def remove_outliers(df):
    '''
    Takes a dataframe and removes outliers from it, returning the filtered
    dataframe. Adapted from the top answer of
    https://stackoverflow.com/questions/68348516/automating-removing-outliers-from-a-pandas-dataframe-using-iqr-as-the-parameter
    written by username: "filiabel" on 14/07/2021. Accessed on 14/5/2022.
    '''

    # Calculate quantiles and IQR
    Q1 = df[X_COLS].quantile(0.25)
    Q3 = df[X_COLS].quantile(0.75)
    IQR = Q3 - Q1

    # Define the condition on which to remove outliers
    condition = ~((df[X_COLS] < (Q1 - 1.5 * IQR)) | (df[X_COLS] > (Q3 + 1.5 * IQR))).any(axis=1)

    # Filter our dataframe to remove outliers
    filtered_df = df[condition]

    return filtered_df


def wrangle(df_merged):
    # Replace values that have zero 'damage_turrets' and have non-zero 'turret_kills' or
    # that have zero 'turret_kills' and have non-zero 'damage_turrets' with NaN (invalid data)
    df_merged.loc[(df_merged['damage_turrets'] == 0) & (df_merged['turret_kills'] != 0), ['damage_turrets']] = np.NaN
    df_merged.loc[(df_merged['turret_kills'] == 0) & (df_merged['damage_turrets'] != 0), ['turret_kills']] = np.NaN

    # Fill 'damage_objectives' with value from 'damage_turrets' if 'damage_objectives' is empty
    df_merged['damage_objectives'] = df_merged.apply(
        lambda row: row['damage_turrets'] if np.isnan(row['damage_objectives']) else row['damage_objectives'], axis=1)

    # Fill 'kda' with kda derived from helper function `kda_filler`
    df_merged['kda'] = df_merged.apply(kda_filler, axis=1)

    # Remove unused features
    for col in FILTERED_COLS:
        del df_merged[col]

    # Split the data into training and testing sets
    train, test = train_test_split(df_merged, test_size=0.2)

    # For continuous variables, fill all empty values in `train` and `test`
    # with the numerical averages from `train`. For discrete variables, remove
    # rows with empty values.
    train = train.fillna(train.mean(numeric_only=True))
    train = train.dropna()
    test = test.fillna(train.mean(numeric_only=True))
    test = test.dropna()

    # Return train and test datasets
    return train, test


def discretisation(df, col):
    '''
    Takes dataframe `df` and column name `col`, then standardises the
    specified column in `df`. The standaridised column is then discretised
    into equal-frequency bins and inserted into `df`.
    '''

    df[col] = (df[col] - df[col].mean()) / df[col].std()
    category = pd.cut(df[col], bins=3, labels=[1, 2, 3])
    df.insert(0, f'{col}_discretised', category)
    return df


def mutual_info(df):
    '''
    Compute the Normalized Mutual Information scores for each feature against
    'role'. Outputs a barchart to visualise NMI scores, for feature selection.
    '''

    # Stores Mutual Information scores of features against 'role'
    role_MI = []

    # Compute Normalized Mutual Information scores for continuous variables
    for col in COLS:
        role_MI.append(normalized_mutual_info_score(df['role'], df[f'{col}_discretised'], average_method='arithmetic'))

    # Compute NMI scores for discrete variables
    for col in COLS_DISC:
        role_MI.append(normalized_mutual_info_score(df['role'], df[col], average_method='arithmetic'))

    data = {
        'column': COLS + COLS_DISC,
        'role_MI': role_MI
    }

    df_Mi = pd.DataFrame(data)

    # Plot a barchart of NMI scores for all features in ascending order
    df_sorted = df_Mi.sort_values(by="role_MI")
    fig, ax = plt.subplots()

    x_pos = np.arange(len(df_sorted['role_MI']))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Normalized Mutual Information")
    ax.set_title("Normalized Mutual Information between Features and Role")
    ax.bar(df_sorted["column"], df_sorted["role_MI"])
    ax.set_xticks(x_pos)

    plt.xticks(rotation=45, ha='right')

    plt.savefig('NMI.png', dpi=300, bbox_inches='tight')
    plt.close()
    return df_Mi


def find_k_knn(df, plot=True):
    '''
    Perform 5-fold Cross Validation to tune the hyperparameter 'k' for k-NN
    algorithm. If plot=True, plots a line graph to compare the performance
    metrics of each k value. The optimal k value is returned and used for
    model evaluation and final testing.
    '''

    # create our X and y features for KNN
    X = df[X_COLS]
    y = df[Y_COL]

    kf_CV = KFold(n_splits=K_FOLD, random_state=None)

    # Stores a dictionary for each k value, which contains performance metric results
    results_dict = dd(dict)

    for k_knn in range(51, 110, 2):
        accuracies = []
        recalls = []
        precisions = []
        f1s = []
        for train_index, test_index in kf_CV.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Retrieve performance metrics for k-NN algorithm for this k
            knn_performance = k_nearest_neighbour(X_train, X_test, y_train, y_test, k_knn, plot=False)
            accuracies.append(knn_performance[0])
            recalls.append(knn_performance[1])
            precisions.append(knn_performance[2])
            f1s.append(knn_performance[3])

        results_dict[k_knn] = {'Accuracy': round(np.mean(accuracies), 5),
                               'Recall': round(np.mean(recalls), 5),
                               'Precision': round(np.mean(precisions), 5),
                               'F1': round(np.mean(f1s), 5)}

    if plot:
        df = pd.DataFrame.from_dict({k: dict(v) for k, v in results_dict.items()}, orient='index')

        plt.figure()

        df.plot(y=['Accuracy', 'Recall', 'Precision', 'F1'],
                title='Average Performance Metrics from 5-fold Cross Validation for k values')
        plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
        plt.xlabel('K')
        plt.ylabel('Average Score')
        plt.savefig("Avg Scores vs K Value", bbox_inches='tight')
        plt.close()

        print("Tuned hyperparameter k for k-NN: ")
        print("Maximum Accuracy:", df["Accuracy"].max(), "at k =", df["Accuracy"].idxmax())
        print("Maximum Recall:", df["Recall"].max(), "at k =", df["Recall"].idxmax())
        print("Maximum Precision:", df["Precision"].max(), "at k =", df["Precision"].idxmax())
        print("Maximum F1:", df["F1"].max(), "at k =", df["F1"].idxmax())
        print("Use k = ", df["F1"].idxmax())
        print(LINE)

    return df["F1"].idxmax()


def kfold_cv(df, k):
    '''
    Performs K-fold Cross Validation to compare k-NN and Decision Tree models.
    The scores for each performance metric for each fold of the two algorithms
    are plotted with barcharts for evaluation. The model which is more suited
    to the dataset will be chosen as the final algorithm.
    '''

    # create our X and y features for KNN
    X = df[X_COLS]
    y = df[Y_COL]

    kf_CV = KFold(n_splits=K_FOLD, random_state=None)

    # Store the results from each fold for each performance metric
    knn_results = dd(list)
    dt_results = dd(list)

    # Create X and y features for decision tree
    X_tree = OrdinalEncoder().fit_transform(df[[f'{col}_discretised' for col in COLS]])
    y_tree = OrdinalEncoder().fit_transform(df[[Y_COL]])

    for train_index, test_index in kf_CV.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train_tree, X_test_tree = X_tree[train_index], X_tree[test_index]
        y_train_tree, y_test_tree = y_tree[train_index], y_tree[test_index]

        # Retrieve performance metrics for k-NN algorithm
        knn_performance = k_nearest_neighbour(X_train, X_test, y_train, y_test, k, plot=False)

        # Training for decision tree
        dt = DecisionTreeClassifier(criterion='entropy', max_depth=4)
        dt.fit(X_train_tree, y_train_tree)

        knn_results['Accuracy'].append(knn_performance[0])
        knn_results['Recall'].append(knn_performance[1])
        knn_results['Precision'].append(knn_performance[2])
        knn_results['F1'].append(knn_performance[3])

        # Perform predictions for decision tree
        y_pred_tree = dt.predict(X_test_tree)
        dt_results['Accuracy'].append(dt.score(X_test_tree, y_test_tree))
        dt_results['Recall'].append(recall_score(y_test_tree, y_pred_tree))
        dt_results['Precision'].append(precision_score(y_test_tree, y_pred_tree))
        dt_results['F1'].append(f1_score(y_test_tree, y_pred_tree))

    print('KNN K-fold CV Metrics: ')
    for metric, results in knn_results.items():
        print(metric, ': ', [round(result, 5) for result in results], 'average: ', round(np.mean(results), 5))

    print('DT K-fold CV Metrics: ')
    for metric, results in dt_results.items():
        print(metric, ': ', [round(result, 5) for result in results], 'average: ', round(np.mean(results), 5))

    print(LINE)

    # Plot barcharts for each performance metric to compare k-NN against Decision Trees
    for metric in METRICS:
        fig, axs = plt.subplots(2)
        fig.suptitle(metric + 'Scores of K-NN against Decision Tree in 5 Folds')
        fig.set_figheight(12)
        fig.set_figwidth(15)

        for ax in fig.axes:
            ax.set_ylabel(metric)
            ax.set_ylim(0.0, 1.0)
            plt.sca(ax)
            plt.xticks(rotation=45)

        fig.axes[0].set_xlabel("K-Nearest Neighbour")
        fig.axes[1].set_xlabel("Decision Tree")
        axs[0].bar(['1st Fold', '2nd Fold', '3rd Fold', '4th Fold', '5th Fold'], knn_results[metric])
        axs[1].bar(['1st Fold', '2nd Fold', '3rd Fold', '4th Fold', '5th Fold'], dt_results[metric])

        plt.savefig(f'{metric}.png')
        plt.close()


def k_nearest_neighbour(x_train, x_test, y_train, y_test, k, plot=True):
    '''
    Implementation of k-NN analysis. Standardises the data, then fits the
    model and performs predictions to test. Performance metrics are returned
    as a tuple. Produces a confusion matrix if `plot` == True.
    '''

    # Standardise the data
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit to the train dataset
    knn.fit(x_train, y_train)

    # Perform predictions
    y_pred = knn.predict(x_test)

    accuracy = round(accuracy_score(y_test, y_pred), 5)
    recall = round(recall_score(y_test, y_pred, pos_label="Other"), 5)
    precision = round(precision_score(y_test, y_pred, pos_label="Other"), 5)
    f1 = round(f1_score(y_test, y_pred, pos_label="Other"), 5)

    if plot:
        print("Performed KNN on final testing set...")
        print('Accuracy: ', accuracy)
        print('Recall:', recall)
        print('Precision:', precision)
        print('F1:', f1)
        print(LINE)

        # Output a confusion matrix to visualise performance
        cm = confusion_matrix(y_test, y_pred, labels=['Other', 'TopLane_Jungle'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Other', 'TopLane_Jungle'])

        disp.plot()

        plt.title("Confusion Matrix")
        plt.savefig('cm_roles', bbox_inches='tight')
        plt.close()

    return accuracy, recall, precision, f1


def dec_tree(train, test):
    '''
    Implementation of Decision Tree analysis. Encodes the data, then fits the
    model and produces a plotted decision tree.
    '''

    # Encode the training data
    X = OrdinalEncoder().fit_transform(train[[f'{col}_discretised' for col in COLS]])
    y = OrdinalEncoder().fit_transform(train[['role']])

    dt = DecisionTreeClassifier(criterion='entropy', max_depth=4)

    # Fit to the train dataset
    dt.fit(X, y)

    # Plot the Decision Tree
    fig, ax = plt.subplots()
    fig.set_figheight(25)
    fig.set_figwidth(55)
    plot_tree(dt, feature_names=COLS, class_names=['Other', 'TopLane_Jungle'], filled=True, fontsize=18)

    plt.title("Decision Tree Classifier - Roles")
    plt.savefig("dt5_roles", bbox_inches='tight', dpi=100)
    plt.close()


def main():
    # Aggregate data from all regions
    data_dict = {
        "KR_df": pd.read_csv('/course/data/a2/games/KRmatch.csv'),
        "EU_df": pd.read_csv('/course/data/a2/games/EUmatch.csv'),
        "NA_df": pd.read_csv('/course/data/a2/games/NAmatch.csv')
    }

    # Add a column into each dataframe to indicate 'region'
    for region, df in data_dict.items():
        reg = [re.sub(r'_df', '', region) for i in range(len(df))]
        df["region"] = reg

    # Merge the dataframes for all regions
    df_merged = pd.concat(data_dict.values())

    # Perform data wrangling tasks, returning the preprocessed training and
    # testing datasets
    train, test = wrangle(df_merged)
    train['minions_killed'].replace(['Many', 'Few'],
                                    [1, 0], inplace=True)
    test['minions_killed'].replace(['Many', 'Few'],
                                   [1, 0], inplace=True)

    if (input("Remove outliers? Enter Y or N\n") == 'Y'):
        train = remove_outliers(train)
        print('Outliers removed.')
    else:
        print('Outliers not removed.')

    print(LINE)

    # Make a copy of the training dataset which contains discretised features
    train_disc = train.copy()

    for col in COLS:
        discretisation(train_disc, col)

    # Perform feature selection with the discretised training dataset
    print('Normalized Mutual Info scores between feature and role:')
    print(mutual_info(train_disc))
    print(LINE)

    # Perform K-fold Cross Validation for hyperparameter tuning
    k = find_k_knn(train_disc, plot=True)

    # Perform K-fold Cross Validation for model evaluation and selection
    kfold_cv(train_disc, K_FOLD)

    # k-NN was chosen as the final algorithm, fit with the entire training set
    # and test with the independent testing set!
    x_train = train[X_COLS]
    y_train = train[Y_COL]
    x_test = test[X_COLS]
    y_test = test[Y_COL]
    k_nearest_neighbour(x_train, x_test, y_train, y_test, k, plot=True)

    dec_tree(train_disc, test)


main()