import argparse
from pathlib import Path
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import pickle


def train_knn(df, neighbors, features, label, output):
    neigh = KNeighborsClassifier(n_neighbors=neighbors)
    neigh = neigh.fit(features, label)
    path = Path(output, "model_KNN.pkl")
    with open(path, 'wb') as f:
        pickle.dump(neigh, f)


def train_neural_network(df, learning_rate, layers, iterations, tol, features, label, output):
    model = MLPClassifier(hidden_layer_sizes=layers, learning_rate=learning_rate, max_iter=iterations, tol=tol)
    model = model.fit(features, label)
    path = Path(output, "model_neural_networks.pkl")
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def train_decision_tree(df, features, label, output):
    model = DecisionTreeClassifier()
    model = model.fit(features, label)
    path = Path(output, "model_decision_tree.pkl")
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def train_random_forest(df, n_trees, depth, features, label, output):
    model = RandomForestClassifier(n_estimators=n_trees, max_depth=depth)
    model = model.fit(features, label)
    path = Path(output, "model_random_forest.pkl")
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def train_svm(df, features, label, output):
    model = SVC(kernel="linear")
    model = model.fit(features, label)
    path = Path(output, "model_svm.pkl")
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def train_logistic_regression(df, iterations, tol, features, label, output):
    model = LogisticRegression(solver='liblinear', max_iter=iterations, tol=tol)
    model = model.fit(features, label)
    path = Path(output, "model_logistic_regression.pkl")
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def train_gaussian_classifier(df, iterations, features, label, output):
    model = GaussianProcessClassifier(max_iter_predict=iterations)
    model = model.fit(features, label)
    path = Path(output, "model_gaussian_classifier.pkl")
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def train_gaussian_naive_bayes(df, features, label, output):
    model = GaussianNB()
    model = model.fit(features, label)
    path = Path(output, "model_gaussian_naive.pkl")
    with open(path, 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train some (or all) models on the training set provided')
    parser.add_argument("train_set", help="The path to the training set csv file")
    parser.add_argument('-o', '--output', default='models', help='folder path to save trained model(s) into')
    parser.add_argument('models', nargs='*',
                        choices=['KNN', 'MLP', 'DT', 'RF', 'SVM', 'LR', 'GPC', 'NB', 'DTree', 'DecisionTree',
                                 'Neighbors', 'Forest', 'RandomForest', 'Logistic', 'LogisticRegression', 'Gaussian',
                                 'Bayes', 'NaiveBayes', 'NeuralNetwork', 'all'],
                        default='all', help='One or more algorithms to be trained')
    parser.add_argument('-c', '--columns', nargs='+', help='List of all columns names of the dataset')
    parser.add_argument('-f', '-x', '--features', nargs='+', help='Name of dataset columns to be handled as features')
    parser.add_argument('-lb', '-y', '--label', nargs='+', help='Name of the dataset column to be handled as label')
    parser.add_argument('-n', '--neighbors', nargs='*', type=int, default=5,
                        help='Number of neighbors for KNN')
    parser.add_argument('-lr', '--learning_rate', nargs='?', type=float, default=0.001,
                        help='Learning rate for Multilayer Perceptron')
    parser.add_argument('-i', '--iterations', nargs='?', type=int, default=200,
                        help='Maximum number iterations for MultilayerPerceptron, LogisticRegression and/or Gaussian')
    parser.add_argument('-tol', '--tolerance', type=float, default=1e-4,
                        help='Tolerance for MultilayerPerceptron and/or LogisticRegression training termination')
    parser.add_argument('-l', '--layers', nargs='*', type=int, default=100,
                        help='Number of neurons per layer for the Multilayer Perceptron')
    parser.add_argument('-t', '--trees', nargs='?', type=int, default=100,
                        help='Number of trees in the RandomForest')
    parser.add_argument('-d', '--depth', nargs='?', type=int, default=None,
                        help='Maximum depth for each tree in the RandomForest')
    args = parser.parse_args()
    if Path(args.train_set).is_file() and Path(args.test_set).suffix == '.csv':
        train_set = pd.read_csv(Path(args.train_set), header=0, names=args.columns, usecols=args.features + args.label,
                                na_filter=False, encoding='utf-8')
        if any(_ in ['KNN', 'Neighbors', 'all'] for _ in args.models):
            train_knn(train_set, args.neighbors, args.fatures, args.label, args.output)
        elif any(_ in ['MLP', 'NeuralNetwork', 'all'] for _ in args.models):
            train_neural_network(train_set, args.learning_rate, args.layers, args.iterations, args.tolerance,
                                 args.fatures, args.label, args.output)
        elif any(_ in ['DT', 'DTree', 'DecisionTree', 'all'] for _ in args.models):
            train_decision_tree(train_set, args.fatures, args.label, args.output)
        elif any(_ in ['RF', 'RandomForest', 'Forest', 'all'] for _ in args.models):
            train_random_forest(train_set, args.trees, args.depth, args.fatures, args.label, args.output)
        elif any(_ in ['SVM', 'all'] for _ in args.models):
            train_svm(train_set, args.fatures, args.label, args.output)
        elif any(_ in ['LR', 'Logistic', 'LogisticRegression', 'all'] for _ in args.models):
            train_logistic_regression(train_set, args.iterations, args.tolerance, args.fatures, args.label, args.output)
        elif any(_ in ['GPC', 'Gaussian', 'all'] for _ in args.models):
            train_gaussian_classifier(train_set, args.iterations, args.fatures, args.label, args.output)
        elif any(_ in ['NB', 'Bayes', 'NaiveBayes', 'all'] for _ in args.models):
            train_gaussian_naive_bayes(train_set, args.fatures, args.label, args.output)
