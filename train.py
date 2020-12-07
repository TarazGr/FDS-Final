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
from sklearn.naive_bayes import BaseNB
import pandas as pd
import pickle


def train_knn(df, neighbors, output):
    pass


def train_neural_network(df, learning_rate, layers, iterations, tol, output):
    pass


def train_decision_tree(df, output):
    pass


def train_random_forest(df, n_trees, depth, output):
    pass


def train_svm(df, output):
    pass


def train_logistic_regression(df, iterations, tol, output):
    pass


def train_gaussian_classifier(df, iterations, output):
    pass


def train_naive_bayes(df, output):
    pass


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
            train_knn(train_set, args.neighbors, args.output)
        elif any(_ in ['MLP', 'NeuralNetwork', 'all'] for _ in args.models):
            train_neural_network(train_set, args.learning_rate, args.layers, args.iterations, args.tolerance,
                                 args.output)
        elif any(_ in ['DT', 'DTree', 'DecisionTree', 'all'] for _ in args.models):
            train_decision_tree(train_set, args.output)
        elif any(_ in ['RF', 'RandomForest', 'Forest', 'all'] for _ in args.models):
            train_random_forest(train_set, args.trees, args.depth, args.output)
        elif any(_ in ['SVM', 'all'] for _ in args.models):
            train_svm(train_set, args.output)
        elif any(_ in ['LR', 'Logistic', 'LogisticRegression', 'all'] for _ in args.models):
            train_logistic_regression(train_set, args.iterations, args.tolerance, args.output)
        elif any(_ in ['GPC', 'Gaussian', 'all'] for _ in args.models):
            train_gaussian_classifier(train_set, args.iterations, args.output)
        elif any(_ in ['NB', 'Bayes', 'NaiveBayes', 'all'] for _ in args.models):
            train_naive_bayes(train_set, args.output)
