import argparse


def func():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train some (or all) models on the training set provided')
    parser.add_argument("train_set", help="The path to the training set csv file")
    parser.add_argument('-o', '--output', default='models', help='folder path to save trained model(s) into')
    parser.add_argument('models', nargs='+',
                        choices=['KNN', 'MLP', 'DT', 'RF', 'SVM', 'LR', 'GPC', 'GPR', 'NB', 'DTree', 'DecisionTree',
                                 'RandomForest', 'Logistic', 'LogisticRegression', 'Gaussian', 'NeuralNetwork', 'all'],
                        help='One or more algorithms to be trained')
    parser.add_argument('-n', '--neighbors', nargs='*', type=int, default=5,
                        help='Number of neighbors for KNN')
    parser.add_argument('-lr', '--learning-rate', nargs='?', type=float, default=0.001,
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
