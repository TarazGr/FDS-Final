import argparse
from pathlib import Path
import pickle
from sklearn.metrics import *
import pandas as pd


def compute_measures(df, measures, features, label):
    if any(_ in ['F1', 'all'] for _ in measures):
        pass
    elif any(_ in ['precision', 'all'] for _ in measures):
        pass
    elif any(_ in ['recall', 'all'] for _ in measures):
        pass
    elif any(_ in ['accuracy', 'all'] for _ in measures):
        pass
    elif any(_ in ['confusion', 'matrix', 'confusionmatrix', 'all'] for _ in measures):
        pass
    elif any(_ in ['roc', 'all'] for _ in measures):
        pass
    elif any(_ in ['report', 'all'] for _ in measures):
        pass
    elif any(_ in ['main', 'all'] for _ in measures):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the provided model(s) with the testing set in input')
    parser.add_argument('test-set', help='The path to the testing set csv file to conduct tests on')
    parser.add_argument('models', nargs='+', help='Path(s) to the models to test;\n'
                                                  'if a folder is provided, all models in the folder will be tested')
    parser.add_argument('measures', nargs='*',
                        choices=['F1', 'precision', 'recall', 'accuracy', 'confusion', 'roc', 'matrix',
                                 'confusionmatrix', 'report', 'main', 'all'],
                        default='all', help='measures to be provided as test results')
    parser.add_argument('-c', '--columns', nargs='+', help='List of all columns names of the dataset')
    parser.add_argument('-f', '-x', '--features', nargs='+', help='Name of dataset columns to be handled as features')
    parser.add_argument('-lb', '-y', '--label', nargs='+', help='Name of the dataset column to be handled as label')
    args = parser.parse_args()
    if Path(args.test_set).is_file() and Path(args.test_set).suffix == '.csv':
        test_set = pd.read_csv(Path(args.train_set), header=0, names=args.columns, usecols=args.features + args.label,
                               na_filter=False, encoding='utf-8')
        for m in args.models:
            if Path.is_dir(m):
                for model in Path(m).iterdir():
                    if model.is_file() and model.suffix in ['.model', '.pkl']:
                        compute_measures(test_set, args.measures, args.features, args.label)
            elif Path(m).suffix in ['.model', '.pkl']:
                compute_measures(test_set, args.measures, args.features, args.label)
