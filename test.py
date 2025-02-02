import argparse
import pickle
from pathlib import Path

import pandas as pd
import sklearn.metrics as metric


def compute_measures(path, df, measures, features, label):
    with open(path, 'rb') as f:
        mod = pickle.load(f)
    predictions = mod.predict(df[features])
    measures = [measures] if measures is str else measures
    data = {}
    if any(_ in ['F1', 'all'] for _ in measures):
        data.update({'F1': metric.f1_score(df[label].values.ravel(), predictions)})
    if any(_ in ['precision', 'all'] for _ in measures):
        data.update({'precision': metric.precision_score(df[label].values.ravel(), predictions)})
    if any(_ in ['recall', 'all'] for _ in measures):
        data.update({'recall': metric.recall_score(df[label].values.ravel(), predictions)})
    if any(_ in ['accuracy', 'all'] for _ in measures):
        data.update({'accuracy': metric.accuracy_score(df[label].values.ravel(), predictions)})
    if any(_ in ['confusion', 'matrix', 'confusionmatrix', 'all'] for _ in measures):
        data.update({'confusionmatrix': metric.confusion_matrix(df[label].values.ravel(), predictions)})
    if any(_ in ['roc', 'all'] for _ in measures):
        data.update({'roc': metric.roc_auc_score(df[label].values.ravel(), predictions)})
    if any(_ in ['report', 'all'] for _ in measures):
        data.update({'report': metric.classification_report(df[label].values.ravel(), predictions)})
    if any(_ in ['main', 'all'] for _ in measures):
        data.update({'main': metric.precision_recall_fscore_support(df[label].values.ravel(), predictions)})
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the provided model(s) with the testing set in input')
    parser.add_argument('test_set', help='The path to the testing set csv file to conduct tests on')
    parser.add_argument('models', nargs='+', help='Path(s) to the models to test;\n'
                                                  'if a folder is provided, all models in the folder will be tested')
    parser.add_argument('-m', '--measures', nargs='*',
                        choices=['F1', 'precision', 'recall', 'accuracy', 'confusion', 'roc', 'matrix',
                                 'confusionmatrix', 'report', 'main', 'all'],
                        default='all', help='measures to be provided as test results')
    parser.add_argument('-o', '--output', default='../', help='folder path to save metric csv into')
    parser.add_argument('-c', '--columns', nargs='+', help='List of all columns names of the dataset')
    parser.add_argument('-f', '-x', '--features', nargs='+', help='Name of dataset columns to be handled as features')
    parser.add_argument('-lb', '-y', '--label', nargs='+', help='Name of the dataset column to be handled as label')
    args = parser.parse_args()
    if Path(args.test_set).is_file() and Path(args.test_set).suffix == '.csv':
        test_set = pd.read_csv(Path(args.test_set), header=0, names=args.columns, usecols=args.features + args.label,
                               na_filter=False, encoding='utf-8')
        res = pd.DataFrame(columns=['model'] + args.measures)
        for m in args.models:
            if Path(m).is_dir():
                for model in Path(m).iterdir():
                    if model.is_file() and model.suffix in ['.model', '.pkl']:
                        metrics = compute_measures(model, test_set, args.measures, args.features, args.label)
                        metrics.update({'model': model.name.rstrip(model.suffix)})
                        res = res.append(metrics, ignore_index=True)
            elif Path(m).suffix in ['.model', '.pkl']:
                metrics = compute_measures(m, test_set, args.measures, args.features, args.label)
                metrics.update({'model': m.name.rstrip(m.suffix)})
                res = res.append(metrics, ignore_index=True)
        res.to_csv(Path(args.output, 'test_results.csv'), index=False, encoding='utf-8')
