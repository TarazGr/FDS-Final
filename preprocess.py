import argparse
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess and split data into training and testing sets"
                                                 "This file handles all the feature engineering prior to training")
    parser.add_argument("dataset", help="The csv dataset to perform feature engineering on")
    parser.add_argument('-o', '--output', default='datasets',
                        help='folder path to save train and test sets into')
    parser.add_argument('-c', '--columns', nargs='+', help='List of all columns names of the dataset')
    parser.add_argument('-f', '-x', '--features', nargs='+', help='Name of dataset columns to be handled as features')
    parser.add_argument('-lb', '-y', '--label', nargs='+', help='Name of the dataset column to be handled as label')
    parser.add_argument('-dn', '--dropna', nargs='*',
                        help="column names which if a row's value is NaN, it should be dropped")
    parser.add_argument('-n', '--normalize', nargs='*', help='labels whose values should be normalized in (0, 1)')
    parser.add_argument('-s', '--substitute', nargs='*',
                        help='labels whose NaN values should be substituted with the values specified in --subvalues\n'
                             'values specified here should correspond respectively to the ones in --subvalues')
    parser.add_argument('-sv', '--subvalues', nargs='*',
                        help='values to be used for substitution of NaN for columns specified in --substitute\n'
                             'values specified here should correspond respectively to the ones in --substitute')
    args = parser.parse_args()
    if Path(args.dataset).is_file() and Path(args.dataset).suffix == '.csv':
        dataset = pd.read_csv(Path(args.dataset), header=0, names=args.columns,
                              usecols=args.features + args.label, encoding='utf-8')
        if args.dropna:
            dataset.dropna(subset=args.dropna, inplace=True)
        if args.substitute and args.subvalues:
            dataset.fillna(dict(zip(args.substitute, args.subvalues)), inplace=True)
        if args.normalize:
            dataset[args.normalize] = MinMaxScaler(copy=False).fit_transform(dataset[args.normalize])
        x_train, x_test, y_train, y_test = train_test_split(dataset[args.features], dataset[args.label], test_size=0.2)
        pd.concat([x_train, y_train], axis=1, copy=False).to_csv(Path(args.output, 'train_set.csv'),
                                                                 index=False, encoding='utf-8')
        pd.concat([x_test, y_test], axis=1, copy=False).to_csv(Path(args.output, 'test_set.csv'),
                                                               index=False, encoding='utf-8')
