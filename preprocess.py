import argparse
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def func():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess and split data into training, testing and validation sets"
                                                 "This file handles all the feature engineering prior to training")
    parser.add_argument("dataset", help="The csv dataset to perform feature engineering on")
    parser.add_argument('-o', '--output', default='datasets',
                        help='folder path to save train, test and validations sets into')
    parser.add_argument('-c', '--columns', nargs='+', help='List of all columns names of the dataset')
    parser.add_argument('-f', '-x', '--features', nargs='+', help='Name of dataset columns to be handled as features')
    parser.add_argument('-lb', '-y', '--label', nargs='+', help='Name of the dataset column to be handled as label')
    parser.add_argument('-dn', '--dropna', nargs='*',
                        help="column names which if a row's value is NaN, it should be dropped")
    parser.add_argument('-n', '--normalize', nargs='*', help='labels whose values should be normalized in (0, 1)')
    args = parser.parse_args()
    if Path(args.dataset).is_file() and Path(args.dataset).suffix == '.csv':
        dataset = pd.read_csv(Path(args.dataset), header=0, names=args.columns,
                              usecols=args.features + args.label, encoding='utf-8')
        for col in args.dropna:
            dataset[col].dropna(inplace=True)
        scaler = MinMaxScaler(copy=False)
        dataset[args.normalize] = scaler.fit_transform(dataset[args.normalize])
