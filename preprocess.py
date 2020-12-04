import argparse


def func():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess and split data into training, testing and validation sets"
                                                 "This file handles all the feature engineering prior to training")
    parser.add_argument("dataset", help="The csv dataset to perform feature engineering on")
    parser.add_argument('-c', '--columns', nargs='+', help='List of all columns names of the dataset')
    parser.add_argument('-f', '-x', '--features', nargs='+', help='Name of dataset columns to be handled as features')
    parser.add_argument('-l', '-y', '--label', help='Name of the dataset column to be handled as label')
    parser.parse_args()
