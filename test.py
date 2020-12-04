import argparse


def func():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the provided model(s) with the testing set in input')
    parser.add_argument('test-set', help='The path to the testing set csv file to conduct tests on')
    parser.add_argument('models', nargs='+', help='Path(s) to the models to test; '
                                                  'if a folder is provided, all models in the folder will be tested')
    parser.add_argument('measures', nargs='+',
                        choices=['F1', 'precision', 'recall', 'accuracy', 'confusion', 'roc', 'confusion',
                                 'matrix', 'confusionmatrix', 'report', 'main', 'all'],
                        help='measures to be provided as test results')
    parser.parse_args()
