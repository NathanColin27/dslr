import argparse
import sys
import pandas as pd
import numpy as np
from data.data import Data
import data.data_tools as dt
from LR_models import MultipleLogisticRegression
from distutils.util import strtobool

def user_yes_no_query(question):
    sys.stdout.write('%s [y/n]\n' % question)
    while True:
        try:
            return strtobool(input().lower())
        except ValueError:
            sys.stdout.write('Please respond with \'y\' or \'n\'.\n')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--alpha", type=float,
                        help="learning rate of the logistic regression algorithm", default=0.0001)
    parser.add_argument(
        "-i", "--iterations", type=int, help="number of iterations of the logistic regression algorithm", default=5000)
    parser.add_argument("-d", "--data", help="dataset csv file input",
                        default="./datasets/dataset_train.csv")
    # parser.add_argument("-v", "--verbose", help="increase output verbosity",
    #                     action="store_true")
    args = parser.parse_args()

    if not 10 <= args.iterations <= 10000:
        print("Iterations must be a value between 10 and 10000")
        exit(1)
    return args


def main(sys_argv):
    args = parse_args()
    # Data cleaning
    data = Data(datafile=args.data)
    dt.clean_dataset(data)
    dt.normalize_set(data)
    dt.export_dataset(data.df, args.data)

    # train
    df = pd.read_csv('./datasets/clean/dataset_train.csv')

    X_train = np.array(df.drop(columns=data.houses))
    y_train = np.array(df[data.houses])

    Models = MultipleLogisticRegression()
    Models.fit(X_train, y_train, alpha=args.alpha, iterations=args.iterations)
    # Models.fit(X_train, y_train, alpha=args.alpha /
    #            10, iterations=args.iterations)


    # Prompt user to save parameters or not
    if (user_yes_no_query("Do you wish to save the parameters?")):
        Models.save_weights("data/parameters.csv")

    # with open("weights.json", "w") as outfile:
    #     json.dump(Models.save_weights(), outfile)


if __name__ == '__main__':
    main(sys.argv)
