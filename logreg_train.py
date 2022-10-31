import argparse
import json
import numpy as np
import pandas as pd
import sys
from data.data import Data
import data.data_tools as dt
from LR_models import MultipleLogisticRegression


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--alpha", type=float,
                        help="learning rate of the logistic regression algorithm", default=0.001)
    parser.add_argument(
        "-i", "--iterations", type=int, help="number of iterations of the logistic regression algorithm", default=50)
    parser.add_argument("-d", "--data", help="dataset csv file input",
                        default="./datasets/dataset_train.csv")
    # parser.add_argument("-v", "--verbose", help="increase output verbosity",
    #                     action="store_true")
    args = parser.parse_args()

    if not 10 <= args.iterations <= 100:
        print("Iterations must be a value between 10 and 100")
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
    Models.fit(X_train, y_train, alpha=args.alpha /
               10, iterations=args.iterations)

    with open("weights.json", "w") as outfile:
        json.dump(Models.save_weights(), outfile)
        print(f"Models weights saved under {outfile.name}")


if __name__ == '__main__':
    main(sys.argv)
