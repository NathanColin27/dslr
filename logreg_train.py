import argparse
import sys
import pandas as pd
import numpy as np
from data.data import Data
from LR_models import LogisticRegression, MultipleLogisticRegression
import os
import json



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


def normalize(x, mean, std):
    """
    x: np array of feature X
    """
    return (x - mean) / std


def clean_dataset(data):
    # We start by dropping columns that are most likely not relevant to our target value (from our intuition)
    columns_to_drop = ['Index', 'First Name',
                       'Last Name', 'Birthday', 'Best Hand']

    # The pair plot we saw before showed us that some variables have a histogram with almost
    # all similar traits within the same House. We will drop them.
    columns_to_drop.extend(
        ['Arithmancy', 'Care of Magical Creatures', 'Potions'])
    df_train = data.df.drop(columns=columns_to_drop)
    data.features = [f for f in data.features if f not in columns_to_drop]
    # Transform string values (Houses) to category
    df_train = pd.concat(
        [df_train, pd.get_dummies(df_train['Hogwarts House'])], axis=1)
    df_train = df_train.drop(columns='Hogwarts House')

    # In order to preserve the whole dataset, we are going to replace the missing entries by their column average value
    df_train = df_train.fillna(df_train.mean())

    return df_train.fillna(df_train.mean())


def export_dataset(df, name):
    name = os.path.basename(os.path.normpath(name))
    os.makedirs('./datasets/clean', exist_ok=True)
    df.to_csv(f'./datasets/clean/{name}', index=False)


def main(sys_argv):
    args = parse_args()

    data = Data(datafile=args.data)
    data.df = clean_dataset(data)
    for col in data.features:
        mean = data.mean(col)
        std = data.std(col)
        data.df[col] = normalize(data.df[col], mean, std)
    export_dataset(data.df, args.data)
    df = pd.read_csv('./datasets/clean/dataset_train.csv')
    target_columns = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    X_train = np.array(df.drop(columns=target_columns))
    y_train = np.array(df[target_columns])
    models = MultipleLogisticRegression()
    models.fit(X_train, y_train, alpha=args.alpha, iterations=args.iterations)
    models.fit(X_train, y_train, alpha=args.alpha /
               10, iterations=args.iterations)

    with open("weights.json", "w") as outfile:
        json.dump(models.save_weights(), outfile)


if __name__ == '__main__':
    main(sys.argv)
