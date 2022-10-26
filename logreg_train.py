import argparse
import sys
import pandas as pd
from data import Data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--alpha", type=float,
                        help="learning rate of the logistic regression algorithm", default=0.001)

    parser.add_argument(
        "-i", "--iterations", type=int, help="number of iterations of the logistic regression algorithm", default=1000)
    parser.add_argument("-d", "--data", help="dataset csv file input",
                        default="./datasets/dataset_train.csv")
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    args = parser.parse_args()
    # check args validity
    return args

def normalize(x, mean, std):
    """
    x: np array of feature X
    """
    return (x - mean) / std

def clean_dataset(df):
    # We start by dropping columns that are most likely not relevant to our target value (from our intuition)
    columns_to_drop = ['Index', 'First Name',
                       'Last Name', 'Birthday', 'Best Hand']

    # The pair plot we saw before showed us that some variables have a histogram with almost
    # all similar traits within the same House. We will drop them.
    columns_to_drop.extend(
        ['Arithmancy', 'Care of Magical Creatures', 'Potions'])
    df_train = df.drop(columns=columns_to_drop)

    # Transform string values (Houses) to category
    df_train = pd.concat(
        [df_train, pd.get_dummies(df_train['Hogwarts House'])], axis=1)
    df_train = df_train.drop(columns='Hogwarts House')

    # In order to preserve the whole dataset, we are going to replace the missing entries by their column average value
    df_train = df_train.fillna(df_train.mean())

    return df_train.fillna(df_train.mean())


def main(sys_argv):
    args = parse_args()
    data = Data(pd.read_csv(args.data))
    df = clean_dataset(df)



if __name__ == '__main__':
    main(sys.argv)
