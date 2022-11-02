import argparse
import sys
import pandas as pd
import numpy as np
from data.data import Data
from LR_models import LogisticRegression, MultipleLogisticRegression
import data.data_tools as dt
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="dataset csv file input",
                        default="./datasets/dataset_test.csv")
    # parser.add_argument("-v", "--verbose", help="increase output verbosity",
    #                     action="store_true")
    args = parser.parse_args()
    return args

def normalize_set(data_test, data_train):
    for col in data_test.features:
        mean = data_train.mean(col)
        std = data_train.std(col)
        data_test.df[col] = dt.normalize_feature(data_test.df[col], mean, std)

def clean_dataset(data):
    # We start by dropping columns that are most likely not relevant to our target value (from our intuition)
    columns_to_drop = ['Index', 'First Name',
                       'Last Name', 'Birthday', 'Best Hand', 'Hogwarts House']

    # The pair plot we saw before showed us that some variables have a histogram with almost
    # all similar traits within the same House.
    # 'Defense Against the Dark Arts' has a complete linearity with 'Astronomy',
    # and the histogram shows that they are almost identical (with the sign reversed). We'll drop Defence.
    columns_to_drop.extend(
        ['Arithmancy', 'Care of Magical Creatures', 'Defense Against the Dark Arts'])

    df_train = data.df.drop(columns=columns_to_drop)
    data.features = [f for f in data.features if f not in columns_to_drop]

    
    # In order to preserve the whole dataset, we are going to replace the missing entries by their column average value
    df_train = df_train.fillna(df_train.mean())
    data.df = df_train

def main(sys_argv):
    args = parse_args()
    df_train = Data(datafile="./datasets/dataset_train.csv")
    dt.clean_dataset(df_train)

    df_test = Data(datafile=args.data)

    clean_dataset(df_test)
    # using mean and std from training set to normalize the 
    normalize_set(df_test, df_train)
    target_columns = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    X_test = np.array(df_test.df)

    
    Models = MultipleLogisticRegression()
    Models.load_weights("Model_weights.json")    # load saved weights 
    predictions = Models.predict(X_test)
    
    # Export predictions (numpy array) to csv with name of houses + index column
    df_export = pd.DataFrame(predictions, columns=['Hogwarts House'])
    df_export = df_export.reset_index()
    df_export = df_export.rename(columns={'index': 'Index'})
    for i in range(len(df_export)):
        idx = df_export['Hogwarts House'][i]
        df_export['Hogwarts House'][i] = target_columns[idx]

    print(df_export.head(10))

    # Save to csv
    df_export.to_csv('houses.csv', index=False)


if __name__ == '__main__':
    main(sys.argv)
 