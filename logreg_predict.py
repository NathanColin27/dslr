import argparse
import sys
import pandas as pd
import numpy as np
from data.data import Data
from LR_models import LogisticRegression, MultipleLogisticRegression
import data.data_tools as dt
import json


def main(sys_argv):
    Models = MultipleLogisticRegression()
    data = Data(datafile="datasets/dataset_test.csv")
    with open("weights.json", "r") as outfile:
        weights = json.load(outfile)
    print(weights)
    df = pd.read_csv("datasets/cleaned_test.csv")
    data.df = dt.clean_dataset(data)

if __name__ == '__main__':
    main(sys.argv)
