import argparse
import sys
import pandas as pd
import numpy as np
from data.data import Data
from LR_models import LogisticRegression, MultipleLogisticRegression
import data.data_tools as dt
import json

def main(sys_argv):
    df_test = pd.read_csv("datasets/clean/cleaned_test.csv", index_col = False)
    
    target_columns = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    X_test = np.array(df_test)

    Models = MultipleLogisticRegression()
    Models.load_weights("Model_parameters.json")    # load saved weights 
    predictions = Models.predict(X_test)

    # Export predictions (numpy array) to csv with name of houses
    df_export = pd.DataFrame(predictions, columns=['Hogwarts House'])
    for i in range(len(df_export)):
        idx = df_export['Hogwarts House'][i]
        df_export['Hogwarts House'][i] = target_columns[idx]

    print(df_export.head(10))

    # Save to csv
    df_export.to_csv('prediction.csv', index=False)


if __name__ == '__main__':
    main(sys.argv)
