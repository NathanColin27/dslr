import os
import pandas as pd


def clean_dataset(data):
    # We start by dropping columns that are most likely not relevant to our target value (from our intuition)
    columns_to_drop = ['Index', 'First Name',
                       'Last Name', 'Birthday', 'Best Hand']

    # The pair plot we saw before showed us that some variables have a histogram with almost
    # all similar traits within the same House.
    # 'Defense Against the Dark Arts' has a complete linearity with 'Astronomy',
    # and the histogram shows that they are almost identical (with the sign reversed). We'll drop Defence.
    columns_to_drop.extend(
        ['Arithmancy', 'Care of Magical Creatures', 'Defense Against the Dark Arts'])

    df_train = data.df.drop(columns=columns_to_drop)
    data.features = [f for f in data.features if f not in columns_to_drop]
    # Transform string values (Houses) to category

    df_train = pd.concat([df_train, pd.get_dummies(df_train['Hogwarts House'])], axis=1)
    df_train = df_train.drop(columns='Hogwarts House')

    # In order to preserve the whole dataset, we are going to replace the missing entries by their column average value
    df_train = df_train.fillna(df_train.mean())
    data.df = df_train


def export_dataset(df, name):
    name = os.path.basename(os.path.normpath(name))
    os.makedirs('./datasets/clean', exist_ok=True)
    df.to_csv(f'./datasets/clean/{name}', index=False)


def normalize_feature(x, mean, std):
    """
    x: np array of feature X
    """
    return (x - mean) / std


def normalize_set(data):
    for col in data.features:
        mean = data.mean(col)
        std = data.std(col)
        data.df[col] = normalize_feature(data.df[col], mean, std)
