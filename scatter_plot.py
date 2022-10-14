from Data.data import Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = Data()
courses = data.features[1:]


def covariance(f1: str, f2: str):
    list1 = data.df[f1]
    list2 = data.df[f2]
    f1_mean = data.mean(f1)
    f2_mean = data.mean(f2)

    numerator = 0
    if len(list1) == len(list2):
        for i in range(len(list1)):
            if pd.notna(list1[i]) and pd.notna(list2[i]):
                numerator += (list1[i] - f1_mean) * (list2[i] - f2_mean)
        return numerator / (len(list1) - 1)
    else:
        print("Error")


def correlation_coefficient(f1: str, f2: str):
    numerator = covariance(f1, f2)
    denominator = data.std(f1) * data.std(f2)
    return abs(numerator/denominator)


def scatter_plot(data):
    _, ax = plt.subplots()

    for house, color in zip(data.houses, data.colors):
        x = data.df.loc[data.df['Hogwarts House'] == house]['Astronomy']
        y = data.df.loc[data.df['Hogwarts House'] == house]['Astronomy']
        plt.scatter(x, y, color=color, alpha=0.5)
    ax.set_xlabel('Astronomy')
    ax.set_ylabel('Defense Against the Dark Arts')
    plt.show()


if __name__ == '__main__':

    #  Uncomment this section to show data justification
    # 
    # correlation_scores = {}
    # for i in courses:
    #     for j in courses:
    #         if i != j:
    #             correlation_scores[str(i + '-' + j)
    #                             ] = correlation_coefficient(i, j)
    # print("Best correlation coefficient found between classes : ",
    # max(correlation_scores, key=correlation_scores.get))
    # print(correlation_scores['Astronomy-Defense Against the Dark Arts'])
    
    data = Data()
    scatter_plot(data)
