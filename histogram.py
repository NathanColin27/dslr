from data.data import Data
import numpy as np
import matplotlib.pyplot as plt

def histogram(data):
    # removing index
    courses = data.features[1:]

    plt.figure(figsize=(30, 15))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle("Score distribution by class", fontsize=18, y=0.95)

    n = 0
    for course in courses:
        ax = plt.subplot(4, 4, n + 1)
        n += 1
        for house, color in zip(data.houses, data.colors):
            marks = data.df.loc[data.df['Hogwarts House'] == house][course]
            ax.hist(marks, color=color, alpha=0.5)
            ax.set_title(course)
    plt.show()


if __name__ == '__main__':
    data = Data()
    histogram(data)