from Data.data import Data
import numpy as np
import matplotlib.pyplot as plt


def pair_plot(data):
    
    plt.figure(figsize=(30, 15))
    plt.subplots_adjust(hspace=0.8)
    plt.suptitle("Pair plot", fontsize=18, y=0.95)
    
    courses = [f for f in list(data.df.columns) if np.issubdtype(data.df[f].dtype, np.number)][1:]
    n = 0
    for index_x, course_x in enumerate(courses):
        for index_y, course_y in enumerate(courses):
            ax = plt.subplot(13, 13, n + 1)
            n += 1
            ax.set_title(course_y)
            if course_x == course_y:
                for house, color in zip(data.houses, data.colors):
                    marks = data.df.loc[data.df['Hogwarts House'] == house][course_x]
                    ax.hist(marks, color=color, alpha=0.5)

            else:
                for house, color in zip(data.houses, data.colors):
                    x = data.df.loc[data.df['Hogwarts House'] == house][course_x]
                    y = data.df.loc[data.df['Hogwarts House'] == house][course_y]
                    plt.scatter(x, y, color=color, alpha=0.5, s=2)
    plt.show()  
if __name__ == '__main__':
    data = Data()
    pair_plot(data)