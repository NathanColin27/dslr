from Data.data import Data
import numpy as np
import matplotlib.pyplot as plt


def pair_plot(data):
    plt.figure("Pair plot",figsize=(35, 15))
    plt.subplots_adjust(hspace=0.9)
    
    courses = [f for f in list(data.df.columns) if np.issubdtype(data.df[f].dtype, np.number)][1:]
    n = 0
    for index_x, course_x in enumerate(courses):
        for index_y, course_y in enumerate(courses):
            ax = plt.subplot(13, 13, n + 1)
            n += 1
            
            if course_x == course_y:
                for house, color in zip(data.houses, data.colors):
                    marks = data.df.loc[data.df['Hogwarts House'] == house][course_x]
                    ax.hist(marks, color=color, alpha=0.5)
                    if index_y == 0:
                       ax.set_ylabel(course_x,fontsize=8)
                    if index_x == len(courses) - 1:
                        ax.set_xlabel(course_y,fontsize=8)
                    plt.xticks(fontsize=5)
                    plt.yticks(fontsize=5)


            else:
                for house, color in zip(data.houses, data.colors):
                    x = data.df.loc[data.df['Hogwarts House'] == house][course_x]
                    y = data.df.loc[data.df['Hogwarts House'] == house][course_y]
                    plt.scatter(x, y, color=color, alpha=0.5, s=2)
                    if index_y == 0:
                       plt.ylabel(course_x,fontsize=8)
                    if index_x == len(courses) - 1:
                        plt.xlabel(course_y,fontsize=8)
                        
                        
                    plt.xticks(fontsize=5)
                    plt.yticks(fontsize=5)
    plt.show()
if __name__ == '__main__':
    
    data = Data()
    pair_plot(data)