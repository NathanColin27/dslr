import pandas as pd
import numpy as np
import math


class Data():
    def __init__(self, datafile="datasets/dataset_train.csv"):
        self.df = pd.read_csv(datafile)

        self.features = [f for f in list(
            self.df.columns) if np.issubdtype(self.df[f].dtype, np.number)]
        
        self.houses = ['Gryffindor', 'Slytherin', 'Hufflepuff','Ravenclaw' ]
        self.colors = ["red", "green", "yellow", "blue"]

    def __str__(self):
        return f"""
                data
                """

    def count(self, feature: str) -> int:
        """
        Count non-NA (NaN, NaT, None) cells for the requested feature.
        """
        return len(self.df[feature].dropna())

    def mean(self, feature: str) -> float:
        """
        Returns the mean for the values of the requested feature.
        """
        return (sum(self.df[feature].dropna()) / self.count(feature))

    def variance(self, feature: str) -> float:
        mean_val = self.mean(feature)
        numerator = 0
        for i in self.df[feature].dropna():
            numerator += (i - mean_val) ** 2
        return (numerator/(self.count(feature) - 1))

    def std(self, feature: str) -> float:
        """
            Returns sample standard deviation for the requested feature.
        """
        return math.sqrt(self.variance(feature))

    def min(self, feature: str) -> float:
        min = np.nan
        for i in self.df[feature].dropna():
            min = min if i > min else i
        return min
    
    def max(self,feature : str) -> float:
        max = np.nan
        for i in self.df[feature].dropna():
            max = max if i < max else i 
        return max

    def percentile(self, feature : str, percent : int) -> float:
        arr = np.array(self.df[feature].dropna())
        arr.sort()
        l = self.count(feature)/100
        return arr[int(percent * l)]
    
    def export_parameters():
        pass
    
    def import_parameters():
        pass