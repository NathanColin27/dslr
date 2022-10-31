from os import lseek
from data.data import Data
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from functools import partial


def format_decimal(number, decimal):
    return format(number, f'.{decimal}f')


def prepare_data(data: Data):
    func_list = [data.count, data.mean, data.std, data.min, partial(data.percentile, percent=25),
                 partial(data.percentile, percent=50), partial(data.percentile, percent=75),  data.max]
    vals = []
    for func in func_list:
        vals.append(list(map(func, data.features)))
    # tranpose the 2d array and rounds to 6 decimals
    vals = [list(map(partial(format_decimal, decimal=6), y))
            for y in [list(x) for x in zip(*vals)]]
    # insert the label of each row
    vals.insert(0, ["count", "mean", "std", "min", "25%", "50%", "75%", "max"])
    data.vals = vals


def draw_table(data):

    headerColor = 'grey'
    rowEvenColor = 'lightgrey'
    rowOddColor = 'white'
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['', *data.features],
            line_color='darkslategray',
            fill_color=headerColor,
            align=['left', 'center'],
            font=dict(color='Black', size=12)
        ),
        cells=dict(
            values=data.vals,
            line_color='darkslategray',
            # 2-D list of colors for alternating rows
            fill_color=[[rowOddColor, rowEvenColor, rowOddColor, rowEvenColor,
                         rowOddColor, rowEvenColor, rowOddColor,  rowEvenColor]],
            align=['left', 'center'],
            font=dict(color='darkslategray', size=11)
        ))
    ])

    fig.show()


if __name__ == '__main__':
    data = Data()
    prepare_data(data)
    draw_table(data)
