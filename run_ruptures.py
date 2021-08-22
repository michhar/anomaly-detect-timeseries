"""
Example of ruptures Python packages for change point detection
https://github.com/deepcharles/ruptures
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ruptures as rpt
import matplotlib
import pandas as pd
import os
import numpy as np
from _config import Config
matplotlib.use('TkAgg')


if __name__ == '__main__':
    # Read input data from csv
    input_data = pd.read_csv(os.path.join('data', 'input_data.csv'),
                        header=0,
                        index_col=[0],
                        parse_dates=[0])

    print('Shape of input data = ', input_data.shape)

    # Detection with PELT
    algo = rpt.Pelt(model="rbf").fit(input_data)
    result = algo.predict(pen=100)

    print(','.join([str(input_data.index[i-1]) for i in result]))

    # Display
    rpt.display(input_data, result, figsize=(12, 6))
    plt.show()
