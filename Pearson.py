import pandas as pd
import numpy as np
import InPut

def get_correlation_matrix():

    data=InPut.Exp_input()

    data_values = data.iloc[:, 2:].values
    data_values_transposed = data_values.T

    df = pd.DataFrame(data_values_transposed)

    correlation_matrix = df.corr(method='pearson')
    print(correlation_matrix.shape)
    return correlation_matrix
