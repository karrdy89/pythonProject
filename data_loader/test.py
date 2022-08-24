import numpy as np
import pandas as pd

class Test:
    def __init__(self):
        pass

    def pipe(self):
        xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
        ys = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=float)
        df = pd.DataFrame({'col1': xs, 'col2': ys})
        return df