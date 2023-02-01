from script.pipeline import Output, Dataset, PipelineComponent


@PipelineComponent
def get_test_data() -> Output[Dataset]:
    import numpy as np
    import pandas as pd
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=float)
    df = pd.DataFrame({'col1': xs, 'target': ys})
    out = Dataset()
    out.framework = 'pd'
    out.data = df
    return out
