import mlflow
import json
import numpy as np
logged_model = '/mlruns/12/1c2b9737c0204b0ca825811c35fb6c64/artifacts/s444409'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

with open(f'{logged_model}/input_example.json') as f:
    data = json.load(f)
    input_example = np.array([data['inputs'][0]], dtype=np.float32)

# Predict on a Pandas DataFrame.
import pandas as pd
print(f'Prediction: {loaded_model.predict(input_example)}')