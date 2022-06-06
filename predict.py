import json
import mlflow
import sys
import numpy as np

#input = sys.argv[1]

logged_model = 'mlruns/1/70439eb482b54d56b54b0ecc6f1ca96f/artifacts/s444409'
loaded_model = mlflow.pyfunc.load_model(logged_model)


with open('input_example.json') as f:
    data = json.load(f)
    input_example = np.array([data['inputs'][0]], dtype=np.float32)

print(f'Prediction: {loaded_model.predict(input_example)}')