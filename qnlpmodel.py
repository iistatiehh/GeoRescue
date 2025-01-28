import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from qiskit_aer import Aer
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_machine_learning.optimizers import COBYLA,ADAM,SPSA

def model_load(path=[str]):
  Qubits = 5
  featureMap = ZZFeatureMap(Qubits, reps=1, entanglement='linear').decompose()
  estimator = Estimator()
  optimizer = SPSA(maxiter=100)
  QNN = EstimatorQNN(
    circuit = featureMap,
    estimator = estimator,
    input_params = featureMap.parameters
    )
  classifier = NeuralNetworkClassifier(
    neural_network = QNN,
    optimizer = optimizer,
    initial_point=None,
    )
  return classifier.load(path)

def load_vectorier(path=[str]):
  dataset = pd.read_csv(path)
  vectorizer = TfidfVectorizer(max_features=5)
  vectorizer.fit(dataset['cleanedText'])
  return vectorizer

model = model_load("classifer.cfl")
vectorizer = load_vectorier("data.csv")

print(model.predict(vectorizer.transform(["fire in los angeles"]))[0][0]==1)