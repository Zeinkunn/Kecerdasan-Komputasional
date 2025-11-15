import numpy as np

def step_function(x):
    """Fungsi step: 1 jika x >= 0, selain itu 0"""
    return 1 if x >= 0 else 0

def predict(input_data, weights, bias):
    """Menghitung prediksi perceptron"""
    weighted_sum = np.dot(input_data, weights) + bias
    return step_function(weighted_sum)

training_data = np.array([
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

weights = np.array([1, 1])
bias = -0.5

print("Implementasi Gerbang Logika: OR")
print("Bobot (Weights):", weights)
print("Bias:", bias)

print("\nHasil Prediksi untuk gerbang logika OR:\n")

for data in training_data:
    input_val = data[:2]
    expected_output = data[2]
    predicted_output = predict(input_val, weights, bias)
    print(f"Input: {input_val}, Output yang diharapkan: {expected_output}, Output diprediksi: {predicted_output}")