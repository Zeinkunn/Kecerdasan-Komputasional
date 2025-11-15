import numpy as np

def step_fungtin(x):
    return 1 if x >= 0 else 0

training_data = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1]
])

weights = np.array([1, 1])
bias = -1.5

print("Bobot (Weights):", weights)
print("Bias:", bias)

def predict(input, weights, bias):
    weights_sum = np.dot(input, weights) + bias
    return step_fungtin(weights_sum)

print("\nHasil Prediksi untuk gerbang logika AND:\n")


for data in training_data:
    input = data[:2]
    expected_output = data[2]
    predicted_output = predict(input, weights, bias)
    print(f"Input: {input}, Output yang diharapkan: {predicted_output}, Output diprediksi: {expected_output}")
    
test_input = np.array([1, 1])
predicted_test_output = predict(test_input, weights, bias)
print(f"\nPrediksi untuk input {test_input}: Hasil prediksi adalah {predicted_test_output}")