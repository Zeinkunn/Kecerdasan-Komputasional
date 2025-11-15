import numpy as np

def step_function(x):
    """Fungsi step: 1 jika x >= 0, selain itu 0"""
    return 1 if x >= 0 else 0

def predict(input_data, weights, bias):
    """Menghitung prediksi perceptron"""
    if not isinstance(input_data, np.ndarray):
        input_data = np.array(input_data)
        
    weighted_sum = np.dot(input_data, weights) + bias
    return step_function(weighted_sum)

training_data = np.array([
    [0, 1],
    [1, 0]
])

weights = np.array([-2]) # Hanya satu bobot
bias = 1

print("Implementasi Gerbang Logika: NOT")
print("Bobot (Weights):", weights)
print("Bias:", bias)

print("\nHasil Prediksi untuk gerbang logika NOT:\n")

for data in training_data:
    input_val_array = data[0:1] 
    input_val_scalar = data[0] # Untuk logging
    expected_output = data[1]
    
    predicted_output = predict(input_val_array, weights, bias)
    
    print(f"Input: {input_val_scalar}, Output yang diharapkan: {expected_output}, Output diprediksi: {predicted_output}")