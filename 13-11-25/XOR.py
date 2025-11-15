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

weights_or = np.array([1, 1])
bias_or = -0.5

weights_and = np.array([1, 1])
bias_and = -1.5

weights_not = np.array([-2])
bias_not = 1

training_data = np.array([
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])

print("Implementasi Gerbang Logika: XOR")
print("Metode: Multi-Layer Perceptron")
print("Struktur: (A OR B) AND (NOT (A AND B))\n")

print("Hasil Prediksi untuk gerbang logika XOR:\n")

for data in training_data:
    input_val = data[:2]
    expected_output = data[2]
    
    out_or = predict(input_val, weights_or, bias_or)
    
    out_and = predict(input_val, weights_and, bias_and)

    
    out_not_and = predict(np.array([out_and]), weights_not, bias_not)
    
    input_layer_2 = np.array([out_or, out_not_and])

    predicted_output = predict(input_layer_2, weights_and, bias_and)
    
    print(f"Input: {input_val}, Output yang diharapkan: {expected_output}, Output diprediksi: {predicted_output}")