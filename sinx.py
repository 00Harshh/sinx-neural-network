import numpy as np
import matplotlib.pyplot as plt # type: ignore
import random
x = np.linspace(-2*np.pi, 2*np.pi, 100).reshape(100, 1)  #(100,1)
y_true = np.sin(x).reshape(-1, 1) 

# --- Initialize weights and biases
np.random.seed(0)
input_size = 1
hidden_size = 30
output_size = 1

w1 = np.random.randn(input_size, hidden_size) * 0.5 #(1, 5)
b1 = np.zeros((1, hidden_size))  #(1, 5)
w2 = np.random.randn(hidden_size, 1) * 0.5 #(5, 1)
b2 = np.zeros((1, output_size))    #(1, 1)
  
# --- activation cells
def tanh(x):
    return np.tanh(x)

def tanh_derivatie(x):
    return 1 - np.tanh(x) ** 2
# --- loss functions
def mse(y_true, y_pred):
    return np.mean((y_pred - y_true)**2)
# --- Training loop
lr = 0.02
epochs = 10000
 
for epoch in range(epochs):
    z1 = x@w1 + b1    #(100, 5)
    a1 = tanh(z1)      # (100, 5)
    z2 = a1 @ w2 + b2   
    y_pred = z2

    # --- loss
    Loss = mse(y_true, y_pred)

    # --- Backpropagation 
    dloss = 2 * (y_pred - y_true) / len(x)
    dw2 = a1.T @ dloss
    db2 = np.sum(dloss, axis=0, keepdims=True)

    da1 = dloss @ w2.T
    dz1 = da1 * tanh_derivatie(z1)

    dw1 = x.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)
    
    current_lr = lr
    if epoch > 3000:
        current_lr = lr * 0.5
    if epoch > 6000:
        current_lr = lr * 0.2
 
    # ---- Gradient Descent Update
    w1 -= current_lr * dw1
    w2 -= current_lr * dw2

    b2 -= current_lr * db2
    b1 -= current_lr * db1

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {Loss:.4f}, Learning rate: {current_lr}')

# --- Calculate final accuracy
Final_loss = mse(y_true, y_pred)
Max_loss = np.max(np.abs(y_pred - y_true))
print(f'\nFinal Loss: {Final_loss:.6f}\nMax Loss: {Max_loss:.6f}')


# --- Plot result
plt.figure(figsize=(10, 6))
plt.plot(x, y_true, label="True sin(x)", linewidth=2, color='blue')
plt.plot(x, y_pred, label="NN Prediction", linestyle="--", linewidth=2, color='red')
plt.legend()
plt.title("Neural Network approximating sin(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True, alpha=0.3)
plt.show()
# --- Show the error
plt.figure(figsize=(10, 4))
error = np.abs(y_pred - y_true)
plt.plot(x, error, color='orange', linewidth=2)
plt.title("Prediction Error")  
plt.xlabel("x")
plt.ylabel("Absolute Error")
plt.grid(True, alpha=0.3)
plt.show()