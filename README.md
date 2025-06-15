# Neural Network From Scratch

A simple neural network built from scratch using only NumPy to approximate the sin(x) function. This project demonstrates the core concepts of neural networks without using any machine learning frameworks.

## 🎯 Project Overview

This project implements a 2-layer neural network that learns to approximate the mathematical sine function through backpropagation and gradient descent. The network has never seen the actual sin(x) formula - it learns the pattern purely from input-output examples!

## 🧠 What It Does

- **Input**: x values from -2π to 2π
- **Output**: Approximated sin(x) values
- **Learning**: Uses backpropagation to minimize prediction error
- **Result**: Achieves 99%+ accuracy in approximating the sine wave

## 🛠️ Technologies Used

- **Python 3.x**
- **NumPy** - For matrix operations and mathematical computations
- **Matplotlib** - For data visualization

## 📊 Network Architecture

```
Input Layer (1 neuron) → Hidden Layer (30 neurons) → Output Layer (1 neuron)
```

- **Activation Function**: Hyperbolic tangent (tanh)
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Gradient Descent with learning rate scheduling
- **Training Data**: 100 points sampled from sin(x)

## 🚀 Key Features

- ✅ **Pure NumPy Implementation** - No ML frameworks used
- ✅ **From Scratch Backpropagation** - Manual gradient computation
- ✅ **Learning Rate Scheduling** - Adaptive learning rate for better convergence
- ✅ **Xavier Weight Initialization** - Better starting weights for faster learning
- ✅ **Visualization** - Plots showing learning progress and error analysis

## 📈 Results

The neural network successfully learns to approximate sin(x) with high accuracy:

- **Final Loss**: < 0.001
- **Max Error**: < 0.05
- **Training Time**: ~8000 epochs

## 🔧 Installation & Usage

### Prerequisites
```bash
pip install numpy matplotlib
```

### Running the Code
```bash
git clone https://github.com/yourusername/neural-network-from-scratch.git
cd neural-network-from-scratch
python sinx.py
```

### Expected Output
1. Training progress printed to console
2. Graph showing true sin(x) vs neural network prediction
3. Error analysis plot showing prediction accuracy
![graph1](https://github.com/user-attachments/assets/0cf6af26-cbda-4ba8-adcc-5dd5b1a9bb7c)
![graph2](https://github.com/user-attachments/assets/cf809afb-c7a0-4efc-89f7-a4a480871571)



## 📚 What I Learned

- **Forward Propagation**: How data flows through the network
- **Backpropagation**: How errors propagate backward to update weights
- **Gradient Descent**: How the network "learns" by minimizing error
- **Weight Initialization**: Why starting values matter for learning
- **Learning Rate Scheduling**: Balancing speed vs accuracy in learning

## 🔮 Next Steps

- [ ] Implement different activation functions (ReLU, Sigmoid)
- [ ] Add support for multiple hidden layers
- [ ] Experiment with different optimizers (Adam, RMSprop)
- [ ] Apply to more complex functions (cos, polynomial, exponential)
- [ ] Build Physics-Informed Neural Networks for differential equations

## 📁 Project Structure

```
neural-network-from-scratch/
│
├── sinx.py                 # Main neural network implementation
├── README.md              # Project documentation
└── requirements.txt       # Dependencies
```

## 🎓 Learning Resources

This project was inspired by understanding the mathematical foundations of neural networks:

- **Linear Algebra**: Matrix multiplication for layer computations
- **Calculus**: Chain rule for backpropagation
- **Optimization**: Gradient descent for parameter updates

## 🤝 Contributing

Feel free to fork this project and experiment with:
- Different network architectures
- Alternative activation functions
- Various optimization techniques
- New target functions to approximate



## 🔗 Connect

- **LinkedIn**: [www.linkedin.com/in/harsh-joshi-8b2219297]
- **Twitter**: [https://x.com/008Harshh]

---

*Built with curiosity and caffeine ☕ - Learning AI one matrix multiplication at a time!*
