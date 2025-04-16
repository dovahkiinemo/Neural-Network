# Custom Neural Network with One Hidden Layer

## Overview  
This repository contains a custom neural network implementation built from scratch using Python and NumPy. The neural network features a single hidden layer and is designed for educational and experimental purposes. The project includes an application on the **Iris dataset** to verify its performance.

## Features  

- **From Scratch Implementation**: The neural network is built without relying on high-level machine learning frameworks, using only Python and NumPy for computations.  
- **Single Hidden Layer**: The architecture includes one hidden layer using the **tanh activation function** (only tanh is supported at the moment).  
- **Basic Training & Prediction**: Supports model training and predictions on structured datasets.  
- **Application**:  
  - **Real Dataset**: Performance evaluation on the **Iris dataset** (classification).  
- **Performance Metric**:  
  - **Accuracy**: Current evaluation is based on classification accuracy (additional metrics like loss and convergence plots may be added later).  

## Repository Structure  

```
/Neural-Network  
│── /__pycache__            # Python cache files (auto-generated)  
│── NN_Model.py             # Core neural network implementation  
│── analise.ipynb           # Jupyter Notebook with Iris dataset testing and analysis  
│── README.md               # This file  
```

## Installation  

1. Clone the repository:  
   ```bash  
   git clone https://github.com/dovahkiinemo/Neural-Network.git  
   cd Neural-Network  
   ```  

2. Install dependencies (NumPy and Jupyter for the notebook):  
   ```bash  
   pip install numpy jupyter  
   ```  

## Usage  

### Basic Example  
```python  
import NN_Model  

# Load your data (example: Iris dataset)  
x_train, y_train = ...  # Training data  
x_test, y_test = ...    # Test data  

# Initialize model (1 hidden layer, output layer size = 3 for Iris)  
model = NN_Model.NN_Model(x_train, y_train, n_hidden_layers=5, n_output_layer=3)  

# Train the model  
model.fit(epochs=1000, learning_rate=0.01)  

# Predict and evaluate accuracy  
predictions = model.predict(x_test)  
accuracy = (predictions == y_test).mean()  
print(f"Test Accuracy: {accuracy:.2%}")  
```  

### Available Methods  
- **`NN_Model(x_train, y_train, n_hidden_layers, n_output_layer)`**  
  - Initializes the neural network with specified hidden layer size and output dimensions.  
- **`fit(epochs, learning_rate)`**  
  - Trains the model on the provided data.  
- **`predict(x_test)`**  
  - Returns predictions for input `x_test`.  

## Future Improvements  
Planned enhancements include:  
- **Additional Activation Functions**: Extend support to ReLU, sigmoid, etc.  
- **More Metrics**: Implement loss tracking and convergence plots.  
- **Optimizations**: Add batch training, learning rate scheduling, or momentum.  
- **Extended Datasets**: Adapt for other small-scale datasets (e.g., MNIST).  

## Contributions  
Feedback and contributions are welcome! Open an issue or submit a PR for suggestions.  

## License  
MIT License. See repository for details.  
