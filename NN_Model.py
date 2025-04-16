import numpy as np

class NN_Model:
    def __init__(self, x: np.ndarray, y: np.ndarray, n_hidden_layer: int = 10, n_output_layer: int = 2):
        self.x = x
        self.y = y
        self.hidden_layer = n_hidden_layer
        self.output_layer = n_output_layer
        self.input_layer = self.x.shape[1]

        # Inicialize weights and bias
        # Xavier Inicialization -> Variance of weights is the same for all layers
        self.W1 = np.random.randn(self.input_layer, self.hidden_layer) / np.sqrt(self.input_layer)
        self.B1 = np.zeros((1,self.hidden_layer))
        self.W2 = np.random.randn(self.hidden_layer, self.output_layer) / np.sqrt(self.hidden_layer)
        self.B2 = np.zeros((1,self.output_layer))

        self.model_dict = {'W1': self.W1, 'B1' : self.B1, 'W2' : self.W2, 'B2': self.B2}
        self.z1 = 0 
        self.f1 = 0

    def feed_forward(self, x):
        # Feedfoward Equation 1
        self.z1 = x.dot(self.W1) + self.B1

        # Activation Function 1
        self.f1 = np.tanh(self.z1)

        # Feedfoward Equation 2
        z2 = self.f1.dot(self.W2) + self.B2

        # Normalization (Softmax)
        z_max = np.max(z2, axis=1, keepdims=True)
        z_stable = z2 - z_max
        z2 = np.exp(z_stable)
        self.f2_norm = np.exp(z2)/np.sum(np.exp(z2), axis=1, keepdims=True)

        return self.f2_norm

    def loss_evaluation(self, f2_norm):
        # Cross Entropy
        predictions = np.zeros(self.y.shape[0])
        for i, correct_index in enumerate(self.y):
            predicted = f2_norm[i][correct_index]
            predictions[i] = predicted
        
        log_prob = -np.log(np.mean(predictions))
        return log_prob/self.y.shape[0]

    def back_propagation(self, f2_norm: np.ndarray, learning_rate: float) -> None:
        delta_2 = np.copy(f2_norm)
        delta_2[range(self.x.shape[0]), self.y] -= 1
        delta_w2 = (self.f1.T).dot(delta_2)
        delta_b2 = np.sum(delta_2, axis=0, keepdims=True)

        delta_1 = delta_2.dot(self.W2.T)*(1-np.power(np.tanh(self.z1),2))
        delta_w1 = (self.x.T).dot(delta_1)
        delta_b1 = np.sum(delta_1, axis=0, keepdims=True)

        # Actualization of weights and bias

        self.W1 += - learning_rate*delta_w1
        self.W2 += - learning_rate*delta_w2
        self.B1 += - learning_rate*delta_b1
        self.B2 += - learning_rate*delta_b2

    def fit(self, epochs: int, learning_rate: float):
        for epoch in range(epochs):
            outputs = self.feed_forward(self.x)
            loss = self.loss_evaluation(outputs)
            self.back_propagation(outputs, learning_rate)

            # Accuracy
            prediction = np.argmax(outputs, axis=1)
            correct = (prediction == self.y).sum()
            accuracy = correct/self.y.shape[0]
            if (int(epoch+1) % (epochs/10)) == 0:
                print(f' Epoch: [{epoch+1} / {epochs}] Accuracy: {accuracy:.4f} Loss: {loss.item():.13f}')

        return prediction

    def predict(self, x):
        f2_norm = self.feed_forward(x)
        return np.argmax(f2_norm, axis=1)