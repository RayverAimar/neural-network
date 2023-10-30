import numpy as np

class NeuralNetwork():
    def __init__(self, input_size, output_size, lr):
        self.input_neurons = input_size
        self.output_neurons = output_size
        self.lr = lr
        self.weights = []
        self.biases = []
        self.neurons_per_layer = [ input_size ]
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def d_sigmoid(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def cost(self, h, y):
        return (1/2) * np.sum(np.square(h - y))
    
    def d_cost(self, h, y):
        return (h - y)
    
    def add_layer(self, neurons):
        self.neurons_per_layer.append(neurons)
        
    def initialize_weights_and_biases(self):
        self.neurons_per_layer.append(self.output_neurons)
        self.biases = [np.random.rand(1, l) for l in self.neurons_per_layer[1:]]
        self.weights = [np.random.rand(l_prev, l) for l_prev, l in zip(self.neurons_per_layer[:-1], 
                                                                       self.neurons_per_layer[1:])]
    
    def forward(self, X):
        h = X
        for b, w in zip(self.biases, self.weights):
            z = np.dot(h, w) + b
            h = self.sigmoid(z)
        return h
    
    def backward(self, X, y):
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        
        h = X
        hs = [X]
        zs = []
        
        # forward pass
        for b, w in zip(self.biases, self.weights):
            z = np.dot(h, w) + b
            zs.append(z)
            h = self.sigmoid(z)
            hs.append(h)
                
        cost = self.cost(hs[-1], y)

        # backward pass
        delta = self.d_cost(hs[-1], y) * self.d_sigmoid(zs[-1])
        gradient_b[-1] = np.sum(delta, axis=0, keepdims=True)
        gradient_w[-1] = np.dot(hs[-2].T, delta)
        
        # backward propagation
        for l in range(2, len(self.neurons_per_layer)):
            z = zs[-l]
            d_s = self.d_sigmoid(z)
            delta = np.dot(delta, self.weights[-l+1].T) * d_s
            gradient_b[-l] = np.sum(delta, axis=0, keepdims=True)
            gradient_w[-l] = np.dot(hs[-l-1].T, delta)
        
        return gradient_b, gradient_w, cost
        
    def train(self, X, y, epochs=30):
        self.initialize_weights_and_biases()
        for _ in range(epochs):
            grad_b, grad_w, cost = self.backward(X,y)
            self.weights = [w - self.lr * gw for w, gw in zip(self.weights, grad_w)]
            self.biases = [b - self.lr * gb for b, gb in zip(self.biases, grad_b)]
            print(cost)
    
    def predict(self, X):
        return self.forward(X)
    

batch, h_in, h_out = 10, 3, 2

x = np.random.rand(batch, h_in)
y = np.random.rand(batch, h_out)

nn = NeuralNetwork(input_size=3, output_size=2, lr=0.5)
nn.add_layer(2)
nn.add_layer(4)
nn.add_layer(3)
nn.train(x,y)