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
        return np.sum(np.square(h - y))
    
    def d_cost(self, h, y):
        return (h - y)
    
    def add_layer(self, neurons):
        self.neurons_per_layer.append(neurons)
        
    def initialize_weights_and_biases(self):
        self.neurons_per_layer.append(self.output_neurons)
        np.random.seed(42)
        self.biases = [np.random.randn(1, l) for l in self.neurons_per_layer[1:]]
        self.weights = [np.random.randn(l_prev, l) for l_prev, l in zip(self.neurons_per_layer[:-1], self.neurons_per_layer[1:])]
        
    def forward(self, X):
        h = X
        for b, w in zip(self.biases, self.weights):
            z = np.dot(h, w)
            h = np.maximum(z, 0)
        return h
        
    def backward(self, X, y):
        d_b = [np.zeros(b.shape) for b in self.biases]
        d_w = [np.zeros(w.shape) for w in self.weights]
        
        h = X
        hs = [X]
        zs = []
        
        # forward pass
        for b, w in zip(self.biases, self.weights):
            z = np.dot(h, w)
            zs.append(z)
            h = np.maximum(z, 0)
            hs.append(h)

        cost = self.cost(hs[-1], y)

        d_z = [np.zeros(z.shape) for z in zs]
        
        # backward pass
        d_z[-1] = 2 * (hs[-1] - y)
        d_w[-1] = np.dot(hs[-2].T, d_z[-1])
        
        # backward propagation
        for l in range(2, len(self.neurons_per_layer)):
            d_z[-l] = np.dot(d_z[-l+1], self.weights[-l+1].T)
            dz_c = d_z[-l].copy()
            dz_c[zs[-l] < 0] = 0
            d_w[-l] = np.dot(hs[-l-1].T, dz_c)
    
        return d_b, d_w, cost
        
    def train(self, X, y, epochs=30):
        self.initialize_weights_and_biases()
        for epoch in range(epochs):
            grad_b, grad_w, cost = self.backward(X,y)
            self.weights = [w - self.lr * gw for w, gw in zip(self.weights, grad_w)]
            self.biases = [b - self.lr * gb for b, gb in zip(self.biases, grad_b)]
            
            if epoch%10 == 0:
                print(f"Cost in epoch N°{epoch}: {cost}")
    
    def predict(self, X):
        return self.forward(X)

batch, h_in, h_out = 64, 1000, 10

np.random.seed(42)
x = np.random.randn(batch, h_in)
y = np.random.rand(batch, h_out)

nn = NeuralNetwork(input_size=h_in, output_size=h_out, lr=1e-6)
nn.add_layer(100)
nn.train(x,y, 400)
print(nn.predict(x[0]))
print(y[0])