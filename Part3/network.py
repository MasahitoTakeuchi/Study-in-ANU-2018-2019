import numpy as np
import function

class Network:

    def __init__(self, 
                 num_nodes_in_layers, 
                 batch_size,
                 num_epochs,
                 learning_rate
                 ):

        self.num_nodes_in_layers = num_nodes_in_layers
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        # Build the network with one hidden layer
        self.weight1 = np.random.normal(0, 1, [self.num_nodes_in_layers[0], self.num_nodes_in_layers[1]])
        self.bias1 = np.zeros((1, self.num_nodes_in_layers[1]))
        self.weight2 = np.random.normal(0, 1, [self.num_nodes_in_layers[1], self.num_nodes_in_layers[2]])
        self.bias2 = np.zeros((1, self.num_nodes_in_layers[2]))
        self.loss = []

    
    def train(self, inputs, labels):
        
        for epoch in range(self.num_epochs):
            iteration = 0
            print("Epoch:", epoch+1)
            
            while iteration < len(inputs):

                # Batch input
                inputs_batch = inputs[iteration:iteration+self.batch_size]
                labels_batch = labels[iteration:iteration+self.batch_size]
                
                # Forward pass
                z1 = np.dot(inputs_batch, self.weight1) + self.bias1
                a1 = function.relu(z1)
                z2 = np.dot(a1, self.weight2) + self.bias2
                y = function.softmax(z2)
                
                # Compute loss
                loss = function.cross_entropy(y, labels_batch)
                loss += function.L2_regularization(0.01, self.weight1, self.weight2) #lambda
                self.loss.append(loss)

                # Backward pass
                delta_y = (y - labels_batch) / y.shape[0]
                delta_hidden_layer = np.dot(delta_y, self.weight2.T) 
                delta_hidden_layer[a1 <= 0] = 0 # derivatives of Relu

                # Backpropagation
                weight2_gradient = np.dot(a1.T, delta_y) # forward * backward
                bias2_gradient = np.sum(delta_y, axis = 0, keepdims = True)
                weight1_gradient = np.dot(inputs_batch.T, delta_hidden_layer)
                bias1_gradient = np.sum(delta_hidden_layer, axis = 0, keepdims = True)

                # L2 regularization
                weight2_gradient += 0.01 * self.weight2
                weight1_gradient += 0.01 * self.weight1

                # Stochastic Gradient Descent (SGD)
                self.weight1 -= self.learning_rate * weight1_gradient # update weight and bias
                self.bias1 -= self.learning_rate * bias1_gradient
                self.weight2 -= self.learning_rate * weight2_gradient
                self.bias2 -= self.learning_rate * bias2_gradient
                

                iteration += self.batch_size
                
        print("Finish training")
            
                
    def test(self, inputs, labels):
        
        input_layer = np.dot(inputs, self.weight1)
        hidden_layer = function.relu(input_layer + self.bias1)
        scores = np.dot(hidden_layer, self.weight2) + self.bias2
        probs = function.softmax(scores)
        acc = float(np.sum(np.argmax(probs, 1) == labels)) / float(len(labels))
        
        print("Finish testing")
        print("Accuracy: ", acc*100)

