import numpy as np

class Sequential:

    def __init__(self, loss):
        self.layers = []
        self.loss = loss

    def add(self, layer):
        self.layers.append(layer)

    def forward_propagation(self, X):
        output = X
        outputs = []
        grads = []
        for layer in self.layers:
            g = layer.gradient_dict(output)
            grads.append(g)
            output = layer.eval(output)
            outputs.append(output)
        return outputs, grads

    def back_propagate(self, grads, outputs, y):
        grad_loss = self.loss.grad_input(y, outputs[-1]) # dL/dlast_layer_output
        for layer, grad in list(zip(self.layers, grads))[::-1]:
            dL_dwi, dL_dbi, grad_loss = layer.backprop_grad(grad_loss, grad)
            layer.update((dL_dwi[0], dL_dbi), self.optimizer)            

    def fit(self, X, y, epochs, optimizer, learning_rate, verbose=1):
        self.optimizer = optimizer(learning_rate)
        for i in range(epochs):
            outputs, grads = self.forward_propagation(X)
            self.back_propagate(grads, outputs, y)
            if verbose==1:
                print(f"\rEpoch: {i+1} Loss: {self.loss(y, outputs[-1])}", end="")
        if verbose==0:
            print(f"Epoch: {i} Loss: {self.loss(y, outputs[-1])}")

    def eval(self, X):
        return self.forward_propagation(X)[0][-1]
    
    def summary(self):
        from tabulate import tabulate
        headers = ["#", "Layer Type", "W.shape", "b.shape", "Total parameters"]
        table = []
        total_p = 0
        for i, layer in enumerate(self.layers):
            w_shape, b_shape = layer.get_parameter_shape()
            p = layer.get_total_parameters() # total parameters of a layer
            table.append([i+1, layer.__class__.__name__, w_shape, b_shape, p])
            total_p += p
        print(tabulate(table, headers, tablefmt="pretty"))
        print("Total no. of model parameters", total_p)
        
    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.eval(output)
        return output
    
    def accuracy(self, y, ypred):
        return (y==(ypred>0.5).astype('int')).mean()
            