import numpy as np

class Sequential:

    def __init__(self, loss):
        self.layers = []
        self.loss = loss

    def add(self, Layer, *args, **kwargs):
        if len(self.layers)>=1:
            kwargs['input_size'] = self.layers[-1].get_output_size()
        self.layers.append(Layer(*args, **kwargs))

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
            # print(layer.__class__.__name__)
            dL_dwi, dL_dbi, grad_loss = layer.backprop_grad(grad_loss, grad)
            layer.update((dL_dwi, dL_dbi), self.optimizer)            

    def get_batch(self, X, y, batch_size):
        ixs = np.arange(X.shape[0])
        np.random.shuffle(ixs)
        for i in range(0, X.shape[0], batch_size):
            x_batch, y_batch = X[ixs[i:i+batch_size]], y[ixs[i:i+batch_size]]
            if len(x_batch):
                yield x_batch, y_batch.reshape(-1,1)
        return 

    def fit(self, X, y, epochs, optimizer, learning_rate, verbose=1, **kwargs):
        self.optimizer = optimizer(learning_rate)
        if kwargs.get('batch_size') != None:
            batch_size = kwargs['batch_size']
        else:
            batch_size = X.shape[0]
        
        for i in range(epochs):
            for X_batch, y_batch in self.get_batch(X, y, batch_size):
                outputs, grads = self.forward_propagation(X_batch)
                self.back_propagate(grads, outputs, y_batch)
                if verbose==1:
                    # ypred = self.predict(X)
                    # Loss = self.loss(y, ypred)
                    print(f"\rEpoch: {i+1}/{epochs} Loss: {self.loss(y_batch, outputs[-1])}", end="")
        if verbose==0:
            print(f"Epoch: {i} Loss: {self.loss(y_batch, outputs[-1])}")

    def eval(self, X):
        return self.forward_propagation(X)[0][-1]
    
    def summary(self):
        from tabulate import tabulate
        headers = ["#", "Layer Type", "W.shape", "b.shape", "Output shape", "Total parameters"]
        table = []
        total_p = 0
        for i, layer in enumerate(self.layers):
            w_shape, b_shape = layer.get_parameter_shape()
            o_shape = layer.get_output_size()
            p = layer.get_total_parameters() # total parameters of a layer
            table.append([i+1, layer.__class__.__name__, w_shape, b_shape, o_shape, p])
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
            