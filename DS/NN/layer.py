import numpy as np

class Dot:

    def __init__(self, input_size, units):
        self.W = np.random.randn(input_size, units)
        self.b = np.random.randn(1, units)

    def __call__(self, X):
        return self.eval(X)

    def eval(self, X):
        return X.dot(self.W) + self.b

    def grad_input(self, X):
        return np.stack([self.W.T]*X.shape[0], axis=0) 

    def grad_w(self, X):
        I = np.identity(self.b.shape[1])
        m1 = np.stack([I]*self.W.shape[0], axis=1)
        return np.einsum('ijk,mj->mijk', m1, X)

    def grad_b(self, X):
        return np.stack([np.identity(self.b.shape[1])]*X.shape[0], axis=0)
    
    def get_parameter_shape(self):
        return self.W.shape, self.b.shape


class Dense:

    def __init__(self, input_size, activation, units):
        """
        input_size: no. of neurons in previous layer
        activation: some activation funtion
        units: no. of neurons in current layer 
        """
        if isinstance(input_size, tuple):
            input_size = input_size[1]
        self.activation = activation
        self.units = units
        self.dot = Dot(input_size, units)

    def eval(self, X):
        return self.activation(self.dot(X))

    def grad_input(self, X):
        g1 = self.activation.grad_input( self.dot(X) )
        g2 = self.dot.grad_input(X)
        return np.einsum('mij,mjk->mik', g1, g2)

    def grad_parameters(self, X):
        da_dI = self.activation.grad_input(self.dot(X))
        dI_dw = self.dot.grad_w(X)
        da_dw = np.einsum('mij,mjkl->mikl', da_dI, dI_dw)

        dI_db = self.dot.grad_b(X)
        # print(da_dI.shape, dI_dw.shape, dI_db.shape)
        da_db = np.einsum('mij,mjk->mik',  da_dI, dI_db)
        return da_dw, da_db

    def backprop_grad(self, grad_loss, grad):
        dL_dwi = np.einsum('mij,mjkl->mikl', grad_loss, grad['w']).sum(axis=0)[0]
        dL_dbi = np.einsum('mij,mjk->mik', grad_loss, grad['b']).sum(axis=0)
        grad_loss = np.einsum('mij,mjk->mik', grad_loss, grad['input'])
        return dL_dwi, dL_dbi, grad_loss
        
    def gradient_dict(self, output):
        g = {}
        g['input'] = self.grad_input(output)
        g['w'], g['b'] = self.grad_parameters(output)
        return g

    def update(self, grad, optimizer):
        """ grad: (dL_dwi, dL_dbi)"""
        self.dot.W = optimizer.minimize(self.dot.W, grad[0])
        self.dot.b = optimizer.minimize(self.dot.b, grad[1])
        
    def get_parameter_shape(self):
        return self.dot.get_parameter_shape()
    
    def get_total_parameters(self):
        w_shape, b_shape = self.dot.get_parameter_shape()
        return np.prod(w_shape) + np.prod(b_shape)

    def get_output_size(self):
        return self.dot.b.shape # (1,units)



class Flatten:
    
    def __init__(self, input_size):
        if input_size[0] <= 0 or input_size[1] <= 0:
            raise ValueError(f"Input image size is invalid, got {input_size}")
        self.h, self.w, self.c = input_size
        
    def eval(self, X):
        return X.reshape(-1, self.h*self.w*self.c)
    
    
    def gradient_dict(self, X):
        return {}

    def backprop_grad(self, grad_loss, grad): # abcd (grad_loss: 10,1,1620)
        return None, None, grad_loss[:, 0, :].reshape(-1, self.h, self.w, self.c) # -> 10, 18, 18, 5 
        
    def update(self, grad, optimizer):
        """ grad: (dL_dwi, dL_dbi)"""
        pass
        
    def get_parameter_shape(self):
        return ("-","-")
    
    def get_output_size(self):
        return (1,self.h*self.w*self.c)
    
    def get_total_parameters(self):
        return 0


class Conv2D:
    
    def __init__(self, ksize, filters, input_size, activation, stride=1, padding=0):
        if input_size[0] <= 0 or input_size[1] <= 0:
            raise ValueError(f"Input image size is invalid, got {input_size}")
        self.ksize = ksize
        self.filters = filters # no. of kernels in a layer -> no. of channels in each output
        self.stride = stride
        self.padding = padding
        self.input_size = input_size # to decide no. of channels in the kernel
        self.channels = input_size[-1]
        self.activation = activation
        self.kernels = []
        for i in range(self.filters):
            k = np.random.randn(ksize, ksize, self.channels)
            self.kernels.append(k)
        self.bias = np.random.randn(1,self.filters)
        
    @staticmethod
    def _rotate(inp):
        assert len(inp.shape)==4, f"No. of dim in inp not equal to 4, got {inp.shape}"
        return np.flip(inp, axis=(1,2))

    @staticmethod
    def _convolution_op_helper(inp, kernel, stride=1):
        assert len(inp.shape)==4, f"No. of dim in inp not equal to 4, got {inp.shape}"
        assert len(kernel.shape)==4, f"No. of dim in kernel not equal to 4, got {kernel.shape}"
        assert inp.shape[-1] == kernel.shape[-1], f"Mismatch in no. of channels in inp and kernel, got inp {inp.shape[-1]}, kernel {kernel.shape[-1]}"
        assert kernel.shape[1] == kernel.shape[2], f"dim 0 of kernel doesn't match dim 1, got {kernel.shape}"
        assert inp.shape[1]>=kernel.shape[1] and inp.shape[2]>=kernel.shape[2], f"Inp map dim(1,2) < kernel dim(1,2), got inp map dim 1, 2 {inp.shape[1:-1]}, kernel dim 1,2 {kernel.shape[1:-1]}"

        # flip the kernel
        kernel = Conv2D._rotate(kernel)

        oup = []
        start_rloc = 0
        end_rloc = kernel.shape[1]
        while end_rloc <= inp.shape[1]:
            output = []
            start_cloc = 0
            end_cloc = kernel.shape[2]
            while end_cloc <= inp.shape[2]:
                conv = (inp[:,start_rloc:end_rloc, start_cloc:end_cloc]*kernel).sum(axis=(1,2,3))
                output.append(conv)

                start_cloc += stride
                end_cloc += stride
            oup.append(output)
            start_rloc += stride
            end_rloc += stride
        return np.moveaxis(oup, -1, 0)
    
    def _convolution_op(self, inp):
        output = []
        for kernel in self.kernels:
            o = Conv2D._convolution_op_helper(inp, np.expand_dims(kernel, axis=0), self.stride)
            output.append(o)
        output = np.stack(output, axis=-1)
        return output
    
    def _pad_grad_I(self, grad_I):
        return np.pad(grad_I, [(0, 0), (0, self.input_size[0] - grad_I.shape[1]), (0, self.input_size[1] - grad_I.shape[2]), (0,0)])
            
    @staticmethod
    def _pad(inp, pad_width):   
        assert len(inp.shape)==4, f"No. of dim in inp not equal to 4, got {inp.shape}"
        return np.pad(inp, ((0,0), (pad_width,pad_width), (pad_width,pad_width), (0,0)))

    @staticmethod
    def _inside_pad(inp, pad_width):
        assert len(inp.shape)==4, f"No. of dim in inp not equal to 4, got {inp.shape}"
        ix = np.repeat(np.arange(1, inp.shape[1]), pad_width)
        inp = np.insert(inp, ix, 0, axis=1)
        return np.insert(inp, ix, 0, axis=2)
        
    def eval(self, X):
        o_ = self._convolution_op(X) + self.bias
        return self.activation(o_)

    def grad_activation(self, X): #pqrs
        o_ = self._convolution_op(X) + self.bias # shape: m, h, w, c; eg (50, 3,3,2)
        m, h, w, c = o_.shape # (50, 2,2, 5)
        do_do_ = self.activation.grad_input(o_.reshape(m, h*c*w)) # shape of do_do-: (50, 20, 20)
        return np.diagonal(do_do_, axis1=1, axis2=2).reshape(o_.shape)
    
    def gradient_dict(self, X):
        g = {}
        g['activation'] = self.grad_activation(X) # do_do_
        g['input'] = self.get_input(X)
        return g
        
    def get_input(self, X):
        out_h, out_w, _ = self.get_output_size()
        h = (out_h-1)*self.stride - 2*self.padding + self.ksize
        w = (out_w-1)*self.stride - 2*self.padding + self.ksize
        return Conv2D._rotate(X[:, :h, :w, :]) # flip input

    def backprop_grad(self, grad_loss, grad): # abcd
        # to find dL_dwi and dL_dbi, we need dL_do and do_do_. 
        
        """grad: dictionary, keys: activation, input"""
        do_do_ = grad['activation'] # pqrs
        ##################################
        #                                #
        #          dL_dbi                #
        #                                #
        ##################################
        b, h, w, c = grad_loss.shape
        dL_do_ = grad_loss * do_do_[:,:h, :w,:]
        dL_dbi = []
        for c in range(dL_do_.shape[-1]):
            b = dL_do_[:,:,:,c].sum(axis =(1, 2, 0))
            dL_dbi.append(b)
        dL_dbi = np.array(dL_dbi).reshape(1,-1)
        ##################################
        #                                #
        #          dL_dwi                #
        #                                #
        ##################################
        kernels = Conv2D._inside_pad(dL_do_, self.stride-1) # abcd*pqrs -> act as a kernel while computing dL_dwi # 18,18,5
        inps = grad['input'] # 20, 20,10
        dL_dwi = [] # len should be same no. of filters in this layer
        for i in range(dL_do_.shape[-1]): # 5 times
            kernel = kernels[:,:,:,i] # 1, 18,18, 1
            dwi = []
            for j in range(inps.shape[-1]): # 10 times
                inp = inps[...,j] # 1, 20, 20, 1
                conv = Conv2D._convolution_op_helper(np.expand_dims(inp,axis=-1) , np.expand_dims(kernel, axis=-1))
                dwi.append(conv)
#             print(conv.shape) # 10,m, 3 ,3
            dwi = np.transpose(np.array(dwi), (1,0,2,3)).sum(axis=0) # (m, 10, 3, 3).sum(axis=0) -> 10, 3, 3
            dwi = np.transpose(dwi, (1,2,0)) 
            dL_dwi.append(dwi)
        ##################################
        #                                #
        #          dL_dI                 #
        #                                #
        ##################################
        inps = Conv2D._pad(kernels, self.ksize-1)
        kernels = self.kernels
        dL_dI = []
        for i in range(self.input_size[-1]):
            ## ith channel of jth kernel needs to convolve with jth channel of inp 
            kernel = [self.kernels[j][...,i] for j in range(len(self.kernels))]
            kernel = np.stack(kernel, axis=-1) # 3,3,5
            conv = Conv2D._convolution_op_helper(inps, np.expand_dims(kernel, axis=0))
            dL_dI.append(conv)
        dL_dI = np.stack(dL_dI, axis=-1)
        return dL_dwi, dL_dbi, self._pad_grad_I(dL_dI)
    
    def _pad_grad_I(self, grad_I):
        return np.pad(grad_I, [(0, 0), (0, self.input_size[0] - grad_I.shape[1]), (0, self.input_size[1] - grad_I.shape[2]), (0,0)])
        
    def update(self, grad, optimizer):
        """ grad: (dL_dwi, dL_dbi)"""
        self.bias = optimizer.minimize(self.bias, grad[1])
        for i in range(len(self.kernels)):
            self.kernels[i] = optimizer.minimize(self.kernels[i], grad[0][i]) 
            
    def get_parameter_shape(self):
        return self.kernels[0].shape, self.bias.shape
    
    def get_output_size(self):
        m, n, k, p, s = self.input_size[0], self.input_size[1], self.ksize, self.padding, self.stride
        return ((m-k+(2*p))//s)+1, ((n-k+(2*p))//s)+1, self.filters
    
    def get_total_parameters(self):
        return np.prod((len(self.kernels), *self.kernels[0].shape)) + np.prod(self.bias.shape)