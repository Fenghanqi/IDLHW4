import numpy as np
class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)

    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        # TODO: Implement forward pass
        
        # Store input for backward pass
        self.A = A
        
        # Store original shape for backward pass
        self.input_shape = A.shape
        
        # Reshape input to 2D: (batch_size, in_features) where batch_size is the product of all dimensions except the last
        batch_size = np.prod(A.shape[:-1])
        A_reshaped = A.reshape(batch_size, A.shape[-1])
        
        # Compute the linear transformation: Z = A @ W^T + b
        Z_reshaped = A_reshaped @ self.W.T + self.b
        
        # Reshape back to original dimensions but with out_features as the last dimension
        output_shape = list(self.input_shape[:-1]) + [self.W.shape[0]]
        Z = Z_reshaped.reshape(output_shape)
        
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # TODO: Implement backward pass
        # Compute gradients (refer to the equations in the writeup)
        
        # Reshape gradient to 2D: (batch_size, out_features)
        batch_size = np.prod(dLdZ.shape[:-1])
        dLdZ_reshaped = dLdZ.reshape(batch_size, dLdZ.shape[-1])
        
        # Reshape stored input to 2D: (batch_size, in_features)
        A_reshaped = self.A.reshape(batch_size, self.A.shape[-1])
        
        # Compute gradient of loss with respect to weights: dL/dW = (dL/dZ)^T @ A
        self.dLdW = dLdZ_reshaped.T @ A_reshaped
        
        # Compute gradient of loss with respect to bias: dL/db = sum(dL/dZ) across batch dimension
        self.dLdb = np.sum(dLdZ_reshaped, axis=0)
        
        # Compute gradient of loss with respect to input: dL/dA = dL/dZ @ W
        dLdA_reshaped = dLdZ_reshaped @ self.W
        
        # Reshape back to original input shape
        self.dLdA = dLdA_reshaped.reshape(self.input_shape)
        
        # Return gradient of loss wrt input
        return self.dLdA