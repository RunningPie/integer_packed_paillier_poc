import numpy as np
from crypto_utils import encrypt_elementwise_paillier, encrypt_packed_paillier, encrypt_ckks

class SimpleMLP:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # Initialize weights using a small random distribution
        # Inputs: 4, Hidden1: 4, Hidden2: 4, Outputs: 3
        # Total parameters = (4*4 + 4) + (4*4 + 4) + (4*3 + 3) = 20 + 20 + 15 = 55
        self.W1 = np.random.randn(4, 4) * np.sqrt(2.0 / 4)
        self.b1 = np.zeros(4)
        self.W2 = np.random.randn(4, 4) * np.sqrt(2.0 / 4)
        self.b2 = np.zeros(4)
        self.W3 = np.random.randn(4, 3) * np.sqrt(2.0 / 4)
        self.b3 = np.zeros(3)

        # Cache for backpropagation
        self.Z1 = None
        self.A1 = None
        self.Z2 = None
        self.A2 = None
        self.Z3 = None
        self.A3 = None

    def get_parameters(self):
        """Flatten all weights and biases into a single 1D numpy array."""
        return np.concatenate([
            self.W1.ravel(),
            self.b1.ravel(),
            self.W2.ravel(),
            self.b2.ravel(),
            self.W3.ravel(),
            self.b3.ravel()
        ])

    def set_parameters(self, params):
        """Set weights and biases from a 1D numpy array."""
        assert len(params) == 55, f"Parameter vector length must be 55, got {len(params)}"
        self.W1 = params[0:16].reshape(4, 4)
        self.b1 = params[16:20]
        self.W2 = params[20:36].reshape(4, 4)
        self.b2 = params[36:40]
        self.W3 = params[40:52].reshape(4, 3)
        self.b3 = params[52:55]

    def forward(self, X):
        """Perform forward propagation."""
        # Layer 1
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.maximum(0, self.Z1)  # ReLU
        
        # Layer 2
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = np.maximum(0, self.Z2)  # ReLU
        
        # Layer 3 (Output)
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        # Stable Softmax
        exp_Z3 = np.exp(self.Z3 - np.max(self.Z3, axis=-1, keepdims=True))
        self.A3 = exp_Z3 / np.sum(exp_Z3, axis=-1, keepdims=True)
        return self.A3

    def backward(self, X, Y, lr=0.05):
        """Perform backward propagation and update weights via SGD."""
        m = X.shape[0]
        
        # Output layer gradient
        dZ3 = (self.A3 - Y) / m
        dW3 = np.dot(self.A2.T, dZ3)
        db3 = np.sum(dZ3, axis=0)
        
        # Hidden layer 2 gradient
        dZ2 = np.dot(dZ3, self.W3.T) * (self.Z2 > 0)
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0)
        
        # Hidden layer 1 gradient
        dZ1 = np.dot(dZ2, self.W2.T) * (self.Z1 > 0)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0)
        
        # Update weights and biases
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W3 -= lr * dW3
        self.b3 -= lr * db3

class Client:
    def __init__(self, client_id, X_train, y_train, seed=None):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train  # Expecting one-hot encoded labels of shape (N, 3)
        self.model = SimpleMLP(seed=seed)

    def set_model_parameters(self, params):
        """Update client's local model parameters with global parameters."""
        self.model.set_parameters(params)

    def get_model_parameters(self):
        """Get current local model parameters."""
        return self.model.get_parameters()

    def train_local(self, epochs, batch_size=8, lr=0.05):
        """Train the local MLP model using the local training dataset shard."""
        num_samples = self.X_train.shape[0]
        for epoch in range(epochs):
            # Shuffle local data
            indices = np.random.permutation(num_samples)
            X_shuffled = self.X_train[indices]
            y_shuffled = self.y_train[indices]
            
            # Mini-batch SGD
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                if len(X_batch) == 0:
                    continue
                
                self.model.forward(X_batch)
                self.model.backward(X_batch, y_batch, lr=lr)

    def encrypt_parameters(self, scheme, crypto_key, packing_method='twos_complement'):
        """
        Encrypt the current local model parameters.
        scheme: 'plaintext', 'elementwise_paillier', 'packed_paillier', or 'ckks'
        crypto_key: public key or context depending on the scheme
        packing_method: 'twos_complement' or 'biased' (used for packed_paillier)
        """
        params = self.model.get_parameters()
        if scheme == 'plaintext':
            return params.copy()
        elif scheme == 'elementwise_paillier':
            return encrypt_elementwise_paillier(crypto_key, params)
        elif scheme == 'packed_paillier':
            return encrypt_packed_paillier(crypto_key, params, packing_method=packing_method)
        elif scheme == 'ckks':
            return encrypt_ckks(crypto_key, params)
        else:
            raise ValueError(f"Unknown encryption scheme: {scheme}")
