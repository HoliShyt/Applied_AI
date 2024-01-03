from typing import Sequence

import torch


# Generic MLP Classifier builder allowing for different hidden layer sizes, and activation functions
class MLPNNClassifier(torch.nn.Module):
    def __init__(self, input_size: int = 1, output_size: int = 1,
                 hidden_layer_sizes: Sequence[int] = None,
                 activation_fcn: str = 'relu'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        if not hidden_layer_sizes:
            hidden_layer_sizes = []
        self.hidden_layer_sizes = hidden_layer_sizes

        layer_sizes = [input_size] + hidden_layer_sizes

        # Checking activation function given
        self.activation_fcn = activation_fcn
        if self.activation_fcn not in MLPClassifierActs.keys():
            raise ValueError(f"Invalid activation function descriptor '{self.activation_fcn}'.\n"
                             f"SUupported values are {tuple(MLPClassifierActs.keys())}.")
        activation = MLPClassifierActs[self.activation_fcn]

        # Creating all layers of required sizes and activations
        for i in range(len(layer_sizes)-1):
            self.__setattr__(f"fc{i+1}", torch.nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            self.__setattr__(f"act{i+1}", activation())
        self.n_hidden_layers = len(hidden_layer_sizes)

        # Output layer + probability function layer (log_softmax)
        self.output = torch.nn.Linear(layer_sizes[-1], output_size)
        self.probability_layer = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Feed-forward function, go through each layer in order
        for i in range(1, self.n_hidden_layers + 1):
            x = self.__getattr__(f"fc{i}")(x)
            x = self.__getattr__(f"act{i}")(x)
        x = self.output(x)
        x = self.probability_layer(x)
        return x

    def __str__(self):
        repr_str = f"-MLP Classifier-\n" \
                   f"\tLayers: {[self.input_size]+self.hidden_layer_sizes+[self.output_size]}\n" \
                   f"\tActivation function: {self.activation_fcn}\n"
        return repr_str


# Dictionnaries of implemented activations for MLP classifier
MLPClassifierActs = {'relu': torch.nn.ReLU,
                     'sigmoid': torch.nn.Sigmoid,
                     'tanh': torch.nn.Tanh}

# Mapping of model names to associated classes
MODEL_MAPPINGS = {'mlp': MLPNNClassifier}

# List of all available models
AVAILABLE_MODELS = list(MODEL_MAPPINGS.keys())


# Instantiate a model given a model name and some arguments
def instantiate_model(model_type: str, *args, **kwargs):
    if model_type not in AVAILABLE_MODELS:
        raise ValueError(f"Invalid model type '{model_type}'. Only 'mlp' is supported.\n"
                         f"Supported values are {AVAILABLE_MODELS}.")
    return MODEL_MAPPINGS[model_type](*args, **kwargs)
