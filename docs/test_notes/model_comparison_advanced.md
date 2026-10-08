# Comparing ANN, CNN, RNN, LSTM, and GRU

A feedforward artificial neural network, or ANN, maps inputs to outputs without a recurrent hidden state. ANN is also a broad term that includes CNNs and RNNs; here it refers specifically to a multilayer perceptron.

CNNs use local connections and shared filters to learn spatial or temporal patterns. Vanilla RNNs use a recurrent hidden state to process sequences, but may struggle with long-term dependencies because of vanishing gradients.

LSTMs use a separate cell state and input, forget, and output gates. GRUs combine memory into the hidden state and use update and reset gates. Standard GRUs have fewer parameters than standard LSTMs at equal input and hidden dimensions; accuracy depends on the task and training setup.

CNN convolutions can process multiple positions in parallel, while standard recurrent models process time steps sequentially. Choose models based on data structure, dependency length, computational budget, and validation results rather than assuming one architecture is universally best.
