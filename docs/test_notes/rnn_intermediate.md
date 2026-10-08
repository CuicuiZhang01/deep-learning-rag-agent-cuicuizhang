# Recurrent Neural Network

Recurrent Neural Network, or RNN, processes sequential data by updating a hidden state at each time step. The hidden state depends on the current input and the previous hidden state.

The same recurrent weights are shared across time steps. This allows an RNN to model ordered data such as text and time series. Training typically uses backpropagation through time.

Vanilla RNNs can suffer from vanishing or exploding gradients, making long-term dependencies difficult to learn. Gradient clipping helps limit exploding gradients. LSTMs and GRUs introduce gating mechanisms that help preserve useful information over longer sequences.
