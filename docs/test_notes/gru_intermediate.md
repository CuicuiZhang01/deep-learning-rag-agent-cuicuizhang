# Gated Recurrent Unit

Gated Recurrent Unit, or GRU, is a type of recurrent neural network designed to learn dependencies in sequential data.

GRUs use two gates: the update gate and the reset gate. The update gate controls how much previous hidden-state information is retained. The reset gate controls how much previous information is used when computing the candidate hidden state.

Unlike LSTMs, GRUs do not maintain a separate cell state. A standard GRU has fewer parameters than a standard LSTM with the same input and hidden dimensions, but neither model is always more accurate.
