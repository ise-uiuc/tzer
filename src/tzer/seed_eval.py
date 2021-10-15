from random import random
from abc import ABC, abstractmethod

import numpy as np
from typing import List, Tuple

import torch
from torch import nn

class PassSeedEvaluator(ABC):
    @abstractmethod
    def evaluate_one(self, seq):
        pass

    def get_best(self, seqs) -> Tuple[int, List]:
        idx = np.argmax(self.evaluate_one(seq) for seq in seqs)
        return idx, seqs[idx]

class RandomEvaluator(PassSeedEvaluator):
    def evaluate_one(self, _):
        return random()

class SimpleLSTMEvaluator(PassSeedEvaluator):
    # Simply report the total coverage.
    def __init__(self, n_pass, num_layers = 3):
        self.n_pass = n_pass 

        # dim: 3 -> {batch, n_seq, n_pass (one hot)}
        class Model(torch.nn.Module) :
            def __init__(self, embedding_dim, hidden_dim, num_layers) :
                super().__init__()
                self.lstm = nn.LSTM(
                    embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
                self.linear = nn.Linear(hidden_dim, 1)
                
            def forward(self, x):
                _, (ht, _) = self.lstm(x)
                return self.linear(ht[-1])

        self.lstm = Model(
            embedding_dim=self.n_pass, 
            hidden_dim=8,
            num_layers=num_layers)

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.lstm.parameters())
        self.iter = 0
    
    def update(self, seq, cov):
        self.lstm.train()
        self.lstm.zero_grad()

        inp = torch.zeros((1, len(seq), self.n_pass))

        for i, idx in enumerate(seq):
            inp[0, i, idx] = 1

        out = self.lstm(inp)

        loss = self.loss(out.squeeze(), torch.scalar_tensor(cov))
        loss.backward()
        nn.utils.clip_grad_norm_(self.lstm.parameters(), 5)
        self.optimizer.step()
        self.iter += 1

    @torch.no_grad()
    def evaluate_one(self, seq):
        self.lstm.eval()

        inp = torch.zeros((1, len(seq), self.n_pass))

        for i, idx in enumerate(seq):
            inp[0, i, idx] = 1

        out = self.lstm(inp)

        return out.squeeze().detach().numpy()

if __name__ == "__main__":
    evaluator = SimpleLSTMEvaluator(30)
    for _ in range(100):
        evaluator.update([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 0.5)
    print(evaluator.evaluate_one([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
