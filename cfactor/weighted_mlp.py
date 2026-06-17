"""Sklearn-compatible MLP regressor with native ``sample_weight`` support.

Drop-in replacement for ``sklearn.neural_network.MLPRegressor`` in the
C-factor pipeline. The key difference: ``fit(X, y, sample_weight=...)``
applies per-sample weights directly in the loss function (weighted MSE),
rather than relying on resampling.

Uses PyTorch for training, but exposes the same hyperparameter interface
as ``MLPRegressor`` so ``build_estimator`` and ``_make_mlp`` need only
swap the class name.

Note on ``alpha`` / weight decay
--------------------------------
sklearn's ``MLPRegressor`` adds ``alpha * 0.5 * ||W||² / n_samples`` to
the loss. PyTorch's ``AdamW`` applies *decoupled* weight decay (Loshchilov
& Hutter, 2019), which subtracts ``weight_decay * param`` directly from the
parameters each step — mathematically different from L2 penalty under Adam.
For the small MLPs used here the practical difference is negligible, but
tuned ``alpha`` values from sklearn may not transfer exactly; re-tune after
switching.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin


class WeightedMLPRegressor(BaseEstimator, RegressorMixin):
    """MLP regressor with weighted MSE loss.

    Parameters match ``sklearn.neural_network.MLPRegressor`` for pipeline
    compatibility. ``solver`` is accepted but only ``'adam'`` is implemented.
    """

    def __init__(self, hidden_layer_sizes=(32,), activation='relu',
                 solver='adam', alpha=1e-4, learning_rate_init=1e-3,
                 max_iter=500, early_stopping=True,
                 validation_fraction=0.1, n_iter_no_change=20,
                 random_state=42, batch_size=256):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver          # stored for get_params; only 'adam' used
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------
    def _build_network(self, n_features: int) -> nn.Sequential:
        activations = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'logistic': nn.Sigmoid}
        act_cls = activations.get(self.activation, nn.ReLU)

        layers: list[nn.Module] = []
        in_size = n_features
        sizes = (self.hidden_layer_sizes
                 if isinstance(self.hidden_layer_sizes, (list, tuple))
                 else (self.hidden_layer_sizes,))
        for h in sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(act_cls())
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, X, y, sample_weight=None):
        torch.manual_seed(self.random_state)
        rng = np.random.RandomState(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        n_samples, n_features = X.shape

        # Normalise weights so their mean is 1 (preserves loss scale
        # relative to the unweighted case, keeping alpha comparable).
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=np.float32)
            w_sum = w.sum()
            w = w * (n_samples / w_sum) if w_sum > 0 else np.ones_like(w)
        else:
            w = np.ones(n_samples, dtype=np.float32)

        # ---- Train / validation split --------------------------------
        if self.early_stopping:
            n_val = max(1, int(n_samples * self.validation_fraction))
            perm = rng.permutation(n_samples)
            val_idx, train_idx = perm[:n_val], perm[n_val:]
        else:
            train_idx = np.arange(n_samples)
            val_idx = np.array([], dtype=int)

        X_tr = torch.from_numpy(X[train_idx])
        y_tr = torch.from_numpy(y[train_idx])
        w_tr = torch.from_numpy(w[train_idx])

        if len(val_idx):
            X_val = torch.from_numpy(X[val_idx])
            y_val = torch.from_numpy(y[val_idx])
            w_val = torch.from_numpy(w[val_idx])
        else:
            X_val = y_val = w_val = None

        # ---- Build model + optimiser ---------------------------------
        self.network_ = self._build_network(n_features)
        optimizer = torch.optim.AdamW(
            self.network_.parameters(),
            lr=self.learning_rate_init,
            weight_decay=self.alpha,
        )

        # ---- Training loop -------------------------------------------
        n_train = len(X_tr)
        batch_size = min(self.batch_size, n_train)
        self.loss_curve_: list[float] = []
        best_val_loss = float('inf')
        no_improve = 0
        best_state: dict | None = None

        for epoch in range(self.max_iter):
            self.network_.train()
            perm_t = torch.randperm(n_train)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_train, batch_size):
                idx = perm_t[start:start + batch_size]
                xb, yb, wb = X_tr[idx], y_tr[idx], w_tr[idx]

                pred = self.network_(xb).squeeze(-1)
                loss = (wb * (pred - yb) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            self.loss_curve_.append(epoch_loss / max(n_batches, 1))

            # ---- Early stopping on weighted validation loss ----------
            if self.early_stopping and X_val is not None:
                self.network_.eval()
                with torch.no_grad():
                    vp = self.network_(X_val).squeeze(-1)
                    val_loss = (w_val * (vp - y_val) ** 2).mean().item()

                if val_loss < best_val_loss - 1e-7:
                    best_val_loss = val_loss
                    no_improve = 0
                    best_state = {k: v.clone()
                                  for k, v in self.network_.state_dict().items()}
                else:
                    no_improve += 1

                if no_improve >= self.n_iter_no_change:
                    break

        # Restore best weights
        if best_state is not None:
            self.network_.load_state_dict(best_state)

        self.n_iter_ = len(self.loss_curve_)
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.network_.eval()
        with torch.no_grad():
            return self.network_(torch.from_numpy(X)).squeeze(-1).numpy()
