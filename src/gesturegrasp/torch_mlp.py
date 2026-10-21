"""
GPU-trained MLP gesture classifier with a scikit-learn style interface.

Training re-augments the raw landmark points every epoch (mirror, aspect, rotation, tremor-like
jitter), so the network sees fresh variants instead of one fixed augmented copy. Probabilities are
temperature-scaled on held-out users. The fitted model pickles with joblib (weights kept on CPU)
and predicts on CPU by default: one 126-float row is faster there than a GPU round trip.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import numpy as np
import torch
from torch import nn

logger = logging.getLogger("gesturegrasp.torch_mlp")


def _build_net(n_in: int, n_out: int, hidden: tuple[int, ...], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = n_in
    for h in hidden:
        layers += [nn.Linear(width, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
        width = h
    layers.append(nn.Linear(width, n_out))
    return nn.Sequential(*layers)


class TorchMLPClassifier:
    def __init__(self, hidden: tuple[int, ...] = (512, 256, 128), dropout: float = 0.15,
                 epochs: int = 30, batch_size: int = 2048, lr: float = 2e-3, weight_decay: float = 1e-4,
                 label_smoothing: float = 0.05, patience: int = 6, seed: int = 42, device: str = "auto"):
        self.hidden = tuple(hidden)
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.patience = patience
        self.seed = seed
        self.device = device
        self.temperature = 1.0
        self.classes_: Optional[np.ndarray] = None
        self.history_: list[dict] = []
        self._state: Optional[dict] = None
        self._n_in: Optional[int] = None
        self._net: Optional[nn.Sequential] = None

    # -- training -----------------------------------------------------------------------------
    def fit_epochs(self, make_train_x: Callable[[int], np.ndarray], y: np.ndarray,
                   x_val: np.ndarray, y_val: np.ndarray, score_fn: Callable[[np.ndarray, np.ndarray], float]):
        """make_train_x(epoch) returns that epoch's (augmented, scaled) feature matrix, row-aligned with y."""
        device = torch.device("cuda" if self.device == "auto" and torch.cuda.is_available() else
                              (self.device if self.device != "auto" else "cpu"))
        torch.manual_seed(self.seed)
        self.classes_ = np.unique(y)
        index = {c: i for i, c in enumerate(self.classes_)}
        y_idx = torch.as_tensor(np.vectorize(index.get)(y), device=device)
        counts = np.bincount(y_idx.cpu().numpy(), minlength=len(self.classes_))
        weights = torch.as_tensor(len(y) / (len(counts) * np.maximum(counts, 1)), dtype=torch.float32, device=device)

        x0 = make_train_x(0)
        self._n_in = x0.shape[1]
        net = _build_net(self._n_in, len(self.classes_), self.hidden, self.dropout).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        steps = self.epochs * int(np.ceil(len(y) / self.batch_size))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps)
        loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=self.label_smoothing)
        xv = torch.as_tensor(x_val, dtype=torch.float32, device=device)

        best, best_state, stale = -1.0, None, 0
        for epoch in range(self.epochs):
            t0 = time.time()
            xt = torch.as_tensor(x0 if epoch == 0 else make_train_x(epoch), dtype=torch.float32, device=device)
            net.train()
            perm = torch.randperm(len(y_idx), device=device)
            total = 0.0
            for start in range(0, len(perm), self.batch_size):
                b = perm[start:start + self.batch_size]
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(net(xt[b]), y_idx[b])
                loss.backward()
                opt.step()
                sched.step()
                total += float(loss) * len(b)
            net.eval()
            with torch.no_grad():
                pred = self.classes_[net(xv).argmax(1).cpu().numpy()]
            score = score_fn(y_val, pred)
            self.history_.append({"epoch": epoch + 1, "loss": total / len(y_idx), "val_macro_f1": score,
                                  "seconds": round(time.time() - t0, 1)})
            logger.info("epoch %2d  loss %.4f  val macro-F1 %.4f  (%.1fs on %s)",
                        epoch + 1, total / len(y_idx), score, time.time() - t0, device)
            if score > best:
                best, stale = score, 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    logger.info("Early stop: no val gain for %d epochs.", self.patience)
                    break
        self._state = best_state
        self._net = None
        return self

    def calibrate_temperature(self, x_val: np.ndarray, y_val: np.ndarray) -> float:
        """Fit one temperature on held-out users by minimising NLL."""
        logits = torch.as_tensor(self._logits(x_val), dtype=torch.float32)
        index = {c: i for i, c in enumerate(self.classes_)}
        target = torch.as_tensor(np.vectorize(index.get)(y_val))
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

        def closure():
            opt.zero_grad()
            loss = nn.functional.cross_entropy(logits / log_t.exp(), target)
            loss.backward()
            return loss

        opt.step(closure)
        self.temperature = float(log_t.exp())
        return self.temperature

    # -- inference ----------------------------------------------------------------------------
    def _model(self) -> nn.Sequential:
        if self._net is None:
            net = _build_net(self._n_in, len(self.classes_), self.hidden, self.dropout)
            net.load_state_dict(self._state)
            net.eval()
            self._net = net
        return self._net

    def _logits(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self._model()(torch.as_tensor(np.asarray(x), dtype=torch.float32)).numpy()

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = self._logits(x) / self.temperature
        logits -= logits.max(1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[self._logits(x).argmax(1)]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_net"] = None
        return state
