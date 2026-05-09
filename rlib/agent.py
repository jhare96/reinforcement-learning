"""Common abstract base class for trainable agents in rlib.

Almost every agent in :mod:`rlib` reimplements the same boilerplate:

* store the same set of LR / grad-clip hyperparameters,
* build a polynomial-decay LR scheduler on top of an optimiser, and
* run an identical
  ``loss.backward → clip_grad_norm_ → optimiser.step → zero_grad → scheduler.step``
  sequence at the end of every ``backprop`` call.

:class:`Agent` factors that out into a single abstract base so agent
implementations can focus on what is actually agent-specific (their
forward pass, ``evaluate`` signature, loss formulation, and ``backprop``
signature).

The base class deliberately does **not** know about any specific RL
algorithm.  Algorithm-specific loss functions live on per-algorithm
subclasses (e.g. :class:`rlib.A2C.A2CModel`) which their concrete
variants (feed-forward, recurrent, ...) can inherit and reuse.

Subclasses are expected to:

1. Call ``super().__init__(config=...)`` with a :class:`ModelConfig`
   (or subclass).
2. Build their network heads (policy/value/Q/etc.) attached to
   ``self.device``.
3. Optionally call :meth:`Agent._build_optimiser` once they have all
   their parameters in place. Composite agents that delegate
   optimisation to a child can simply skip this step.
4. Implement ``forward`` (inherited from :class:`torch.nn.Module`) and
   the abstract :meth:`backprop` and :meth:`evaluate` methods.
5. End each ``backprop`` method with ``return self._train_step(loss)``.

Per-agent ``ModelConfig`` subclasses (``A2CConfig``, ``PPOConfig``, ...)
live next to their respective agent class in ``rlib/<Agent>/model.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from rlib.utils.schedulers import polynomial_sheduler

__all__ = ["Agent", "ModelConfig"]


@dataclass(frozen=True)
class ModelConfig:
    """Shared hyperparameters for every :class:`Agent`.

    Attributes:
        lr: Initial learning rate.
        lr_final: Final learning rate the polynomial scheduler decays to.
        decay_steps: Optimiser steps over which the LR decays from
            ``lr`` to ``lr_final``.  Use ``lr_final == lr`` for a
            constant LR.
        grad_clip: Maximum gradient norm; ``None`` disables clipping.
        device: Torch device string (``"cuda"``, ``"cpu"``, ...).
    """

    lr: float = 1e-3
    lr_final: float = 0.0
    decay_steps: int = 600_000
    grad_clip: float | None = 0.5
    device: str = "cuda"


class Agent(torch.nn.Module, ABC):
    """Abstract base class for trainable rlib agent models.

    Args:
        config: A :class:`ModelConfig` (or subclass) holding LR /
            scheduler / device hyperparameters.

    Attributes:
        config: The original config object (immutable).
        lr, lr_final, decay_steps, grad_clip, device: Convenience
            mirrors of the corresponding ``config`` fields.
    """

    optimiser: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LambdaLR
    config: ModelConfig

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        # Mirror the most-frequently-read fields onto ``self`` for
        # ergonomics — call sites read ``self.lr`` etc. directly.
        self.lr = config.lr
        self.lr_final = config.lr_final
        self.decay_steps = config.decay_steps
        self.grad_clip = config.grad_clip
        self.device = config.device

    # ------------------------------------------------------------------
    # Abstract contract
    # ------------------------------------------------------------------
    # ``forward`` is intentionally not redeclared here — it is inherited
    # from :class:`torch.nn.Module` and each concrete subclass defines
    # its own signature.

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        """Numpy-in / numpy-out inference call used by agent rollouts.

        The exact signature is agent-specific (feed-forward agents take
        a single observation array, recurrent agents also take a
        hidden-state tuple, etc.), so each concrete subclass declares
        its own.  Implementations should run :meth:`forward` under
        ``torch.no_grad()`` and convert torch tensors back to numpy.
        """

    @abstractmethod
    def backprop(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Run a full training step (numpy in / numpy loss-scalar out).

        Implementations typically: convert numpy inputs to tensors,
        call :meth:`forward`, compute the agent-specific loss, then
        return ``self._train_step(loss)``.
        """

    # ------------------------------------------------------------------
    # Optimiser / scheduler construction
    # ------------------------------------------------------------------
    def _build_optimiser(
        self,
        optim: type[torch.optim.Optimizer] = torch.optim.RMSprop,
        optim_args: dict[str, Any] | None = None,
        scheduler_power: float = 1.0,
    ) -> None:
        """Build ``self.optimiser`` and ``self.scheduler``.

        Call this once, after all sub-modules whose parameters should
        be optimised have been registered on ``self``.
        """
        optim_args = optim_args or {}
        # NB: Optimizer's protocol signature doesn't expose ``lr`` as a
        # keyword (only the concrete subclasses do), so pass it positionally.
        self.optimiser = optim(self.parameters(), self.lr, **optim_args)
        self.scheduler = polynomial_sheduler(
            self.optimiser, self.lr_final, int(self.decay_steps), power=scheduler_power
        )

    # ------------------------------------------------------------------
    # Generic loss building blocks (algorithm-agnostic)
    # ------------------------------------------------------------------
    @staticmethod
    def value_loss(R: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Half mean-squared-error between targets ``R`` and values ``V``.

        Used by every value-based agent in this library (A2C critic,
        PPO critic, DQN, Q-aux, ...). Algorithm-specific losses (A2C
        actor-critic loss, PPO clipped objective, ...) belong on the
        relevant per-algorithm :class:`Model` subclass.
        """
        return 0.5 * torch.mean(torch.square(R - V))

    # ------------------------------------------------------------------
    # Training step boilerplate
    # ------------------------------------------------------------------
    def _train_step(self, loss: torch.Tensor) -> np.ndarray:
        """Run the standard backward → clip → step → zero → schedule cycle.

        Returns the loss value as a detached numpy scalar so callers
        can log it without holding on to the autograd graph.
        """
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimiser.step()
        self.optimiser.zero_grad()
        self.scheduler.step()
        return loss.detach().cpu().numpy()
