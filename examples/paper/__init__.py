"""Paper reproduction recipes for arXiv:1910.09281.

Each script reproduces one row of one figure from
*Dealing with Sparse Rewards in Reinforcement Learning* (Hare 2019).

Layout
------
* ``common.py`` — shared building blocks (MLP / NatureCNN, env factories,
  the three classic-control and three Atari env IDs).
* ``classic_<agent>.py`` — runs ``<agent>`` on the three classic-control
  envs (``Acrobot-v1``, ``CartPole-v1``, ``MountainCar-v0``) using
  the paper's classic-control hyperparameters.
* ``atari_<agent>.py`` — runs ``<agent>`` on the three Atari benchmarks
  (``FreewayDeterministic-v4``, ``MontezumaRevengeDeterministic-v4``,
  ``SpaceInvadersDeterministic-v4``) using the paper's Atari
  hyperparameters.

Each ``main(env_id)`` accepts an env id so you can subset what you
actually want to run, e.g.::

    python examples/paper/atari_rnd.py MontezumaRevengeDeterministic-v4

Logs land under ``logs/paper/<agent>/<env_id>/<timestamp>/`` so
TensorBoard runs from the repo root pick everything up::

    tensorboard --logdir logs/

These configs are deliberately the *paper's* values — large
``total_steps`` (50M for Atari, 5M for classic) and ``num_workers=32``.
For a quick smoke-test pass small overrides via the env or shrink
``total_steps`` directly in the script.
"""
