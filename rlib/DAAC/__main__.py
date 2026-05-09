"""``python -m rlib.DAAC`` -- YAML-driven DAAC runner.

Pass a YAML config path; use ``--set key.path=value`` to override
individual fields. See :mod:`rlib._cli` for the YAML schema.

Example::

    python -m rlib.DAAC path/to/config.yaml --set trainer.config.total_steps=1_000_000
"""

from __future__ import annotations

from rlib._cli import run_from_yaml


def main(argv: list[str] | None = None) -> None:
    run_from_yaml(prog="python -m rlib.DAAC", argv=argv)


if __name__ == "__main__":
    main()
