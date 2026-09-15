"""Allow ``python -m mpm`` to behave like the ``mpm`` console script."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
