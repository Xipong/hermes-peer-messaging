"""Native Hermes plugin entry point."""


def register(ctx):
    from . import service

    service.register(ctx)


__all__ = ["register"]
