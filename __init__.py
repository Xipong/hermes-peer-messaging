"""Native Hermes plugin entry point."""


def register(ctx):
    from . import service, threads

    service.register(ctx)
    threads.register(ctx)

__all__ = ["register"]
