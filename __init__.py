"""Native Hermes plugin entry point."""


def register(ctx):
    from . import service, siblings, threads

    service.register(ctx)
    threads.register(ctx)
    siblings.register(ctx)

__all__ = ["register"]
