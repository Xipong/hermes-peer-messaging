"""Native Hermes plugin entry point."""


def register(ctx):
    from . import service, siblings, threads

    # Bind the real child before collecting any turn-start awareness. One
    # result-transform hook then combines both durable and live-batch notices.
    channel = siblings.register(ctx)
    service.register(ctx, sibling_notices=channel.notices)
    threads.register(ctx)

__all__ = ["register"]
