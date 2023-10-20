from typing import Callable, Coroutine
from aiohttp.web import Application, Request, Response
from mplug.core import Graph, Scope

KEY = object()

def create_injector(
    app: Application,
    scope: Graph|Callable[[], Scope]
):
    match scope:
        case Graph():
            func = lambda: Scope(scope)
        case call:
            func = call
    app[KEY] = func # type: ignore

def inject(f: Callable[..., Coroutine]) -> Callable[..., Coroutine]:
    async def w(*args):
        values = {}
        if isinstance(args[0], Application):
            app = args[0]
        elif isinstance(args[0], Request):
            app = args[0].app
            values[Request] = args[0]
        else:
            raise RuntimeError("bug (unhandled case)")
        if len(args) > 1 and isinstance(args[1], Response):
            values[Response] = args[1]
        values[Application] = app
        scope = app[KEY]() # type: ignore
        scope.values.update(values)
        async with scope:
            return await scope.call_async(f)
    w.__name__ = f.__name__
    return w
