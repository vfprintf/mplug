from typing import Any, AsyncGenerator, Callable
import fastapi
from starlette.requests import HTTPConnection
from mplug.core import Dep, Graph, Scope

# TODO: make parametrizable?
KEY = "plug"

def create_injector(
    app: fastapi.FastAPI,
    scope: Graph|Callable[[], Scope],
):
    match scope:
        case Graph():
            func = lambda: Scope(scope)
        case call:
            func = call
    setattr(app.state, KEY, func)

async def http_scope(connection: HTTPConnection) -> AsyncGenerator[Scope, None]:
    scope: Scope = getattr(connection.app.state, KEY)()
    scope.values.setdefault(HTTPConnection, connection)
    async with scope:
        yield scope

def Inject(dep: Dep) -> Any:
    async def injector(scope: Scope = fastapi.Depends(http_scope)) -> Any:
        return await scope.get_async(dep)
    return fastapi.Depends(injector)
