from dataclasses import dataclass
from inspect import Parameter, signature
from mplug.core import Dep, Graph, Scope
from typer import Context, Typer
from typing import Any, Callable

KEY = "__mplug_scope__"

@dataclass(slots=True, frozen=True)
class InjectMarker:
    dep: Dep

def Inject(dep: Dep) -> Any:
    return InjectMarker(dep)

def inject(f: Callable) -> Callable:
    sig = signature(f)
    newparams = []
    marked = {}
    ctx = None
    for p in sig.parameters.values():
        if isinstance(p.default, InjectMarker):
            marked[p.name] = p.default.dep
        else:
            newparams.append(p)
            if isinstance(p.annotation, type) and issubclass(p.annotation, Context):
                ctx = p.name
    fakectx = not ctx
    if not ctx:
        ctx = "__mplug_ctx__"
        newparams.append(Parameter(ctx, Parameter.KEYWORD_ONLY, annotation=Context))
    if not marked:
        return f
    def w(**kwargs: Any):
        c = kwargs[ctx]
        # TODO?: use sub-scope here for subcommands?
        while not (scope := getattr(c, KEY, None)):
            c = c.parent
        if callable(scope):
            scope = scope()
            scope.__enter__()
            c.call_on_close(lambda: scope.__exit__(None, None, None))
            setattr(c, KEY, scope)
        for k,v in marked.items():
            kwargs[k] = scope.get(v)
        if fakectx:
            kwargs.pop(ctx)
        return f(**kwargs)
    w.__signature__ = sig.replace(parameters=newparams)
    w.__name__ = f.__name__
    return w

def create_injector(
    app: Typer,
    scope: Graph|Callable[[], Scope]
):
    match scope:
        case Graph():
            func = lambda: Scope(scope)
        case call:
            func = call
    @app.callback()
    def __mplug_init_scope__(ctx: Context):
        setattr(ctx, KEY, func)
