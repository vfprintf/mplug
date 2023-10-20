from asyncio import Task, create_task, iscoroutinefunction as asyncio_iscoroutinefunction, wait
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager, contextmanager
from dataclasses import dataclass
from functools import partial
from inspect import Parameter, Signature, get_annotations, isasyncgenfunction, isgeneratorfunction, signature
from itertools import chain
from os import getenv
from sys import stderr
from types import ModuleType, NoneType, UnionType
from typing import Annotated, Any, AsyncGenerator, Awaitable, Callable, Generator, Hashable, Iterator, Mapping, Optional, Sequence, TypeVar, Union, get_args, get_origin

TRACE = getenv("MPLUG_TRACE")

Dep = Hashable

T = TypeVar("T")
D = TypeVar("D", bound=Dep)
C = TypeVar("C", bound=Callable)

def annotate(obj: T, annotation: Any) -> T:
    if not hasattr(obj, "__mplug_annotations__"):
        obj.__mplug_annotations__ = [] # type: ignore
    obj.__mplug_annotations__.append(annotation) # type: ignore
    return obj

def annotator(ann: Any) -> Callable[[T], T]:
    def w(o: T) -> T:
        return annotate(o, ann)
    return w

PROVIDES = object()

def provides(f: C) -> C:
    return annotate(f, PROVIDES)

@dataclass(slots=True, frozen=True)
class Token:
    inner: Dep

def requires(token: Dep) -> Callable[[T], T]:
    return annotator(Token(token))

@dataclass(slots=True, frozen=True)
class Provided:
    dep: Optional[Dep] = None

def provided(dep: Optional[Dep]=None) -> Callable[[T], T]:
    return annotator(Provided(dep))

@dataclass(slots=True, frozen=True)
class Export:
    objs: list

    def __call__(self, o: T, *rest: Any) -> T:
        self.objs.append(o)
        self.objs.extend(rest)
        return o

def export(*objs: Any) -> Export:
    return Export(list(objs))

@dataclass(slots=True, frozen=True)
class meta:
    f: Callable[[str], Any]

def callmeta(m: meta, field: Optional[str]) -> Any:
    # TODO: this could also depend on type and default value
    if not field:
        raise ValueError("`meta' must be applied inside Annotated[...]")
    return m.f(field)

# should be `All = NewType("All", Sequence[T])`, but generics don't work with newtypes
class All(Sequence[T]):
    pass

WANT_MARKER = object()

@dataclass(slots=True, frozen=True)
class Instance:
    cls: type

INSTANCE_MARKER: Any = Instance

@dataclass(slots=True, frozen=True)
class Subscription:
    inner: Dep

@dataclass(slots=True, frozen=True)
class Subscribe:
    dep: Dep

def subscribe(event: Dep) -> Callable[[T], T]:
    return annotator(Subscribe(event))

@dataclass(slots=True, frozen=True)
class OptionalDep:
    inner: Dep

@dataclass(slots=True, frozen=True)
class AllDep:
    inner: Dep

@dataclass(slots=True, frozen=True)
class ConvertDep:
    typ: type
    inner: Dep

Edge = OptionalDep | ConvertDep | AllDep | Dep

@dataclass(slots=True, frozen=True)
class Func:
    edges: list[Edge]
    tokens: list[Dep]
    call: Callable
    is_async: bool
    is_context: bool

POSITIONAL_PARAM = (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)

def parsereturn(f: Callable) -> Dep|Instance:
    sig = signature(f)
    for p in sig.parameters.values():
        if p.kind not in POSITIONAL_PARAM:
            continue
        # TODO: handle Instance annotations here
    r = sig.return_annotation
    if r is Signature.empty:
        raise InjectionError("missing return annotation")
    if isgeneratorfunction(f) or isasyncgenfunction(f):
        r = get_args(r)[0]
    return r

def parsedep(dep: Dep, field: Optional[str] = None) -> Edge:
    if isinstance(dep, meta):
        return parsedep(callmeta(dep, field), field)
    origin = get_origin(dep)
    args = get_args(dep)
    if origin == Annotated:
        typ, args = args[0], args[1:]
        for a in args:
            while True:
                match a:
                    case Provided():
                        return ConvertDep(typ, parsedep(a.dep, field))
                    case meta():
                        a = callmeta(a, field)
                    case _:
                        break
    if origin in (Union, UnionType) and len(args) >= 2 and NoneType in args:
        return OptionalDep(next(parsedep(a, field) for a in args if a is not NoneType))
    if origin == All:
        return AllDep(parsedep(args[0], field))
    return dep

Context = Generator[T, None, None]
AsyncContext = AsyncGenerator[T, None]

def parsefunc(f: Callable) -> Func:
    sig = signature(f)
    edges = []
    for p in sig.parameters.values():
        if p.kind not in POSITIONAL_PARAM:
            continue
        elif (a := p.annotation) is not Signature.empty:
            edges.append(parsedep(a, field=p.name))
        else:
            raise InjectionError("missing annotation or Depends marker")
    if isasyncgenfunction(f):
        is_async, is_context = True, True
        f = asynccontextmanager(f)
    elif (is_async := asyncio_iscoroutinefunction(f)):
        is_context = False
    else:
        is_context = isgeneratorfunction(f)
        if is_context:
            f = contextmanager(f)
    tokens = [
        tok.inner
        for tok in getattr(f, "__mplug_annotations__", ())
        if isinstance(tok, Token)
    ]
    return Func(
        edges      = edges,
        tokens     = tokens,
        call       = f,
        is_async   = is_async,
        is_context = is_context
    )

class Graph:
    dep: dict[Dep, list[Func]]
    inst: dict[type, list[Func]]

    def __init__(self, *modules: Any):
        self.dep = {}
        self.inst = {}
        self.load(*modules)

    def __call__(self, want: Dep) -> Iterator[Func]:
        deps = self.dep.get(want)
        if deps is not None:
            yield from deps
        inst = self.inst.get(type(want))
        #print(f"want {want} ({type(want)}) inst: {inst}")
        if inst is not None:
            yield from inst

    def put_dep(self, dep: Dep, func: Func):
        #print(f"put_dep {dep} | {func}")
        try:
            self.dep[dep].append(func)
        except KeyError:
            self.dep[dep] = [func]

    def put_inst(self, cls: type, func: Func):
        try:
            self.inst[cls].append(func)
        except KeyError:
            self.inst[cls] = [func]

    def provides(self, f: C) -> C:
        func = parsefunc(f)
        match parsereturn(f):
            case Instance(cls):
                self.put_inst(cls, func)
            case dep:
                self.put_dep(dep, func)
        return f

    def subscribe(self, event: Dep) -> Callable[[T], T]:
        def deco(f):
            self.put_dep(Subscription(event), parsefunc(f))
            return f
        return deco

    def put_value(self, dep: Dep, value: Any):
        #print(f"put_value {value} for {dep}")
        self.put_dep(dep, Func(
            edges      = [],
            tokens     = [],
            call       = lambda: value,
            is_async   = False,
            is_context = False
        ))

    def provided(self, dep: Dep) -> Callable[[T], T]:
        def deco(x):
            self.put_value(dep, x)
            return x
        return deco

    def update(self, other: "Graph"):
        self.dep.update(other.dep)
        self.inst.update(other.inst)

    def _process_annotation(
        self,
        obj: Any,
        ann: Any,
        field: Optional[str] = None,
        typ: Optional[type] = None
    ) -> bool:
        #print(f"procann {field} | {ann}")
        match ann:
            case Subscribe(event):
                if isinstance(event, meta):
                    return self._process_annotation(
                        obj,
                        Subscribe(callmeta(event, field)),
                        field,
                        typ
                    )
                self.put_dep(Subscription(event), parsefunc(obj))
            case _ if ann is PROVIDES:
                self.provides(obj)
            case Provided(dep):
                if isinstance(dep, meta):
                    return self._process_annotation(
                        obj,
                        Provided(callmeta(dep, field)),
                        field,
                        typ
                    )
                self.put_value(dep, obj)
            case _ if ann is Provided:
                return self._process_annotation(obj, Provided(typ or type(obj)), field, typ)
            case meta():
                return self._process_annotation(obj, callmeta(ann, field), field, typ)
            case _:
                return False
        return True

    def load(self, *modules: Any):
        for m in modules:
            match m:
                case Graph():
                    self.update(m)
                case Export(objs):
                    self.load(*objs)
                case o if hasattr(o, "__mplug_annotations__"):
                    for a in o.__mplug_annotations__:
                        self._process_annotation(o, a)
                case o:
                    ann = get_annotations(o if isinstance(o, (type, ModuleType)) else type(o))
                    for f in dir(o):
                        v = getattr(o, f)
                        if f in ann:
                            if get_origin(ann[f]) == Annotated:
                                args = get_args(ann[f])
                                ok = False
                                for a in args[1:]:
                                    ok = self._process_annotation(v, a, f, args[0]) or ok
                                if ok:
                                    continue
                        if isinstance(v, (Graph, Export)) or hasattr(v, "__mplug_annotations__"):
                            self.load(v)

GraphFn = Callable[[Dep], Iterator[Func]]
AnyGraph = GraphFn|Sequence["AnyGraph"]

EMPTY_GRAPHFN = lambda _: iter(())

class InjectionError(RuntimeError):
    pass

def tographfn(graph: AnyGraph) -> GraphFn:
    if isinstance(graph, Callable):
        return graph
    fns = list(map(tographfn, graph))
    def fn(want: Dep) -> Iterator[Func]:
        for f in fns:
            yield from f(want)
    return fn

class Scope:
    __slots__ = "graph", "topo", "values", "stack"

    graph: GraphFn
    topo: list["Scope"]
    values: dict[Dep, Any]
    stack: AsyncExitStack

    def __init__(
        self,
        graph: AnyGraph = EMPTY_GRAPHFN,
        outer: "Scope"|Sequence["Scope"] = (),
        values: Mapping[Dep, Any] = {}
    ):
        self.graph = tographfn(graph)
        self.values = { Scope: self, **values }
        self.stack = AsyncExitStack()
        topo = []
        if isinstance(outer, Scope):
            outer = (outer, )
        for o in outer:
            for t in o.inclusive_topo:
                if t not in topo:
                    topo.append(t)
        self.topo = topo

    @property
    def inclusive_topo(self) -> Iterator["Scope"]:
        yield from self.topo
        yield self

    @property
    def reverse_topo(self) -> Iterator["Scope"]:
        yield self
        yield from self.topo.__reversed__()

    def put(self, dep: Dep, value: Any):
        self.values[dep] = value

    # TODO? put_async

    def get(self, want: Dep) -> Any:
        try:
            match inject(self, parsedep(want)):
                case Found(AwaitSlot(_)):
                    raise InjectionError("async dependency in sync context")
                case Found(v):
                    return v
        except NotFound:
            raise InjectionError(f"no value: {want}")

    async def get_async(self, want: Dep) -> Any:
        try:
            match inject(self, parsedep(want)):
                case Found(AwaitSlot(task)):
                    return await task
                case Found(v):
                    return v
        except NotFound:
            raise InjectionError(f"no value: {want}")

    def call(self, f: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
        return synccall(self, parsefunc(partial(f, *args, **kwargs)))

    async def call_async(self, f: Callable[..., T|Awaitable[T]], /, *args: Any, **kwargs: Any) -> T:
        return await asynccall(self, parsefunc(partial(f, *args, **kwargs)))

    def getattr(self, obj: Any, attr: str) -> Any:
        o = getattr(obj, attr)
        if callable(o) and hasattr(o, "__self__"):
            o = self.call(o)
        return o

    async def getattr_async(self, obj: Any, attr: str) -> Any:
        o = getattr(obj, attr)
        if callable(o) and hasattr(o, "__self__"):
            o = await self.call_async(o)
        return o

    def inner(self, values: Mapping[Dep, Any] = {}) -> "Scope":
        return Scope(outer=(self, ), values=values)

    def event(self, event: Any) -> "Event":
        hooks = [*subscriptions(self, Subscription(type(event)))]
        if isinstance(event, Hashable):
            hooks.extend(subscriptions(self, Subscription(event)))
        return Event(self, hooks)

    def __enter__(self): # -> Self
        return self

    def __exit__(self, *exc):
        # this will break one day. but that day is not today.
        ExitStack.__exit__(self.stack, *exc) # type: ignore

    def close(self):
        ExitStack.close(self.stack) # type: ignore

    async def __aenter__(self): # -> Self
        return self

    async def __aexit__(self, *exc):
        await self.stack.__aexit__(*exc)

    async def close_async(self):
        await self.stack.aclose()

def synccall(scope: Scope, func: Func) -> Any:
    a = []
    for d in func.edges:
        match inject(scope, d):
            case Found(AwaitSlot(_)):
                raise InjectionError("async dependency in sync context")
            case Found(v):
                a.append(v)
    for t in func.tokens:
        match inject(scope, t):
            case Found(AwaitSlot(_)):
                raise InjectionError("async token in sync context")
    v = func.call(*a)
    if func.is_context:
        v = scope.stack.enter_context(v)
    return v

async def asynccall(scope: Scope, func: Func) -> Any:
    args = [None] * len(func.edges)
    tasks = []
    idx = []
    for i,d in enumerate(chain(func.edges, func.tokens)):
        match inject(scope, d):
            case Found(AwaitSlot(task)):
                tasks.append(task)
                idx.append(i)
            case Found(v):
                if i < len(args):
                    args[i] = v
    if tasks:
        await wait(tasks)
        for i,t in zip(idx, tasks):
            r = t.result()
            if i < len(args):
                args[i] = r
    v = func.call(*args)
    if func.is_async:
        if func.is_context:
            v = await scope.stack.enter_async_context(v)
        else:
            v = await v
    elif func.is_context:
        v = scope.stack.enter_context(v)
    return v

# TODO: should parse signatures just once
def pipe(*fs: Callable) -> Callable:
    if not fs:
        return lambda: None
    first, *rest = fs
    if not rest:
        return first
    if any(asyncio_iscoroutinefunction(f) or isasyncgenfunction(f) for f in fs):
        async def afw(scope: Scope):
            v = await scope.call_async(first)
            for f in rest:
                v = await scope.call_async(f, v)
            return v
        return afw
    else:
        def fw(scope: Scope):
            v = scope.call(first)
            for f in rest:
                v = scope.call(f, v)
            return v
        return fw

@dataclass(slots=True)
class AwaitSlot:
    task: Task

@dataclass(slots=True)
class Found:
    value: Any
    scope: Scope

    @property
    def is_await(self) -> bool:
        return isinstance(self.value, AwaitSlot)

    @property
    def task(self) -> Task:
        return self.value.task

class NotFound(Exception):
    pass

def find_presolved(scope: Scope, want: Dep) -> Optional[Found]:
    #print(f"looking for presolved {want} in {scope}")
    for t in scope.reverse_topo:
        if want in t.values:
            return Found(t.values[want], t)
    return None

def solvefunc(
    base: Scope,
    want: Dep,
    colored: set[Dep],
    scope: Scope,
    func: Func
) -> Found:
    if want in colored:
        raise InjectionError(f"cycle detected: {want}")
    colored.add(want)
    try:
        deps = [
            inject(scope if isinstance(dep, OptionalDep) else base, dep, colored)
            if dep is not WANT_MARKER
            else Found(want, scope)
            for dep
            in chain(func.edges, func.tokens)
        ]
    finally:
        colored.remove(want)
    subgraph = set(d.scope for d in deps)
    subgraph.add(scope)
    owner = next(
        (
            s for s in base.topo
            if all(t in s.inclusive_topo for t in subgraph)
        ),
        base
    )
    args = [deps[i].value for i in range(len(func.edges))]
    if any(d.is_await for d in deps):
        idx = [i for i in range(len(deps)) if deps[i].is_await]
        aws: list[Task] = [deps[i].task for i in idx]
        async def aw():
            await wait(aws)
            for i,a in zip(idx, aws):
                r = a.result()
                if i < len(args):
                    args[i] = r
            v = func.call(*args)
            if func.is_async:
                if func.is_context:
                    v = await owner.stack.enter_async_context(v)
                else:
                    v = await v
            elif func.is_context:
                v = owner.stack.enter_context(v)
            owner.values[want] = v
            return v
        slot = AwaitSlot(create_task(aw()))
        owner.values[want] = slot
        return Found(slot, owner)
    v = func.call(*args)
    if func.is_async:
        async def aw_():
            if func.is_context:
                ret = await owner.stack.enter_async_context(v)
            else:
                ret = await v
            owner.values[want] = ret
            return ret
        slot = AwaitSlot(create_task(aw_()))
        owner.values[want] = slot
        return Found(slot, owner)
    elif func.is_context:
        v = owner.stack.enter_context(v)
    owner.values[want] = v
    return Found(v, owner)

def solveone(scope: Scope, want: Dep, colored: set[Dep]) -> Found:
    #print(f"-> solveone {want} in {scope}")
    #print(f"   inclusive topo: {list(scope.inclusive_topo)}")
    for t in scope.inclusive_topo:
        for func in t.graph(want):
            #print(f"---> trying {func} in {t}")
            try:
                return solvefunc(scope, want, colored, t, func)
            except NotFound:
                pass
    raise NotFound

def solveall(scope: Scope, want: Dep, colored: set[Dep]) -> list[Found]:
    # TODO: this is not optimal, it overwrites scope.values
    result = []
    for t in scope.inclusive_topo:
        for func in scope.graph(want):
            try:
                result.append(solvefunc(scope, want, colored, t, func))
            except NotFound:
                pass
    return result

def inject(scope: Scope, want: Edge, colored: Optional[set[Dep]] = None) -> Found:
    #print(f"inject {want} in {scope}")
    if TRACE:
        print(f"mplug: inject {want} ({scope})", file=stderr)
    if isinstance(want, OptionalDep):
        try:
            return inject(scope, want.inner, colored)
        except NotFound:
            return Found(None, scope)
    if v := find_presolved(scope, want):
        return v
    if isinstance(want, ConvertDep):
        v = inject(scope, want.inner, colored)
        owner = v.scope
        typ = want.typ
        if v.is_await:
            task = v.task
            async def aw():
                value = await task
                if not isinstance(value, typ):
                    value = typ(value)
                owner.values[want] = value
                return value
            value = AwaitSlot(create_task(aw()))
            v = Found(value, v.scope)
        else:
            value = v.value
            if not isinstance(value, typ):
                value = typ(value)
            v = Found(value, v.scope)
        owner.values[want] = value
        return v
    if colored is None:
        colored = set()
    if isinstance(want, AllDep):
        results = solveall(scope, want.inner, colored)
        aws = [r.task for r in results if r.is_await]
        res = [r.value for r in results if not r.is_await]
        if aws:
            async def aw():
                await wait(aws)
                res.extend(t.result() for t in aws) # type: ignore
                scope.values[want] = res
                return res
            res = AwaitSlot(create_task(aw()))
        scope.values[want] = res
        return Found(res, scope)
    else:
        return solveone(scope, want, colored)

def subscriptions(scope: Scope, want: Subscription) -> Iterator[Func]:
    for t in scope.inclusive_topo:
        yield from t.graph(want)

class Event:
    hooks: Sequence[Func]

    def __init__(self, scope: Scope, hooks: Sequence[Func]):
        self.scope = scope
        self.hooks = hooks

    def iter(self) -> Generator[Any, None, None]:
        for f in self.hooks:
            try:
                yield synccall(self.scope, f)
            except NotFound:
                continue

    def fire(self) -> list:
        return list(self.iter())

    def first(self) -> Any:
        return next(self.iter())

    async def fire_async(self) -> list:
        if self.hooks:
            tasks = [create_task(asynccall(self.scope, f)) for f in self.hooks]
            await wait(tasks)
            return [t.result() for t in tasks if not isinstance(t.exception(), NotFound)]
        else:
            return []

    async def first_async(self) -> Any:
        raise NotImplementedError("TODO")
