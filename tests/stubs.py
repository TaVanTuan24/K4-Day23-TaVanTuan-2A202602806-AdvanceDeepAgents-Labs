"""Offline test helpers.

The lab's third-party stack (deepagents, langchain, daytona) is not needed to test the pure logic of
retry / slugify / citation checking / metadata, and it is not installed in every environment.  These
helpers install minimal fakes into sys.modules only for the duration of a `with` block, so the tests run
on a bare Python 3.11+ with no network.

NOTHING HERE PROVES THE REAL LIBRARIES ARE CALLED CORRECTLY - that is what `python tools.py` and one real
`python research.py "<topic>"` run are for.
"""
import contextlib
import sys
import types
from unittest import mock


def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


class _PassthroughTool:
    """Stand-in for langchain_core.tools.tool: keeps the function callable and gives it .invoke()."""

    def __call__(self, fn):
        fn.invoke = lambda payload=None, _fn=fn: _fn(**(payload or {}))
        return fn


def install_tool_stub():
    """Make `from langchain_core.tools import tool` work without langchain installed.

    `dotenv` is stubbed too: tools.py loads .env when imported, and a unit test must not depend on the
    machine's dotenv version or on a .env file being present.
    """
    return mock.patch.dict(sys.modules, {
        "langchain_core.tools": _module("langchain_core.tools", tool=_PassthroughTool()),
        "langchain_core": _module("langchain_core"),
        "dotenv": _module("dotenv", load_dotenv=lambda *a, **k: None),
    })


def install_agent_stubs():
    """Make `agents.py` importable: fake deepagents + the three middleware classes.

    The fakes record what create_deep_agent / the middleware were constructed with, which is exactly what
    a unit test can check about the agent wiring.
    """
    calls = {"create_deep_agent": [], "middleware": []}

    class _Middleware:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            for key, value in kwargs.items():
                setattr(self, key, value)      # keep run_limit / thread_limit / exit_behavior inspectable
            calls["middleware"].append((type(self).__name__, kwargs))

    class TodoListMiddleware(_Middleware):
        pass

    class ModelCallLimitMiddleware(_Middleware):
        pass

    class ToolCallLimitMiddleware(_Middleware):
        pass

    def create_deep_agent(**kwargs):
        calls["create_deep_agent"].append(kwargs)
        return kwargs

    patch = mock.patch.dict(sys.modules, {
        "deepagents": _module("deepagents", create_deep_agent=create_deep_agent),
        "langchain": _module("langchain"),
        "langchain.agents": _module("langchain.agents"),
        "langchain.agents.middleware": _module(
            "langchain.agents.middleware",
            TodoListMiddleware=TodoListMiddleware,
            ModelCallLimitMiddleware=ModelCallLimitMiddleware,
            ToolCallLimitMiddleware=ToolCallLimitMiddleware,
        ),
    })
    return patch, calls


class FakeBackend:
    """Records the shell commands research.py runs in the sandbox."""

    def __init__(self, executed):
        self.executed = executed

    def execute(self, command, **kwargs):
        self.executed.append(command)
        return types.SimpleNamespace(output="", exit_code=0, truncated=False)


def install_sandbox_stubs(executed=None, uploaded=None, download_result=None):
    """Make `research.py` importable without daytona/deepagents: fake model.py and sandbox.py."""
    executed = [] if executed is None else executed
    uploaded = {} if uploaded is None else uploaded

    @contextlib.contextmanager
    def _open_sandbox(*args, **kwargs):
        yield FakeBackend(executed)

    sandbox = _module("sandbox")
    sandbox.download = lambda backend, paths: {path: (download_result or {}).get(path) for path in paths}
    sandbox.upload = lambda backend, files: uploaded.update(files)
    sandbox.open_sandbox = _open_sandbox
    patch = mock.patch.dict(sys.modules, {"sandbox": sandbox,
                                          "model": _module("model", make_model=lambda: "stub-model")})
    return patch


@contextlib.contextmanager
def imported(*module_names):
    """Import the given lab modules with every third-party dependency stubbed.

        with imported("tools") as modules:
            tools = modules["tools"]
            ...                      # stubs stay installed for the whole block
    """
    executed, uploaded, download_result = [], {}, {}
    with contextlib.ExitStack() as stack:
        stack.enter_context(install_tool_stub())
        agent_patch, calls = install_agent_stubs()
        stack.enter_context(agent_patch)
        stack.enter_context(install_sandbox_stubs(executed, uploaded, download_result))
        for name in list(module_names) + ["agents", "tools"]:
            sys.modules.pop(name, None)
        modules = {name: __import__(name) for name in module_names}
        modules["_executed"] = executed
        modules["_uploaded"] = uploaded
        modules["_download_result"] = download_result
        modules["_stub_calls"] = calls
        try:
            yield modules
        finally:
            for name in list(module_names) + ["agents", "tools"]:
                sys.modules.pop(name, None)
            stack.close()  # undo the sys.modules patches
