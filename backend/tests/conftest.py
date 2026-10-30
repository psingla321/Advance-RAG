"""Minimal async-test support, so the suite needs no pytest-asyncio dependency.

Coroutine test functions and coroutine fixtures are run on one shared event loop
that lives for the whole session — which also means a pipeline built in a fixture
is usable from the test that receives it.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest


@pytest.fixture(scope="session")
def event_loop() -> Any:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


def _loop(request: pytest.FixtureRequest) -> asyncio.AbstractEventLoop:
    return request.getfixturevalue("event_loop")


@pytest.hookimpl(hookwrapper=True)
def pytest_fixture_setup(fixturedef: Any, request: pytest.FixtureRequest):
    """Resolve coroutine and async-generator fixtures on the shared loop."""
    func = fixturedef.func
    if not (inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func)):
        yield
        return

    loop = _loop(request)
    if inspect.isasyncgenfunction(func):
        def sync_wrapper(**kwargs: Any) -> Any:
            generator = func(**kwargs)
            value = loop.run_until_complete(generator.__anext__())
            request.addfinalizer(lambda: _finalize(loop, generator))
            return value
    else:
        def sync_wrapper(**kwargs: Any) -> Any:
            return loop.run_until_complete(func(**kwargs))

    fixturedef.func = sync_wrapper
    try:
        yield
    finally:
        fixturedef.func = func


def _finalize(loop: asyncio.AbstractEventLoop, generator: Any) -> None:
    try:
        loop.run_until_complete(generator.__anext__())
    except StopAsyncIteration:
        pass


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function):
    """Run coroutine tests on the shared loop."""
    if not inspect.iscoroutinefunction(pyfuncitem.obj):
        return None
    loop = _loop(pyfuncitem._request)
    kwargs = {name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames}
    loop.run_until_complete(pyfuncitem.obj(**kwargs))
    return True
