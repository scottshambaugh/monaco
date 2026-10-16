# test_globals.py

import pickle
import threading

import pytest
import monaco.globals as mc_globals


@pytest.fixture
def restore_globals(monkeypatch):
    saved = mc_globals.get_global_vars()
    calls = []
    monkeypatch.setattr(mc_globals, "_set_parent_death_signal", calls.append)
    yield calls
    mc_globals.register_global_vars(*saved)


def test_worker_init(restore_globals):
    invars, outvars, constvals = {"a": 1}, {"b": 2}, {"c": 3}
    blobs = [pickle.dumps(x) for x in (invars, outvars, constvals)]
    mc_globals._worker_init(*blobs, parent_pid=123)
    assert mc_globals.get_global_vars() == (invars, outvars, constvals)
    assert restore_globals == [123]


def test_fork_worker_init(restore_globals):
    invars, outvars, constvals = {"a": 1}, {"b": 2}, {"c": 3}
    mc_globals._fork_worker_init(invars, outvars, constvals, parent_pid=456)
    registered = mc_globals.get_global_vars()
    assert all(x is y for x, y in zip(registered, (invars, outvars, constvals)))
    assert restore_globals == [456]


def test_parent_watchdog_missing_parent():
    pytest.importorskip("psutil")
    nthreads = threading.active_count()
    mc_globals._start_parent_watchdog(2**31 - 1)
    assert threading.active_count() == nthreads
