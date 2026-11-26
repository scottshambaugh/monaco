# conftest.py
import matplotlib

matplotlib.use("Agg")


DASK_FIXTURES = {"sim_dask", "sim_dask_expanded", "dask_client"}


def _uses_dask(item):
    callspec = getattr(item, "callspec", None)
    indirect = callspec.params.get("sim_fixture") if callspec else None
    return bool(DASK_FIXTURES & set(item.fixturenames)) or indirect in DASK_FIXTURES


def pytest_collection_modifyitems(config, items):
    # The shared dask cluster keeps threads running once started, and forking a multithreaded
    # process can deadlock, so run the dask tests after all the fork-based multiprocessing tests
    items.sort(key=_uses_dask)
