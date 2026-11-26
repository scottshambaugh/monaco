# dask_plugin.py
from dask.distributed import WorkerPlugin
from monaco.globals import _worker_init


__all__ = [
    "GlobalsPlugin",
]


class GlobalsPlugin(WorkerPlugin):
    def __init__(self, invars_blob, outvars_blob, constvals_blob):
        self.invars_blob = invars_blob
        self.outvars_blob = outvars_blob
        self.constvals_blob = constvals_blob

    def setup(self, worker):
        _worker_init(self.invars_blob, self.outvars_blob, self.constvals_blob)
