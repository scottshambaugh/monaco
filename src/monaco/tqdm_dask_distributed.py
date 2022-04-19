# tqdm_dask_distributed.py

"""
The below code is from Alex Papanicolaou (https://github.com/alexifm), posted in
this comment: https://github.com/tqdm/tqdm/issues/278#issuecomment-507006253
It inherits the MIT license of the tqdm project, copied in full below.

TODO: Remove this file once https://github.com/tqdm/tqdm/issues/1230 is merged

MIT License (MIT)
-----------------

Copyright (c) 2019 Alex Papanicolaou (alexifm)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import tqdm
from tornado.ioloop import IOLoop
from distributed.utils import LoopRunner, is_kernel
from distributed.client import futures_of
from distributed.diagnostics.progressbar import ProgressBar

class TqdmProgressBar(ProgressBar):
    def __init__(
        self,
        keys,
        scheduler=None,
        interval="100ms",
        loop=None,
        complete=True,
        start=True,
        **tqdm_kwargs
    ):
        super(TqdmProgressBar, self).__init__(
            keys, scheduler, interval, complete)
        self.tqdm = tqdm.tqdm(keys, **tqdm_kwargs)
        self.loop = loop or IOLoop()

        if start:
            loop_runner = LoopRunner(self.loop)
            loop_runner.run_sync(self.listen)

    def _draw_bar(self, remaining, all, **kwargs):
        update_ct = (all - remaining) - self.tqdm.n
        self.tqdm.update(update_ct)

    def _draw_stop(self, **kwargs):
        self.tqdm.close()

class TqdmNotebookProgress(ProgressBar):
    def __init__(
        self,
        keys,
        scheduler=None,
        interval="100ms",
        loop=None,
        complete=True,
        start=True,
        **tqdm_kwargs
    ):
        super(TqdmNotebookProgress, self).__init__(
            keys, scheduler, interval, complete)
        self.tqdm = tqdm.tqdm_notebook(keys, **tqdm_kwargs)
        self.loop = loop or IOLoop()

        if start:
            loop_runner = LoopRunner(self.loop)
            loop_runner.run_sync(self.listen)

    def _draw_bar(self, remaining, all, **kwargs):
        update_ct = (all - remaining) - self.tqdm.n
        self.tqdm.update(update_ct)

    def _draw_stop(self, **kwargs):
        self.tqdm.close()

def tqdm_dask(futures, **kwargs):
    notebook = is_kernel()
    futures = futures_of(futures)
    if not isinstance(futures, (set, list)):
        futures = [futures]
    if notebook:
        return TqdmNotebookProgress(futures, **kwargs)
    else:
        TqdmProgressBar(futures, **kwargs)
