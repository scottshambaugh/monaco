# Serial and Parallel Processing

Monaco supports three different processing methods for running Monte Carlo simulations: serial (single-threaded), multiprocessing, and Dask distributed computing. Different methods can be faster depending on your use case.

## Serial Processing (Single-threaded)

Serial processing runs all simulation cases sequentially in a single thread using a simple for loop. This is the most straightforward execution mode with no parallelization overhead, and is the best place to start for ease of debugging.

### Configuration

The `singlethreaded=True` parameter forces serial execution regardless of other parallel settings.

```python
sim = mc.Sim(
    name='MySimulation',
    ndraws=1000,
    fcns=my_functions,
    singlethreaded=True,  # Enable serial processing
    # Other parameters are ignored when singlethreaded=True
)
```

## Multiprocessing

Multiprocessing uses Python's built-in `multiprocessing` module with `ProcessPoolExecutor` to distribute cases across multiple CPU cores on a single machine. It provides good performance improvements for CPU-bound simulations and automatically balances the workload across available cores.

Multiprocessing is built into Python's standard library, requiring no additional dependencies.

### Configuration

Setting `singlethreaded=False` enables parallel processing, while `usedask=False` ensures multiprocessing is used instead of Dask.

```python
sim = mc.Sim(
    name='MySimulation',
    ndraws=1000,
    fcns=my_functions,
    singlethreaded=False,           # Enable parallel processing
    usedask=False,                  # Use multiprocessing instead of Dask
    ncores=4,                       # Optional: specify number of cores
    multiprocessing_method='spawn', # Optional: set start method ('fork', 'spawn', or 'forkserver')
)
```

### Start Methods

The multiprocessing start method significantly affects how worker processes are created. The `'fork'` method is the default on Linux, and provides fast startup with shared memory with the parent process (which can potentially lead to issues with unsafe memory management). The `'spawn'` method, used as the default on Windows and macOS, has slower startup but creates completely separate memory-safe processes. In python 3.14, the default on Linux will be switching to `'spawn'` to match the other platforms. The `'forkserver'` method offers a hybrid approach that can help avoid deadlocks in complex scenarios.

## Dask Distributed Computing

Dask provides distributed computing capabilities that can scale from a single machine to a cluster of machines. For local workflows, it is generally slower and offers no benefits over multiprocessing. However it does allow scaling to distributed compute clusters.

Note that Dask is not a required dependency. It can be installed with `pip install dask[distributed]`.

When using Dask, there is a nice web dashboard available at [http://localhost:8787](http://localhost:8787) to watch the progress of your simulation.

### Configuration

Setting `singlethreaded=False` enables parallel processing, and `usedask=True` selects Dask instead of multiprocessing. The `ncores` parameter overrides the `n_workers` setting in `daskkwargs` if specified. The `daskkwargs` parameter accepts a dictionary of keyword arguments that are passed directly to the Dask Client constructor.

```python
sim = mc.Sim(
    name='MySimulation',
    ndraws=1000,
    fcns=my_functions,
    singlethreaded=False,  # Enable parallel processing
    usedask=True,          # Use Dask
    ncores=4,              # Optional: overrides n_workers in daskkwargs
    daskkwargs={           # Optional: Dask client configuration
        'n_workers': 4,
        'threads_per_worker': 2,
        'memory_limit': '2GB',
        'dashboard_address': ':8787'
    }
)
```

### Distributed Cloud Compute Clusters

Dask can be configured by overwriting a sim's `client` and `cluster` attributes. The below setup builds on [the example from the dask documentation](https://docs.dask.org/en/latest/deploying.html), and will require you to first set up an account with Coiled, connected to AWS or some other cloud compute provider.

```python
import coiled
cluster = coiled.Cluster(
    n_workers=8,
    worker_memory="8 GiB",
    spot_policy="spot_with_fallback",
    wait_for_workers=True,
)
client = cluster.get_client()

sim.client = client
sim.cluster = cluster
```

## Choosing your Processing Approach

Start your development process with a single-threaded setup for ease of debugging.

Once things are working, try out using multiproccessing. Whether this is faster than single-threaded depends on the particulars of your simulation. If your simulation has a lot of input and output data, then pickling these objects and memory transfer to and from the worker processes may mean that multiprocessing is actually slower. However if your simulation is computation-heavy, then multiprocessing can give a significant performance benefit by using multiple cores.

It's worth experimenting with your specific setup. Try single-threaded, and both the `'fork'` and `'spawn'` start methods for multiprocessing to see which is faster.

If you need to scale your simulation beyond a single machine, then move to Dask. A full guide on setting up distributed computation is beyond the scope of these docs, but resources are widely available to get you started.
