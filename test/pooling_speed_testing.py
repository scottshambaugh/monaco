from pathos.pools import ProcessPool
from pathos.pools import ThreadPool
from pathos.pools import ParallelPool
from pathos.pools import SerialPool
from multiprocessing import Pool
import time

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print ('%r (%r, %r) %2.2f sec' % \
              (method.__name__, args, kw, te-ts))
        return result
    return timed

def slowfcn(n):
    from time import sleep
    sleep(1.0)

@timeit
def test(n,p):
    (p.map(slowfcn, range(n)))

def main():
    npool = 4
    ppool = ProcessPool(npool)
    tpool = ThreadPool(npool)
    parapool = ParallelPool(npool)
    spool = SerialPool()
    pool = Pool(npool)

    nloops = 8
    print('For Loop')
    ts = time.time()
    for i in range(nloops):
        slowfcn(i)
    te = time.time()
    print ('%r () %2.2f sec' % ('test', te-ts))
    print('ThreadPool')
    test(nloops,tpool)
    print('ParallelPool')
    test(nloops,parapool)
    print('SerialPool')
    test(nloops,spool)
    print('Pool')
    test(nloops,pool)
    print('ProcessPool')
    test(nloops,ppool)


if __name__ == '__main__':
    main()