# pooling_speed_testing.py

from pathos.pools import ProcessPool
from pathos.pools import ThreadPool
from pathos.pools import ParallelPool
from pathos.pools import SerialPool
from multiprocessing import Pool
from Monaco.helper_functions import timeit

def slowfcn(n):
    from time import sleep
    sleep(1.0)

@timeit
def test(n,p):
    (p.map(slowfcn, range(n)))

@timeit
def forloop(n):
    for i in range(n):
        slowfcn(i)

def main():
    npool = 4
    ppool = ProcessPool(npool)
    tpool = ThreadPool(npool)
    parapool = ParallelPool(npool)
    spool = SerialPool()
    pool = Pool(npool)

    nloops = 8
    print('For Loop')
    forloop(nloops)
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