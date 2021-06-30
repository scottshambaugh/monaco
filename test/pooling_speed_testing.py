from pathos.pools import ProcessPool
from pathos.pools import ThreadPool
from pathos.pools import ParallelPool
from pathos.pools import SerialPool
from multiprocessing import Pool
from Monaco.helper_functions import timeit
from time import time
from test.mcsim_testing_fcns import slowfcn

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
    t0 = time()
    for i in range(nloops):
        slowfcn(i)
    t1 = time()
    print (f'"test" took {(t1 - t0)*1000 : .3f} ms to execute.\n')
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