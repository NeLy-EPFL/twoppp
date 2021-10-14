import multiprocessing as mp
import numpy as np
from time import sleep
from timeit import default_timer as timer
from tqdm import tqdm


if __name__ == "__main__":

    a = np.arange(100)  # np.zeros((10, 480, 736))

    def func(x):
        sleep(0.1)
        print(x)
        return x
    start = timer()
    out = []
    with mp.Pool(10) as p:
        # out = p.map(func, a)  # starts threads in random order, returns all at the same time
        # for x in p.imap_unordered(func, a):  # starts one after the other in the right order
        #     out.append(x)
        for x in tqdm(p.imap(func, a), total=len(a)):  # starts one after the other in the right order
            out.append(x)
    print(out)
    print("took {} s".format(timer() - start))

    pass
