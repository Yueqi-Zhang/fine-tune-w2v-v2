import tempfile
import multiprocessing
import logging
import random
import string
import os
import math
import pickle as pkl
from queue import Queue
import time
import shutil
import numpy as np

#from utils import logging_set
import heapq
import threading

def data_reader(data_path):
    """  read data
    
    Args:
        data_path: 

    Returns:
        (list) data
    """
    with open(data_path, 'rb') as fin:
        #obj = pkl.load(fin)
        obj = np.load(fin)
    return obj

def data_writer(data, output_path):
    """ write data
    
    Args:
        data: list
        output_path: path to write data to
        
    Returns:
        None. Write data to output_path

    """
    with open(output_path, 'wb') as fout:
        #pkl.dump(data, fout, protocol=pkl.HIGHEST_PROTOCOL)
        np.save(fout, data)
    logging.info("data writted to %s" % output_path)


def get_tmp_path(tmp_dir):
    tmp_path = os.path.join(tmp_dir, ''.join(random.sample(string.ascii_letters + string.digits, 16)))
    while os.path.exists(tmp_path):
        tmp_path = os.path.join(tmp_dir, ''.join(random.sample(string.ascii_letters + string.digits, 16)))
    return tmp_path

class SingleProcessSorter(object):
    """ You should define this customized class according to your data
    Note: the key and data_writer can be passed through the parameters because lambda function is not pickleable, which is requeired by multithreading.Pool
    
    """
    def __init__(self):
        #self.key = lambda tup: tup[0]
        self.data_writer = data_writer

    def __call__(self, msg, data, tmp_path):
        #logging.info("%s start" % msg)
        print("%s start" % msg)
        """
        if self.key:
            sorted(data, key=self.key)
        else:
            sorted(data)
        """
        sorted_data = sorted(data, key=lambda tup: tup[0])
        self.data_writer(sorted_data, tmp_path)
        print("%s end" % msg)
        #logging.info("%s end" % msg)

file_lock = threading.Lock()        # lock for file_queue and files_merging
files_for_merge_lock = threading.Lock()

class FileMerger(object):
    def __init__(self):
        pass

    def __call__(self, files_for_merge, file_queue, files_merging, tmp_path):
        """

        Args:
            files_for_merge: 
            tmp_path: 

        Returns:

        """
        files = []
        files_for_merge_lock.acquire()
        while not files_for_merge.empty():
            files.append(files_for_merge.get())
        files_for_merge_lock.release()

        merged_data = heapq.merge(*files, key=self.key)
        for file in files:
            os.remove(file)
        self.tmp_data_writer(merged_data, tmp_path)

        file_lock.acquire()
        files_merging.remove(tmp_path)  # release lock for tmp_path
        file_queue.put(tmp_path)
        file_lock.release()



class ExternalSorter(object):
    def __init__(self, data, key=None, output_path=None, external_tmp_dir=None, nthreads=1, split_size=100000, nsplits_for_merge=10,
                 tmp_data_reader=None, tmp_data_writer=None):
        """ external merge sort.
        
        Args:
            data: list
            key: can be None or a lambda expression 
            external_tmp_dir: save the tmp files
        """
        self.tmp_files = []
        self.data = data
        self.key=key
        self.output_path = output_path
        self.external_tmp_dir = external_tmp_dir
        self.nthreads = nthreads
        self.split_size = split_size
        self.nsplits_for_merge = nsplits_for_merge
        self.tmp_data_reader = tmp_data_reader if tmp_data_reader is not None else data_reader
        self.tmp_data_writer = tmp_data_writer if tmp_data_writer is not None else data_writer

    def sort(self):
        """
        
        Args:
            
        Returns:

        """

        pool = multiprocessing.Pool(processes=self.nthreads)
        for i in range(0, len(self.data), self.split_size):
            #logging.info("Index: %d/%d start" % (i, len(self.data)))
            tmp_path = get_tmp_path(self.external_tmp_dir)
            self.tmp_files.append(tmp_path)
            #pool.apply_async(self.single_sort, ("Sort index: %d/%d" % (i, len(self.data)), self.data[i:i+self.split_size], self.key, tmp_path, self.tmp_data_writer))
            #pool.apply(self.single_sort, ("Sort index: %d/%d" % (i, len(self.data)), self.data[i:i+self.split_size], tmp_path))
            pool.apply_async(SingleProcessSorter(), ("Sort index: %d/%d" % (i, len(self.data)), self.data[i:i+self.split_size], tmp_path))
            #logging.info("Index: %d/%d finished" % (i, len(self.data)))

        pool.close()
        pool.join()

        logging.info("all the sortings done! start merging")
        #return self.merge()
        return self.simple_merge()

    def single_sort(self, msg, data, tmp_path):
        """ sort a small part of data, then write the sorted data to tmp_path

        Args:
            data: 
            key: 
            tmp_path: 
            data_writer: 

        Returns:

        """
        logging.info("%s start" % msg)
        print("%s start" % msg)
        sorted(data, key=self.key)
        self.tmp_data_writer(data, tmp_path)
        print("%s end" % msg)
        logging.info("%s end" % msg)

    def simple_merge(self):
        tmp_data = []
        pool = multiprocessing.Pool(processes=self.nthreads)
        for tmp_file in self.tmp_files:
            tmp_data.append(pool.apply_async(data_reader, [tmp_file]).get())
            #tmp_data.append(self.tmp_data_reader(tmp_file))
        pool.close()
        pool.join()
        merged_data = heapq.merge(*tmp_data, key=self.key)
        if self.output_path:
            self.tmp_data_writer(list(merged_data), self.output_path)
        return merged_data

    def merge(self):
        """

        Returns:
            if output_path is None:
                return sorted data directly and writing data to output_path
            else:
                return None, but write data to output_path
        """
        file_queue = []
        for file in self.tmp_files:
            file_queue.put(file)

        pool = multiprocessing.Pool(processes=self.nthreads)
        files_merging = set()   # the files are merging now
        

        files_for_merge = []
        while True:
            #self.file_lock.acquire()
            for i in range(self.nsplits_for_merge):
                if not file_queue.empty():
                    files_for_merge_lock.acquire()
                    files_for_merge.put(file_queue.get())
                    files_for_merge_lock.release()
                else:
                    break
            #self.file_lock.release()

            if files_for_merge.qsize() > 1:
                tmp_path = get_tmp_path(self.external_tmp_dir)
                files_merging.add(tmp_path) # add lock for tmp_path
                #pool.apply(FileMerger(), (files_for_merge, file_queue, files_merging, tmp_path))
                pool.apply(FileMerger(), (tmp_path, tmp_path, tmp_path, tmp_path))

            file_lock.acquire()
            if file_queue.empty() and len(files_merging) == 0:
                # now, there would be only 1 file left in files_for_merge, which is just the merge results for all the data
                file_lock.release()
                output_path = files_for_merge.get()
                if self.output_path:
                    shutil.move(output_path, self.output_path)
                break
            else:
                file_lock.release()
        pool.close()
        pool.join()

        if self.output_path:
            return self.tmp_data_reader(self.output_path)
        else:
            return self.tmp_data_reader(output_path)


    def part_merge(self, files_for_merge, file_queue, files_merging, tmp_path):
        """
        
        Args:
            files_for_merge: 
            tmp_path: 

        Returns:

        """
        files = []
        files_for_merge_lock.acquire()
        while not files_for_merge.empty():
            files.append(files_for_merge.get())
        files_for_merge_lock.release()

        merged_data = heapq.merge(*files, key=self.key)
        for file in files:
            os.remove(file)
        self.tmp_data_writer(merged_data, tmp_path)

        file_lock.acquire()
        files_merging.remove(tmp_path)      # release lock for tmp_path
        file_queue.put(tmp_path)
        file_lock.release()


if __name__ == "__main__":
    #logging_set('./external_sort.log')
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    logging.basicConfig(filename='./external_sort.log', filemode='w',
        format='%(asctime)s %(levelname)s %(filename)s %(funcName)s %(lineno)d: %(message)s',
        level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s %(filename)s %(funcName)s %(lineno)d: %(message)s'))
    logging.getLogger().addHandler(console)

    random.seed(0)
    #data = [(4, 6, 3), (1, 4, 1), (1, 3, 5), (2, 9, 1), (5, 1, 4)]
    data = []
    for i in range(10000):
        sample = []
        for j in range(3):
            sample.append(random.randint(0, 100))
        data.append(sample)

    external_tmp_dir = os.path.join(tempfile.gettempdir(), 'external_sort')
    if not os.path.isdir(external_tmp_dir):
        os.makedirs(external_tmp_dir)
    sorter = ExternalSorter(data, key=lambda tup: tup[0], output_path='merged.pkl', external_tmp_dir=external_tmp_dir, nthreads=8, split_size=100)
    pairs_sorted = sorter.sort()
