import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.metrics.pairwise import pairwise_kernels
from functools import lru_cache
import scipy as sp
import logging
import os
from datetime import datetime, timedelta

DEFAULT_FORMATTER = logging.Formatter(
                '%(asctime)s %(levelname)s [%(name)s:%(lineno)d]=> %(message)s', "%b-%d %H:%M:%S")

def apply_file_handler(logger, path):
    logger.handlers = []
    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(DEFAULT_FORMATTER)
    logger.addHandler(file_handler)
    return logger

def get_logger(name, level = None, handler = None):
    logger = logging.getLogger(name)
    logger.propagate = False
    if not logger.handlers:
        if handler is None:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(DEFAULT_FORMATTER)
            logger.addHandler(stream_handler)
        else:
            logger.addHandler(handler)
        logger.setLevel(level or os.environ.get("LOG_LEVEL", "INFO"))
    return logger

def split_dates(start, end, intv, fmt = "%Y-%m-%d"):
    start = datetime.strptime(start,fmt)
    end = datetime.strptime(end,fmt)
    step = timedelta(days=intv)
    curr = start
    while curr < end:
        yield(curr.strftime(fmt))
        curr += step
    yield(end.strftime(fmt))

def kmeanspp(X, n_clusters, dist = 'euclidean', seed = 0):
	c_idxs = [np.random.RandomState(seed).randint(X.shape[0])]
	for _ in range(1, n_clusters):
		D_x2 = cdist(X, X[c_idxs], metric = dist).min(axis = 1) ** 2
		p_x = D_x2 / np.sum(D_x2)
		c_idxs.append(p_x.argmax())
	return c_idxs


