import timeit
import logging
from logging import Logger

LOGGER = logging.getLogger(__name__)

# Printing time
def get_time():
    return timeit.timeit()

def calculate_time(t_start, t_end):
    return t_end - t_start

