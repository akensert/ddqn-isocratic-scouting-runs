import os
import sys


SRC_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(SRC_DIR)

WEIGHTS_DIR = SRC_DIR +'/' + '../output/weights/'
LOGS_DIR = SRC_DIR    +'/' + '../logs/'
OUTPUT_DIR = SRC_DIR  +'/' + '../output/'
DATA_DIR = SRC_DIR    +'/' + '../data/'
