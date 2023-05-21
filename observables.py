from quantumsim.functions.ansatz import *
from quantumsim.functions.min_methods import *
from quantumsim.functions.constans import *

from pennylane import qchem
from pennylane import numpy as np
import scipy.linalg as la
import scipy as sc
import sys
import pandas as pd

import yaml


'''
Regex de ejecución: python3 main.py <parametros .yml>


Lectura de parametros de un archivo YML
'''
with open(sys.argv[len(sys.argv)-1], 'r') as stream:
    try:
        params=yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)