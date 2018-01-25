from horton import *
import numpy as np
import pyci
from optimizer.optimizer.pyciminimizer import *


def test_optimizer():
    name = 'he'
    approx = 'lrmu'
    fname = '%s_qz_%s.dat' % (name, approx)
    numbers = np.array([2])
    coordinates = np.array([[0.,0.,0.]])
    obasis = get_gobasis(coordinates, numbers, 'aug-cc-pvqz')

    mol = IOData(numbers=numbers, coordinates=coordinates, obasis=obasis)
    optimizer = PyCIVariationalOptimizer(mol, obasis, 2, name, approx)
    optimizer.set_mu(1.0)
    cpar = 1.92
    apar = 1.5
    optimizer.compute_energy_lr_muspace([cpar, apar])

test_optimizer()
