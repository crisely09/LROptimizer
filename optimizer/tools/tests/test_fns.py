from optimizer.tools.newfunctions import *
import pyci

def test_get_sd_pyci():
    nbasis = 5
    alphas = [[0,1,3], [1,2,3], [0,1,2], [0,2,3], [0,1,3]]
    betas = [[5,6,7], [5,7,8], [6,7,8], [5,8,9], [5,6,8]]
    vec = get_ci_sd_pyci(nbasis, alphas, betas)
    assert vec == [235, 430, 455, 813, 363]

def test_get_dets_pyci():
    # Using PyCI to compute FCI energy
    na = 2
    nb = 1
    nbasis = 3
    wfn = pyci.FullCIWfn(nbasis, na, nb)
    alphas, betas = get_dets_pyci(wfn)

test_get_dets_pyci()
