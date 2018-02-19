""" Horton wrapper to compute HF/LDA calculation"""

def compute_hf(mol, obasis, na, nb, kin, natt, er, olp):
    """
    Compute HF

    Arguments:
    ----------
    mol: IOData
        Horton object with the information of the molecule
    obasis: GOBasis
        Horton object with the basis set information.
    na/nb: int
        Number of alpha/beta electrons

    Returns:
    --------
    total_energy
    core_energy
    orbs
    """
    if na == nb:
        # Do RHF
        # Create alpha orbitals
        orb_alpha = Orbitals(obasis.nbasis)
        # Initial guess
        guess_core_hamiltonian(olp, kin + natt, orb_alpha)
        
        # Construct the restricted HF effective Hamiltonian
        external = {'nn': compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
        terms = [
            RTwoIndexTerm(kin, 'kin'),
            RTwoIndexTerm(natt, 'ne'),
            RDirectTerm(er, 'hartree'),
            RExchangeTerm(er, 'x'),
        ]
        ham = REffHam(terms, external)
        
        # Decide how to occupy the orbitals (5 alpha electrons)
        occ_model = AufbauOccModel(na)
        
        # Converge WFN with Optimal damping algorithm (ODA) SCF
        # - Construct the initial density matrix (needed for ODA).
        occ_model.assign(orb_alpha)
        dm_alpha = orb_alpha.to_dm()

        # - SCF solver
        scf_solver = EDIIS2SCFSolver(1e-6)
        scf_solver(ham, olp, occ_model, dm_alpha)
        ham.reset(dm_alpha)
        fock_alpha = np.zeros((obasis.nbasis, obasis.nbasis))
        ham.compute_fock(fock_alpha)
        orb_alpha.from_fock_and_dm(fock_alpha, dm_alpha, olp)
        orbs = orb_alpha
    else:
        # Do UHF
        # Create alpha orbitals
        orb_alpha = Orbitals(obasis.nbasis)
        orb_beta = Orbitals(obasis.nbasis)
        # Initial guess
        guess_core_hamiltonian(olp, kin + natt, orb_alpha, orb_beta)
        
        # Construct the restricted HF effective Hamiltonian
        external = {'nn': compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
        terms = [
            UTwoIndexTerm(kin, 'kin'),
            UDirectTerm(er, 'hartree'),
            UTwoIndexTerm(natt, 'ne'),
            UExchangeTerm(er, 'x'),
        ]
        ham = UEffHam(terms, external)
        
        # Decide how to occupy the orbitals (5 alpha electrons)
        occ_model = AufbauOccModel((na, nb))
        
        # Converge WFN with Optimal damping algorithm (ODA) SCF
        # - Construct the initial density matrix (needed for ODA).
        occ_model.assign(orb_alpha, orb_beta)
        dm_alpha = orb_alpha.to_dm()
        dm_beta = orb_beta.to_dm()

        # - SCF solver
        scf_solver = EDIIS2SCFSolver(1e-6)
        scf_solver(ham, olp, occ_model, dm_alpha, dm_beta)
        ham.reset(dm_alpha)
        fock_alpha = np.zeros((obasis.nbasis, obasis.nbasis))
        fock_beta = np.zeros((obasis.nbasis, obasis.nbasis))
        ham.compute_fock(fock_alpha, fock_beta)
        orb_alpha.from_fock_and_dm(fock_alpha, dm_alpha, olp)
        orb_beta.from_fock_and_dm(fock_beta, dm_beta, olp)
        orbs = (orb_alpha, orb_beta)
    core_energy = external['nn']
    total_energy = ham.cache['energy']
    return total_energy, core_energy, orbs


def compute_lda(mol, obasis, na, nb):
    """
    Compute LDA

    Arguments:
    ----------
    mol: IOData
        Horton object with the information of the molecule
    obasis: GOBasis
        Horton object with the basis set information.
    na/nb: int
        Number of alpha/beta electrons

    Returns:
    --------
    total_energy
    core_energy
    orbs
    """
    # Create grid for LDA parts
    grid = BeckeMolGrid(mol.coordinates, mol.numbers, mol.pseudo_numbers)

    if na == nb:
        # Do RLDA
        # Create alpha orbitals
        orb_alpha = Orbitals(obasis.nbasis)
        
        # Initial guess
        guess_core_hamiltonian(olp, kin + natt, orb_alpha)
        
        # Construct the restricted HF effective Hamiltonian
        external = {'nn': compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
        terms = [
            RTwoIndexTerm(kin, 'kin'),
            RDirectTerm(er, 'hartree'),
            RTwoIndexTerm(natt, 'ne'),
            RGridGroup(obasis, grid, [
            RLibXCLDA('x'),
            RLibXCLDA('c_vwn'),
        ]),
        ]
        ham = REffHam(terms, external)
        
        # Decide how to occupy the orbitals (5 alpha electrons)
        occ_model = AufbauOccModel(na)
        
        # Converge WFN with Optimal damping algorithm (ODA) SCF
        # - Construct the initial density matrix (needed for ODA).
        occ_model.assign(orb_alpha)
        dm_alpha = orb_alpha.to_dm()

        # - SCF solver
        scf_solver = EDIIS2SCFSolver(1e-6)
        scf_solver(ham, olp, occ_model, dm_alpha)
        ham.reset(dm_alpha)
        fock_alpha = np.zeros((obasis.nbasis, obasis.nbasis))
        ham.compute_fock(fock_alpha)
        orb_alpha.from_fock_and_dm(fock_alpha, dm_alpha, olp)
        orbs = orb_alpha
    else:
        # Do ULDA
        # Create alpha orbitals
        orb_alpha = Orbitals(obasis.nbasis)
        orb_beta = Orbitals(obasis.nbasis)
        
        # Initial guess
        guess_core_hamiltonian(olp, kin + natt, orb_alpha, orb_beta)
        
        # Construct the restricted HF effective Hamiltonian
        external = {'nn': compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
        terms = [
            UTwoIndexTerm(kin, 'kin'),
            UDirectTerm(er, 'hartree'),
            UTwoIndexTerm(natt, 'ne'),
            UGridGroup(obasis, grid, [
            ULibXCLDA('x'),
            ULibXCLDA('c_vwn'),
        ]),
        ]
        ham = UEffHam(terms, external)
        
        # Decide how to occupy the orbitals (5 alpha electrons)
        occ_model = AufbauOccModel((na,nb))
        
        # Converge WFN with Optimal damping algorithm (ODA) SCF
        # - Construct the initial density matrix (needed for ODA).
        occ_model.assign(orb_alpha, orb_beta)
        dm_alpha = orb_alpha.to_dm()
        dm_beta = orb_beta.to_dm()

        # - SCF solver
        scf_solver = EDIIS2SCFSolver(1e-6)
        scf_solver(ham, olp, occ_model, dm_alpha, dm_beta)
        ham.reset(dm_alpha, dm_beta)
        fock_alpha = np.zeros((obasis.nbasis, obasis.nbasis))
        ham.compute_fock(fock_alpha)
        orbs = (orb_alpha, orb_beta)
        
    core_energy = external['nn']
    total_energy = ham.cache['energy']
    return total_energy, core_energy, orbs
