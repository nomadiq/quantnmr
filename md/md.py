import numpy as np
import MDAnalysis as mda
import matplotlib.pyplot as plt
import matplotlib
import sys

from MDAnalysis.analysis import align, rms, dihedrals

from MDAnalysis.analysis.rms import rmsd

from numpy.linalg import norm

import time



def calc_phi_angles(trajectory_list: list, system_list: list):
    '''calculate the phi angle of the CG-SD bond in methionines'''
    from MDAnalysis.analysis import dihedrals
    n_traj = len(trajectory_list)
    n_syst = len(system_list)
    n_frames = trajectory_list[0].trajectory.n_frames
    angles = np.zeros((n_frames, n_traj, n_syst))
    for i in range(n_traj):
        print(f'traj. #{i}')
        for j, s in enumerate(system_list):
            atoms_sel = trajectory_list[i].select_atoms(f"resnum {s} and (name CB or name CG or name SD or name CE)")
            phi_dihedral = dihedrals.Dihedral([atoms_sel])
            a=phi_dihedral.run().results.angles
            for k, aa in enumerate(a):
                angles[k, i, j]  = aa[0]
    return angles
    

def calc_theta_angles(trajectory_list: list, system_list: list):
    '''calculate the theta angle of the CG-SD-CE bond in methionines'''
    n_traj = len(trajectory_list)
    n_syst = len(system_list)
    n_frames = trajectory_list[0].trajectory.n_frames
    angles = np.zeros((n_frames, n_traj, n_syst))
    for i in range(n_traj):
        print(f'traj. #{i}')
        for j, s in enumerate(system_list):
            atoms_sel = trajectory_list[i].select_atoms(f"resnum {s} and (name CE or name SD or name CG)")
            for k, ts in enumerate(trajectory_list[i].trajectory):
                theta_atoms = atoms_sel
                angles[k, i, j]  = theta_atoms.angle.value()
    return angles  

def thetaphi2uv(θ, φ):
    fac = np.pi / 180
    return np.sin(fac*(180 - θ))*np.cos(fac*φ), np.sin(fac*(180 - θ))*np.sin(fac*φ), np.cos(fac*(180 - θ)) 


def calculate_Ct_Palmer(vecs):
    """
    Definition: < P2( v(t).v(t+dt) )  >
    (Rewritten) This proc assumes vecs to be of square dimensions ( nReplicates, nFrames, nResidues, 3).
    Operates a single einsum per delta-t timepoints to produce the P2(v(t).v(t+dt)) with dimensions ( nReplicates, nResidues )
    then produces the statistics from there according to Palmer's theory that trajectory be divide into N-replcates with a fixed memory time.
    Output Ct and dCt should take dimensions ( nResidues, nDeltas )
    """
    sh = vecs.shape
    #print "= = = Debug of calculate_Ct_Palmer confirming the dimensions of vecs:", sh
    if sh[1]<50:
        print >> sys.stderr,"= = = WARNING: there are less than 50 frames per block of memory-time!"

    if len(sh)!=4:
        # Not in the right form...
        #print >> sys.stderr, "= = = ERROR: The input vectors to calculate_Ct_Palmer is not of the expected 4-dimensional form! " % sh
        sys.exit(1)
    nReplicates=sh[0] ; nDeltas=sh[1]//2 ; nResidues=sh[2]
    Ct_ind = np.zeros( (nReplicates, nDeltas,nResidues) )
    Ct  = np.zeros( (nDeltas,nResidues) )
    dCt = np.zeros( (nDeltas,nResidues) )
    bFirst=True
    for delta in range(1,1+nDeltas):
        nVals=sh[1]-delta
        # = = Create < vi.v'i > with dimensions (nRep, nFr, nRes, 3) -> (nRep, nFr, nRes) -> ( nRep, nRes ), then average across replicates with SEM.
        tmp = -0.5 + 1.5 * np.square( np.einsum( 'ijkl,ijkl->ijk', vecs[:,:-delta,...] ,vecs[:,delta:,...] ) ) #(nRep, nFr, nRes, 3) -> (nRep, nFr, nRes) over -delta to delta frames
        tmp  = np.einsum( 'ijk->ik', tmp ) / nVals # (nRep, nFr, nRes) -> ( nRep, nRes )
        Ct_ind[:, delta-1, :] = tmp
        Ct[delta-1]  = np.mean( tmp,axis=0 ) #( nRep, nRes ) -> ( nRes )
        dCt[delta-1] = np.std( tmp,axis=0 ) / ( np.sqrt(nReplicates) - 1.0 )
        #if bFirst:
        #    bFirst=False
        #    print tmp.shape, P2.shape
        #    print tmp[0,0,0], P2[0,0]
        #Ct[delta-1]  = np.mean( tmp,axis=(0,1) )
        #dCt[delta-1] = np.std( tmp,axis=(0,1) ) / ( np.sqrt(nReplicates*nVals) - 1.0 )

    #print "= = Bond %i Ct computed. Ct(%g) = %g , Ct(%g) = %g " % (i, dt[0], Ct_loc[0], dt[-1], Ct_loc[-1])
    # Return with dimensions ( nDeltas, nResidues ) by default.
    return Ct_ind, Ct, dCt



def calc_unit_vectors(trajectory_list: list, resname: str='MET', atoms: list=['SD', 'CE']):
    n_traj = len(trajectory_list)
    n_frames = trajectory_list[0].trajectory.n_frames
    selection_string = f'resname {resname}'
    selection_systems = trajectory_list[0].select_atoms(selection_string)
    n_systems = selection_systems.n_residues
    system_idx = sorted(set(selection_systems.resids))
    print(f'Residue type: {resname} found at {system_idx} and there are {n_systems} of them.')
    #norm_bond_vectors = np.zeros((n_traj, n_frames, n_systems, 3))
    norm_bond_vectors = np.zeros((n_frames, n_traj, n_systems, 3))
    for n, u in enumerate(trajectory_list):
        selection_0 = u.select_atoms(f'resname {resname} and name {atoms[0]}')
        selection_1 = u.select_atoms(f'resname {resname} and name {atoms[1]}')
        for i, ts in enumerate(u.trajectory):
            # generate bond vectors
            
            position_0 = selection_0.positions
            position_1 = selection_1.positions
            ts_bond_vectors = position_0 - position_1 # num_systems selected x 3 (x, y, z)
        
            for j, a in enumerate(ts_bond_vectors):
                norm_bond_vectors[i, n, j, :] = a/np.linalg.norm(a)
        
            
    return norm_bond_vectors