from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB.DSSP import DSSP
from progress.bar import Bar
from collections import defaultdict
import json
import numpy as np 
import pandas as pd 
import os  
from multiprocessing import Pool 

# PDB files do not contain information about the atomic masses, so they have to be inferred from the atom names. 
atomic_masses = {
    'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'S': 32.06, 'P': 30.974
}
backbone_atoms = ['C', 'CA', 'N', 'O'] 
# Amino acid dictionary (3-letter to 1-letter code)
amino_acids = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
               'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
               'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
               'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
# H => alpha helix, B => beta bridge, E => strand, G => helix-3, I => helix-5, T => turn, S => bend
dssp_codes = {'H': 'helix', 'B': 'sheet', 'E': 'sheet', 'G': 'helix', 'I': 'helix', 'T': 'coil', 'S': 'coil'}
# Circular dichroism has limited resolution, so it does not distinguish the different types of helices, so I will lump all helices types into a single category. 


# MARK: utils
def parse_pdb(parser, file: str, ignore_hydrogens=False, backbone_only=False, Calpha_only=False):
    """
    Parses the PDB file (passed as a string) using Bio.PDB parser into an array of atomic masses and an array of coordinates. Option to ignore hydrogen atoms. 
    Input: 
        parser: PDBParser object
        file: PDB file path as a string
        ignore_hydrogens: boolean, if True ignores hydrogen atoms
        backbone_only: boolean, if True only considers backbone atoms
        Calpha_only: boolean, if True only considers alpha carbon atoms
    Output:
        masses: ndarray of atomic masses
        coords: ndarray of coordinates 
    """
    structure = parser.get_structure('protein', file) 
    masses = []
    coords = []

    for atom_name in structure.get_atoms():
        atom = atom_name.element
        if ignore_hydrogens and atom == 'H': continue 
        if backbone_only and atom_name.name not in backbone_atoms: continue
        if Calpha_only and atom_name.name != 'CA': continue
        masses.append(atomic_masses.get(atom, 0))
        coords.append(atom_name.coord)  
    return np.array(masses), np.array(coords)

def get_calpha_only(parser, pdb_file: str) -> np.ndarray: 
    """Does the same as the function above but specific for Calpha, just to speed things up."""
    structure = parser.get_structure('protein', pdb_file)
    calpha_coords = [
        atom.coord for atom in structure.get_atoms() if atom.get_name() == 'CA'
    ]
    return np.array(calpha_coords)

def residue_distance(residue_i, residue_j):
    diff = residue_i["CA"].coord - residue_j["CA"].coord 
    return np.linalg.norm(diff)

def pdb_sequence(parser, file: str):
    structure = parser.get_structure('protein', file)
    sequence = [amino_acids[res.get_resname()] for res in structure.get_residues()]
    return ''.join(sequence)

def print_sequence(root: str): 
    parser = PDBParser(QUIET=True)
    pdb_files = os.listdir(root)
    pdb_paths = [ f"./{root}/{file}" for file in pdb_files]
    sequence = pdb_sequence(parser, pdb_paths[0])
    with open('sequence.txt', 'w') as f:
        f.write(sequence)

def get_pdb_files(root: str):
    pdb_files = os.listdir(root)
    pdb_paths = [ f"./{root}/{file}" for file in pdb_files]
    files_ids = [ f.split('.')[0] for f in pdb_files]
    return pdb_paths, files_ids

def calculate_mass(root: str, vol: float | None = None):
    """
    Computes the total mass of the protein from the PDB file and optionally calculates the concentration if volume is provided. 
    Volume is in nm^3. Concentration output in mg/ml
    """
    parser = PDBParser(QUIET=True)
    pdb_paths, _ = get_pdb_files(root)
    masses, _ = parse_pdb(parser, pdb_paths[0], ignore_hydrogens=False)
    total_mass = np.sum(masses)
    print(f"Total mass of the protein: {total_mass:.2f} g/mol")
    if vol: 
        # 1 mol = 6.02214076e23 molecules
        # 1 nm^3 = 1e-21 ml
        # 1 g = 1000 mg 
        concentration = (total_mass / 6.02214076e23) * 1.0e3 / (vol * 1.0e-21)
        print(f"Concentration: {concentration:.2f} mg/ml")   

def batch_renaming(path: str, prefix: str):
    files = os.listdir(path)
    for file in files: 
        file_id = file.split('.')[0].split('_')[-1]
        extension = file.split('.')[-1]
        new_name = f"{prefix}_{file_id}.{extension}"  
        old_path = os.path.join(path, file)
        new_path = os.path.join(path, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed {file} to {new_name}")

# MARK: Rg
def center_of_mass(masses, coordinates):
    total_mass = np.sum(masses)
    com = np.zeros(3) 
    for i in range(len(coordinates)): 
        com += coordinates[i] * masses[i]
    com /= total_mass
    return com    

def radius_of_gyration(masses, coordinates, com): 
    total_mass = np.sum(masses)
    rg = 0.0  
    for i in range(len(coordinates)): 
        rg += masses[i] * np.linalg.norm(coordinates[i] - com)**2
    rg = np.sqrt(rg / total_mass)   
    return rg 

def ensemble_rg(root: str, ignore_hydrogens=False, filename: str | None = None):
    """
    Computes Rg for each of the PDB files in the directory and saves the results in a CSV file. 
    """
    parser = PDBParser(QUIET=True)
    pdb_paths, files_ids = get_pdb_files(root)
    directory = root.split('/')[-1]

    rg_all = []
    bar = Bar(f'Processing Rg for {directory}...', max = len(pdb_paths))
    for pdb_path in pdb_paths: 
        masses, coords = parse_pdb(parser, pdb_path, ignore_hydrogens)
        com = center_of_mass(masses, coords)
        rg = radius_of_gyration(masses, coords, com)
        rg_all.append(rg)
        bar.next()
    bar.finish()
    
    df = pd.DataFrame({'Rg': rg_all, 'file': files_ids})
    if not filename: 
        filename = f"./results/{directory}/rg.csv"

    df.to_csv(filename, index=False, mode='w', header=True)


# MARK: RMSD methods
def kabsch_rmsd(P, Q): 
    # Number of atoms 
    N = P.shape[0]
    # Translation 
    P = P - np.mean(P, axis=0)
    Q = Q - np.mean(Q, axis=0)
    # Rotation aligment 
    C = np.dot(np.transpose(P), Q)
    V, S, W = np.linalg.svd(C)
    d = np.linalg.det(V) * np.linalg.det(W)
    if d < 0:
        V[:, -1] *= -1
        S[-1] *= -1 
    # Rotation matrix 
    U = np.dot(V, W)  
    # Apply rotation to P 
    P = np.dot(P, U)
    # Calculate difference
    diff = P - Q    
    return np.sqrt(np.sum(diff**2) / N)

def distance_rmsd(ref_coords, structure_coords):
    """
    Calculates the distance RMSD (dRMSD) between two pairs of coordinates.
    """
    # Compute pairwise distances for reference and structure coordinates
    # Note: None adds a new axis to the array, so A[:, None, :] with initial shape (n, 3) is transformed into (n, 1, 3). 
    # Subtracting the arrays with shape (n, 1, 3) and (1, n, 3), NumPy automatically broadcasts them into an array (n, n, 3). 
    # axis = -1 specifies that the operation is to be performed along the last axis. 
    ref_distances = np.linalg.norm(ref_coords[:, None, :] - ref_coords[None, :, :], axis=-1)
    structure_distances = np.linalg.norm(structure_coords[:, None, :] - structure_coords[None, :, :], axis=-1)

    # Compute the squared differences between the distance matrices
    dist_squared_diff = (ref_distances - structure_distances) ** 2

    n = len(ref_coords)
    n_pairs = n * (n - 1) / 2
    d_rmsd = np.sqrt(np.sum(dist_squared_diff) / n_pairs)

    return d_rmsd

def distance_rmsd_wrapper(args) -> tuple[int, int, float]:
    """For optimization purposes"""
    i, j, calpha_coords = args
    if i == j: return (i, j, 0)
    rmsd = distance_rmsd(calpha_coords[i], calpha_coords[j])
    return (i, j, rmsd)

def hs_score_norm(ref_coords, structure_coords, r_0=20): 
    """
    Calculates the NORMALIZED HS score between two pairs of coordinates.
    """
    # Compute pairwise distances for reference and structure coordinates
    ref_distances = np.linalg.norm(ref_coords[:, None, :] - ref_coords[None, :, :], axis=-1)
    structure_distances = np.linalg.norm(structure_coords[:, None, :] - structure_coords[None, :, :], axis=-1)

    # Compute the squared differences between the distance matrices
    dist_diff = (ref_distances - structure_distances)
    dist_sum = (ref_distances + structure_distances)
    exp_term = np.exp(- dist_sum**2 / r_0**2 )
    # fill dist_sum matrix diagonal with np.inf to avoid division by zero errors 
    np.fill_diagonal(dist_sum, np.inf)

    numerator = np.sum( np.abs(dist_diff) / dist_sum * exp_term)
    denominator = np.sum(exp_term)
    score = numerator / denominator     

    return score


# MARK:  RMSD
def calculate_rmsd(root: str, method='D', ignore_hydrogens=True, Calpha_only=False, backbone_only=True): 
    parser = PDBParser(QUIET=True)
    pdb_paths, files_ids = get_pdb_files(root)
    directory = root.split('/')[-1]

    # Reference structure 
    _, coords_ref = parse_pdb(parser, pdb_paths[0], ignore_hydrogens=ignore_hydrogens, Calpha_only=Calpha_only, backbone_only=backbone_only)        

    rmsd_all = []
    for pdb_path in pdb_paths:
        _, coords_other = parse_pdb(parser, pdb_path, ignore_hydrogens=ignore_hydrogens, Calpha_only=Calpha_only, backbone_only=backbone_only) 
        if method == 'K': 
            result = kabsch_rmsd(coords_other, coords_ref)
        elif method == 'D':
            result = distance_rmsd(coords_ref, coords_other)
        elif method == 'HS':
            result = hs_score_norm(coords_ref, coords_other)
        else: 
            raise ValueError("Invalid method. Choose 'K' (for Kabsch RMSD), 'D' (distance RMSD), or 'HS' (Holm and Sander).")
        rmsd_all.append(result)

    df = pd.DataFrame({'RMSD': rmsd_all, 'file': files_ids})
    df.to_csv(f'./results/{directory}/rmsd.csv', index=False, mode='w', header=True)

def rmsd_matrix(root: str, method='K', ignore_hydrogens=True, Calpha_only=False, backbone_only=True): 
    parser = PDBParser(QUIET=True)
    pdb_paths, files_ids = get_pdb_files(root)
    directory = root.split('/')[-1]
    # n: number of conformations
    n = len(pdb_paths)
    rmsd_all = np.zeros((n,n))
    # Reference structure 
    bar = Bar(f'Processing RMSD matrix for {directory}...', max = n)
    for i in range(0, n): 
        _, coords_ref = parse_pdb(parser, pdb_paths[i], ignore_hydrogens=ignore_hydrogens, Calpha_only=Calpha_only, backbone_only=backbone_only)
        for j in range(0, n):
            if i == j: continue
            # Avoid recalculating RMSD for the same pair
            if rmsd_all[j][i] != 0: 
                rmsd_all[i][j] = rmsd_all[j][i]
                continue
            _, coords_other = parse_pdb(parser, pdb_paths[j], ignore_hydrogens=ignore_hydrogens, Calpha_only=Calpha_only, backbone_only=backbone_only)
            if method == 'K': 
                rmsd = kabsch_rmsd(coords_other, coords_ref)
            elif method == 'D':
                rmsd = distance_rmsd(coords_ref, coords_other)
            elif method == 'HS':
                rmsd = hs_score_norm(coords_ref, coords_other)
            else: 
                raise ValueError("Invalid method. Choose 'K' (for Kabsch RMSD), 'D' (distance RMSD) or 'HS' (Holm and Sander).")
            rmsd_all[i][j] = rmsd
        bar.next()
    np.save(f"./results/{directory}/ids_rmsd_matrix.npy", files_ids)
    np.save(f"./results/{directory}/rmsd_matrix.npy", rmsd_all)

def rmsd_matrix_all(directories: list[str], num_workers: int = 8): 
    parser = PDBParser(QUIET=True)
    pdbs = []
    ids = [] 
    for d in directories:
        pdb_paths, files_ids = get_pdb_files(f"./conformations/{d}")
        pdbs += pdb_paths
        ids += files_ids
    n = len(pdbs)
    rmsd_all = np.zeros((n,n))
    # get all coordinates
    calpha_coords = [get_calpha_only(parser, pdb) for pdb in pdbs]
    # Arguments for parallel processing
    args = [(i, j, calpha_coords) for i in range(n) for j in range(i + 1, n)]
    bar = Bar(f'Processing RMSD matrix for all directories...', max = len(args))
    with Pool(num_workers) as pool: 
        for i, j, rmsd in pool.imap_unordered(distance_rmsd_wrapper, args):
            rmsd_all[i][j] = rmsd
            rmsd_all[j][i] = rmsd
            bar.next()
    bar.finish()
    np.save(f"./results/ids_rmsd_matrix_all.npy", ids)
    np.save(f"./results/rmsd_matrix_all.npy", rmsd_all)
     



# MARK: end-to-end distance 
def calculate_end_to_end(root: str):
    parser = PDBParser(QUIET=True)
    pdb_paths, files_ids = get_pdb_files(root)
    directory = root.split('/')[-1]

    end_to_end_all = np.zeros((len(pdb_paths),))
    bar = Bar(f'Processing end-to-end distance for {directory} ...', max = len(pdb_paths))
    for i, pdb_path in enumerate(pdb_paths):
        structure = parser.get_structure('protein', pdb_path)
        coords = np.array([atom.get_coord() for atom in structure.get_atoms() if atom.get_name() == 'CA'])
        end_to_end_all[i] = np.linalg.norm(coords[0] - coords[-1])
        bar.next()
    bar.finish() 
    
    df = pd.DataFrame({'end_to_end': end_to_end_all, 'file': files_ids})
    df.to_csv(f"./results/{directory}/e2e.csv", index=False, mode='w', header=True)


# MARK: Dmax
def calculate_dmax(root: str):
    """
    Proxy for maximum intramolecular distance computed from alpha carbon pairwise distance. 
    """
    parser = PDBParser(QUIET=True)
    pdb_paths, files_ids = get_pdb_files(root)
    directory = root.split('/')[-1]
    
    dmax_all = np.zeros((len(pdb_paths),))
    bar = Bar(f'Processing Dmax for {directory} ...', max = len(pdb_paths))
    for i, pdb_path in enumerate(pdb_paths):
        structure = parser.get_structure('protein', pdb_path)
        coords = np.array([atom.get_coord() for atom in structure.get_atoms() if atom.get_name() == 'CA'])
        dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        dmax_all[i] = np.max(dist_matrix) 
        bar.next()
    bar.finish()

    df = pd.DataFrame({'Dmax': dmax_all, 'file': files_ids})
    df.to_csv(f"./results/{directory}/dmax.csv", index=False, mode='w', header=True)

# MARK: contact map
def contact_matrix(root: str, threshold=8, type='contact', filename='contacts.npy'):
    parser = PDBParser(QUIET=True)
    pdb_paths, _ = get_pdb_files(root)
    contacts_ensemble = np.zeros((219, 219))
    # looop through all structures in the ensemble, sum the binary contacts matrix in each iteration, and then divide by the number of members in the ensemble. 
    for pdb_path in pdb_paths:
        _, coords = parse_pdb(parser, pdb_path, ignore_hydrogens=True, Calpha_only=True, backbone_only=False)
        distance = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        # True contact map (binary)
        if type == 'contact':
            contacts = (distance < threshold).astype(float)
        # Distance map 
        else:     
            mask = distance < threshold 
            contacts = np.where(mask, distance, 0)
        contacts_ensemble += contacts
    contacts_ensemble /= len(pdb_paths)
    np.save(filename, contacts_ensemble)

# MARK: DSSP
def percent_structure(root: str, num_residues=219, structure_type='helix'):
    """
    Using DSSP, determines whether a residue belongs to a helix or not. Then transforms the result into a binary array where 1 = helix and 0 = not helix, sums values across columns and divides by the number of structures in the ensemble to obtain the probability that a residue is helical.
    """
    # Validation 
    if structure_type not in ['helix', 'sheet', 'coil']:
        raise ValueError("Invalid structure type. Choose 'helix', 'sheet', or 'coil'.")

    parser = PDBParser(QUIET=True)
    pdb_paths, _ = get_pdb_files(root)
    directory = root.split('/')[-1]
    n = len(pdb_paths)
    structure_list = np.zeros((num_residues, n)) # storing per-residue helicity/sheet for all files
    
    bar = Bar(f'Processing for {directory}...', max = n)
    # Building the binary array num_residues x n where n is the number of structures  
    for i, pdb_path in enumerate(pdb_paths):
        structure = parser.get_structure('protein', pdb_path)
        dssp = DSSP(structure[0], pdb_path)
        for key in dssp.keys(): 
            code = dssp[key][2]
            ss = dssp_codes[code] if code in dssp_codes else 'coil'
            resid = dssp[key][0]
            if ss == structure_type: 
                structure_list[resid-1, i] = 1
                # print(f"Residue {resid} is a {structure_type} in {pdb_path}")
        bar.next()
    
    bar.finish()
    # Sum columns
    structure_frequency = np.sum(structure_list, axis=1) / n

    # Saving the results
    df = pd.DataFrame({structure_type: structure_frequency, 'Residue': range(1, num_residues + 1)})
    df.to_csv(f"./results/{directory}/{structure_type}_percent.csv", index=False, mode='w', header=True)


def secondary_structure(root: str):
    parser = PDBParser(QUIET=True)
    pdb_paths, files_ids = get_pdb_files(root)
    directory = root.split('/')[-1]

    n = len(pdb_paths)
    helix_all = np.zeros((n,)) 
    sheets_all = np.zeros((n,))
    coil_all = np.zeros((n,))

    bar = Bar(f'Processing DSSP for {directory}...', max = n)

    for i, pdb_path in enumerate(pdb_paths):

        structure = parser.get_structure('protein', pdb_path)
        dssp = DSSP(structure[0], pdb_path)

        counts = {'helix': 0, 'sheet': 0, 'coil': 0}
    
        for key in dssp.keys(): 
            code = dssp[key][2]
            if code in dssp_codes: 
                ss = dssp_codes[code]
                counts[ss] += 1
            else: 
                counts['coil'] += 1 
        total = counts['helix'] + counts['sheet'] + counts['coil']

        # convert to percentage 
        counts_pct = {k: 100*v/total for k, v in counts.items()}
        helix_all[i] = counts_pct['helix']
        sheets_all[i] = counts_pct['sheet']
        coil_all[i] = counts_pct['coil']
        bar.next()
    bar.finish()
    df = pd.DataFrame({'helix': helix_all, 'sheet': sheets_all, 'coil': coil_all, 'file': files_ids})
    df.to_csv(f"./results/{directory}/dssp.csv", index=False, mode='w', header=True)

# MARK: Dihedrals
def dihedral_angles(parser: PDBParser, pdb_path: str):
    structure = parser.get_structure('protein', pdb_path)
    ppb = PPBuilder() 
    angles = []
    for pp in ppb.build_peptides(structure):
        phi_psi = pp.get_phi_psi_list()
        for res, (phi, psi) in zip(pp, phi_psi): 
            residue_number = res.get_id()[1]
            phi_deg = np.degrees(phi) if phi else None 
            psi_deg = np.degrees(psi) if psi else None
            angles.append((residue_number, phi_deg, psi_deg))
    
    return angles 

def ensemble_dihedral(root: str):
    parser = PDBParser(QUIET=True)
    pdb_paths, file_ids = get_pdb_files(root)
    directory = root.split('/')[-1]
    
    # Dictionaries to store phi and psi angles for each residue across files
    # Key: residue number, Value: angle list, with one value per file 
    phi_dict = defaultdict(list)
    psi_dict = defaultdict(list)
    
    bar = Bar(f'Processing dihedrals for {directory}...', max=len(pdb_paths))
    for pdb_path in pdb_paths:
        angles_list = dihedral_angles(parser, pdb_path)
        for res_num, phi, psi in angles_list:
            phi_dict[res_num].append(phi)
            psi_dict[res_num].append(psi)
        bar.next()
    bar.finish()
    
    # Convert dictionaries to data frames
    phi_df = pd.DataFrame.from_dict(phi_dict, orient='index', columns=file_ids)
    psi_df = pd.DataFrame.from_dict(psi_dict, orient='index', columns=file_ids)
    
    # Set residue numbers as the index
    phi_df.index.name = 'Residue'
    psi_df.index.name = 'Residue'
    
    phi_df.to_csv(f"./results/{directory}/phi.csv", index=True, mode='w', header=True)
    psi_df.to_csv(f"./results/{directory}/psi.csv", index=True, mode='w', header=True)

# MARK: Scores 
def ensemble_plddt(root: str, num_residues=219, filename="plddt.csv"):
    score_files = os.listdir(root) # List of score files, e.g. dr_378.json, dr_379.json, etc. 
    score_ids = [ f.split('.')[0] for f in score_files] # List of score file ids, e.g. dr_378, dr_379, etc.
    scores_path = [ f"./{root}/{file}" for file in score_files] # full path for the open function
    directory = root.split('/')[-1] # name of the directory, e.g. shallow_recycle
    n = len(scores_path) # number of score files

    plddt_list = np.zeros((num_residues, n)) # storing per-residue plddt scores for all files

    bar = Bar(f'Processing pLDDT for {directory}...', max = n)
    for i in range(len(scores_path)):
        plddt = json.load(open(scores_path[i]))['plddt']
        plddt_list[:, i] = plddt
        bar.next()
    bar.finish()

    df = pd.DataFrame(plddt_list, columns=score_ids)
    df.index.name = 'Residue'
    df.to_csv(f"./results/{directory}/{filename}", index=True, mode='w', header=True)


if __name__ == "__main__": 
    # calculate_mass(root_dir, vol=892.06)

    directories = ['shallow_recycle', 'deep_recycle', 'shallow_dropout', 'deep_dropout']

# SECTION: Pairwise RMSD using Calpha atoms 
    drmsd = False 
    if drmsd: [ rmsd_matrix(f"./conformations/{d}", method='D', ignore_hydrogens=True, Calpha_only=True, backbone_only=False) for d in directories ]

# !SECTION

# SECTION: Radius of gyration 
    rg = False 
    if rg: [ ensemble_rg(f"./conformations/{d}", ignore_hydrogens=False) for d in directories ]
# !SECTION
    
# SECTION:  D_MAX
    d_max = False 
    if d_max: [ calculate_dmax(f"./conformations/{d}") for d in directories ]
#!SECTION 

# SECTION: END-TO-END
    end_to_end = False
    if end_to_end: [ calculate_end_to_end(f"./conformations/{d}") for d in directories ]
#!SECTION 


# SECTION: DSSP
    dssp = False
    if dssp: [ secondary_structure(f"./conformations/{d}") for d in directories ]
#!SECTION 

# SECTION: DIHEDRALS 
    dihedrals = False
    if dihedrals: [ ensemble_dihedral(root=f"./conformations/{d}") for d in directories ]
#!SECTION 


# SECTION: RENAMING 
    rename = False 
    if rename: [ batch_renaming(f"./conformations/{d}", 
                prefix=d.split('_')[0][0] + d.split('_')[-1][0]) 
                for d in directories 
                ]
# !SECTION 

# SECTION: PLDDT
    plddt = False 
    if plddt: [ ensemble_plddt(root=f"./scores/{d}") for d in directories ]
#!SECTION


#SECTION: Helicity/Sheets
    helicity = False
    if helicity: [ percent_structure(root=f"./conformations/{d}", num_residues=219, structure_type='helix') for d in directories ]
#!SECTION


#SECTION: others...
    others = False 
    protein = "prta"
    if others: 
        ensemble_rg(root=f"./others/{protein}/deep_recycle", ignore_hydrogens=False, filename=f"./others/{protein}_rg_deep_recycle.csv")
        ensemble_rg(root=f"./others/{protein}/shallow_dropout", ignore_hydrogens=False, filename=f"./others/{protein}_rg_shallow_dropout.csv") 

# SECTION: full matrix
    full_mat = True
    if full_mat: rmsd_matrix_all(directories)
#!SECTION