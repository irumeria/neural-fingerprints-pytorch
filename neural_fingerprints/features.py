import numpy as np
from rdkit import Chem

def atom_features(atom):
    return np.array(list(one_of_k_encoding_unk(atom.GetSymbol(),
                                               ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                                                'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
                                                'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',    # H?
                                                'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                                                'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'])) +
                    list(one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])) +
                    list(one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])) +
                    list(one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])) +
                    [atom.GetIsAromatic()])


def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE,
                     bt == Chem.rdchem.BondType.DOUBLE,
                     bt == Chem.rdchem.BondType.TRIPLE,
                     bt == Chem.rdchem.BondType.AROMATIC,
                     bond.GetIsConjugated(),
                     bond.IsInRing()])


def num_atom_features():
    # Return length of feature vector using a very simple molecule.
    m = Chem.MolFromSmiles('CC')
    alist = m.GetAtoms()
    a = alist[0]
    return len(atom_features(a))


def num_bond_features():
    # Return length of feature vector using a very simple molecule.
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return len(bond_features(simple_mol.GetBonds()[0]))


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(
            "input {0} not in allowable set{1}:".format(x, allowable_set))
    return map(lambda s: x == s, allowable_set)


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return map(lambda s: x == s, allowable_set)


# for test
if __name__ == "__main__":
    from rdkit.Chem import MolFromSmiles
    # smiles = 'CC(C)C=C'
    smiles = 'CCC1=CC2=C(C=C1)C1=CC=CC=C21'
    mol = MolFromSmiles(smiles)
    # numbers of atom in molecular
    print("length of molecular:", len(mol.GetAtoms()))
    # print("degree of the atoms")
    for atom in mol.GetAtoms():
        print(atom.GetIdx())  # the index of atom in smiles
        print(atom.GetDegree())  # amount of neibors
        print(atom.GetTotalNumHs())  # H amount
        print(atom.GetImplicitValence())  # == H amount
        print(atom.GetIsAromatic())  # if in the Aromatic Ring

    for atom in mol.GetAtoms():
        print(atom_features(atom))
