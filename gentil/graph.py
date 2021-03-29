import torch
import dgl
import numpy as np
from mstools.topology import Molecule


def mols2dgl(mol_list: [Molecule], elements, edge_max_distance=3):
    '''
    Convert molecules to DGL graphs
    Each atom is a node
    The edges are formed between atoms separated by at most `edge_max_distance` bonds
    The node feature is a one-hot vector of the element of each atom
    The edge feature is a one-hot vector of the bond between the atom pair
    '''
    graph_list = []
    feats_node_list = []
    feats_edge_list = []
    for mol in mol_list:
        feats_node = np.zeros((mol.n_atom, len(elements)))
        for atom in mol.atoms:
            feats_node[atom.id_in_mol][elements.index(atom.symbol)] = 1
        feats_node_list.append(feats_node)
        mat = mol.get_distance_matrix(max_bond=edge_max_distance)

        edges = [[], []]
        feats_edge = []
        for i in range(mol.n_atom):
            for j in range(i + 1, mol.n_atom):
                if mat[i][j] > 0:
                    edges[0].append(i)
                    edges[0].append(j)
                    edges[1].append(j)  # bidirectional
                    edges[1].append(i)
                    feats = [0] * edge_max_distance
                    feats[mat[i][j] - 1] = 1
                    feats_edge.append(feats[:])
                    feats_edge.append(feats[:])
        feats_edge_list.append(np.array(feats_edge))

        u = torch.tensor(edges[0])  # bidirectional
        v = torch.tensor(edges[1])

        graph = dgl.graph((u, v))
        graph_list.append(graph)

    return graph_list, feats_node_list, feats_edge_list
