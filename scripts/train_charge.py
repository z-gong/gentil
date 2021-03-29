import munch
import json
import dgl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from gentil.graph import mols2dgl
from gentil.model import ChargeIncrementModel
from mstools.topology import Molecule, Atom

ELEMENTS = ['C', 'H', 'O']
EDGE_MAX_DISTANCE = 3


def load_molecules(filename, n_max=None, allowed_elements=None):
    mols = []
    with open(filename) as f:
        for n, line in tqdm(enumerate(f), total=n_max):
            if n_max and n >= n_max:
                break

            d = munch.munchify(json.loads(line))
            if allowed_elements and set(d.elements) - set(allowed_elements):
                continue

            mol = Molecule(d.inchi_key)
            mol.id = n
            for elem in d.elements:
                atom = Atom()
                atom.symbol = elem
                mol.add_atom(atom)
            else:
                for i, j, order in d.bonds:
                    mol.add_bond(mol.atoms[i], mol.atoms[j])
                for i, charge in enumerate(d.esp_charge):
                    mol.atoms[i].charge = charge
                mols.append(mol)

    return mols


def init_tensors(mol_list):
    graph_list, feats_node_list, feats_edge_list = mols2dgl(mol_list, ELEMENTS, EDGE_MAX_DISTANCE)
    print(graph_list[0])

    device = torch.device('cpu')
    batch_graph = dgl.batch(graph_list).to(device)
    feats_node = torch.tensor(np.concatenate(feats_node_list), dtype=torch.float32, device=device)
    feats_edge = torch.tensor(np.concatenate(feats_edge_list), dtype=torch.float32, device=device)

    esp_list = [[atom.charge for atom in mol.atoms] for mol in mol_list]
    net_charge_list = [sum(esp) for esp in esp_list]
    init_charge_list = [[net / mol.n_atom] * mol.n_atom for net, mol in zip(net_charge_list, mol_list)]

    charges_init = torch.tensor(np.concatenate(init_charge_list), dtype=torch.float32, device=device)
    charges_target = torch.tensor(np.concatenate(esp_list), dtype=torch.float32, device=device)

    return batch_graph, feats_node, feats_edge, charges_init, charges_target


def train(mol_list, n_epoch):
    batch_graph, feats_node, feats_edge, charges_init, charges_target = init_tensors(mol_list)

    model = ChargeIncrementModel(len(ELEMENTS), EDGE_MAX_DISTANCE, 10, 10, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1E-4)
    board = SummaryWriter()
    for epoch in range(n_epoch):
        model.train()
        optimizer.zero_grad()
        result = model(batch_graph, feats_node, feats_edge, charges_init)
        loss = criterion(result, charges_target)
        loss.backward()
        optimizer.step()
        print(epoch, loss)
        board.add_scalar("Loss/train", loss, epoch)
        torch.save(model, 'model.pt')


def predict(smiles_list):
    mol_list = [Molecule.from_smiles(smiles) for smiles in smiles_list]
    batch_graph, feats_node, feats_edge, charges_init, charges_target = init_tensors(mol_list)

    model = torch.load('model.pt')
    result = model(batch_graph, feats_node, feats_edge, charges_init).detach()

    offset = 0
    for mol in mol_list:
        print([atom.symbol for atom in mol.atoms])
        charges = result[offset: offset + mol.n_atom]
        print(charges)
        offset += mol.n_atom


if __name__ == '__main__':
    mol_list = load_molecules('../data/esp.txt', n_max=1000, allowed_elements=ELEMENTS)
    print(len(mol_list))
    train(mol_list, 500)
    predict(['CCCO', 'COC', 'CC(=O)O', 'c1ccccc1'])
