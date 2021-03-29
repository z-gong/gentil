import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn


class MLPModel(nn.Module):
    '''
    A multilayer perceptron model
    '''
    def __init__(self, in_dim, out_dim, hidden_layers):
        super().__init__()
        layers = []
        for i in range(len(hidden_layers) + 1):
            _in = in_dim if i == 0 else hidden_layers[i - 1]
            _out = out_dim if i == len(hidden_layers) else hidden_layers[i]

            linear = nn.Linear(_in, _out)
            torch.nn.init.normal_(linear.weight, std=0.5)
            torch.nn.init.zeros_(linear.bias)
            layers.append(linear)

            if i != len(hidden_layers):
                layers.append(nn.SELU())

        self.mlp = nn.Sequential(*layers)

    def forward(self, feats):
        return self.mlp(feats)


class ChargeIncrementModel(nn.Module):
    '''
    Charge on each atom is calculated as the summation of charge increments on all edges connected to the atom
    For non-neutral molecule, the net charge is spread to all atoms as initial charge
    '''
    def __init__(self, in_dim_node, in_dim_edge, hidden_dim_node, hidden_dim_edge, n_head=1):
        super().__init__()

        self.conv_list = nn.ModuleList()
        self.conv_list.append(dglnn.EGATConv(in_dim_node, in_dim_edge, hidden_dim_node, hidden_dim_edge, n_head))
        self.conv_list.append(dglnn.EGATConv(hidden_dim_node * n_head, hidden_dim_edge * n_head, hidden_dim_node,
                                             hidden_dim_edge, n_head))
        self.conv_list.append(dglnn.EGATConv(hidden_dim_node * n_head, hidden_dim_edge * n_head, hidden_dim_node,
                                             hidden_dim_edge, n_head))
        self.mlp = MLPModel(hidden_dim_edge * n_head, 1, [2 * hidden_dim_edge])

        self._hidden_dim_node = hidden_dim_node
        self._hidden_dim_edge = hidden_dim_edge
        self._n_head = n_head

    def forward(self, graph, feats_node, feats_edge, charges_init):
        for conv in self.conv_list:
            feats_node, feats_edge = conv(graph, feats_node, feats_edge)
            feats_node = F.relu(feats_node).view(-1, self._hidden_dim_node * self._n_head)
            feats_edge = F.relu(feats_edge).view(-1, self._hidden_dim_edge * self._n_head)

        increment = self.mlp(feats_edge).view(-1)

        with graph.local_scope():
            graph.edata['inc'] = increment
            graph.update_all(dgl.function.copy_e('inc', 'm'), dgl.function.sum('m', 'delta'))
            delta = graph.ndata['delta']

        gr = dgl.reverse(graph, copy_ndata=False, copy_edata=False)
        with gr.local_scope():
            gr.edata['inc_'] = -increment
            gr.update_all(dgl.function.copy_e('inc_', 'm'), dgl.function.sum('m', 'delta_'))
            delta_ = gr.ndata['delta_']

        return delta + delta_ + charges_init
