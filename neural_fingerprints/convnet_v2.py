from torch import nn
import torch
import torch.nn.functional as F
from .features import num_atom_features,num_bond_features
from .mol_graph import degrees, graph_from_smiles_tuple

def sum_and_stack(features, idxs_list_of_lists):
    sum_list = [torch.sum(features[idx_list], axis=0)
             for idx_list in idxs_list_of_lists]
    return torch.stack(sum_list)

class NeuralConvNetwork_v2(nn.Module):

    def __init__(self, num_hidden_features=[100, 100], fp_length=512,
                 normalize=True):
        super(NeuralConvNetwork_v2, self).__init__()

        self.layers_config = [num_atom_features()] + num_hidden_features + [fp_length]
        self.num_hidden_features = num_hidden_features
        self.in_and_out_sizes = zip(
            self.layers_config[:-1],  self.layers_config[1:])
        self.normalize = normalize

        # build the network components
        self.linears = nn.ModuleList()
        self.linear_degrees = nn.ModuleList()
        
        layer_index = 0
        for (in_size, out_size) in self.in_and_out_sizes:
            self.linears.append(nn.Linear(in_size, out_size))
            self.linear_degrees.append(nn.ModuleList())
            for _ in degrees:
                self.linear_degrees[layer_index].append(
                    nn.Linear(in_size+num_bond_features(), out_size))
            layer_index += 1

    def forward(self, smiles):
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        array_rep = array_rep_from_smiles(tuple(smiles))
        atom_features = array_rep['atom_features'].to(device)
        bond_features = array_rep['bond_features'].to(device)
        num_layers = len(self.layers_config)-1


        for i in range(num_layers):
            self_logits = self.linears[i](atom_features)

            neighbors_logits_by_degree = []
            for degree in degrees:
                atom_neighbors_list = array_rep[('atom_neighbors', degree)]
                bond_neighbors_list = array_rep[('bond_neighbors', degree)]
                if len(atom_neighbors_list) > 0:
                    neighbor_features = [atom_features[atom_neighbors_list],
                                         bond_features[bond_neighbors_list]]
                    # dims of stacked_neighbors are [atoms, neighbors, atom and bond features]
                    stacked_neighbors = torch.cat(neighbor_features, axis=2)
                    summed_neighbors = torch.sum(stacked_neighbors, axis=1)
                    neighbors_logits_by_degree.append(
                        self.linear_degrees[i][degree](summed_neighbors))
            neighbors_logits = torch.cat(neighbors_logits_by_degree, axis=0)
            total_logits = 0.

            if self.normalize:
                total_logits = F.relu(F.normalize(
                    neighbors_logits+self_logits))
            else:
                total_logits = F.relu(neighbors_logits+self_logits)

            atom_features = total_logits

        logits = sum_and_stack(atom_features, array_rep['atom_list'])

        return logits

def array_rep_from_smiles(smiles):
    molgraph = graph_from_smiles_tuple(smiles)
    arrayrep = {'atom_features': torch.from_numpy(molgraph.feature_array('atom')).float(),
                'bond_features': torch.from_numpy(molgraph.feature_array('bond')).float(),
                # List of lists.
                'atom_list': molgraph.neighbor_list('molecule', 'atom'),
                'rdkit_ix': molgraph.rdkit_ix_array()}  # For plotting only.
    for degree in degrees:
        arrayrep[('atom_neighbors', degree)] = \
            torch.tensor(molgraph.neighbor_list(
                ('atom', degree), 'atom'), dtype=int)
        arrayrep[('bond_neighbors', degree)] = \
            torch.tensor(molgraph.neighbor_list(
                ('atom', degree), 'bond'), dtype=int)
    return (arrayrep)
