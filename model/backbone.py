import torch
from torch import nn
from math import ceil
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj

class SAGEConvolutions(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 normalize=True,
                 lin=True):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize=normalize)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        if lin is True:
            self.lin = nn.Linear(hidden_channels, out_channels)
        else:
            self.lin = None

    def forward(self, x, adj, mask=None):
        x = F.relu(self.conv1(x, adj, mask))
        if self.lin is not None:
            x = self.lin(x)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        if self.lin is not None:
            self.lin.reset_parameters()


class PoolLayer(nn.Module):

    def __init__(self, dim_input, dim_hidden, dim_embedding, no_new_clusters):
        super().__init__()
        self.no_new_clusters = no_new_clusters
        self.gnn_pool = SAGEConvolutions(dim_input, dim_hidden, no_new_clusters)
        self.gnn_embed = SAGEConvolutions(dim_input, dim_hidden, dim_embedding, lin=False)

    def forward(self, x, adj, mask=None):
        s = self.gnn_pool(x, adj, mask)
        x = self.gnn_embed(x, adj, mask)
        x, adj, l, e = dense_diff_pool(x, adj, s, mask)
        return x, adj, l, e

    def reset_parameters(self):
        self.gnn_pool.reset_parameters()
        self.gnn_embed.reset_parameters()


class Backbone(nn.Module):
    def __init__(self, data, args):
        super().__init__()

        self.max_num_nodes = data.max_node_nums
        dim_features = data.num_features
        dim_target = data.num_classes

        self.args = args
        self.num_layers = args.num_layers
        self.dropout_ratio = args.dropout_ratio
        self.JK = args.jump_connection
        gnn_dim_hidden = args.hid_dim  # embedding size of first 2 SAGE convolutions
        dim_embedding = args.hid_dim  # embedding size of 3rd SAGE convolutions (eq. 5, dim of Z)
        dim_embedding_MLP = args.hid_dim  # hidden neurons of last 2 MLP layers

        coarse_factor = args.pooling_ratio

        gnn_dim_input = dim_features
        no_new_clusters = ceil(coarse_factor * self.max_num_nodes)
        gnn_embed_dim_output = dim_embedding

        layers = []
        self.pool_layer = PoolLayer(gnn_dim_input, gnn_dim_hidden, dim_embedding, no_new_clusters)
        for i in range(self.num_layers):
            sage_layer = SAGEConvolutions(gnn_dim_hidden, gnn_dim_hidden, dim_embedding, lin=False)
            layers.append(sage_layer)

        self.sage_layers = nn.ModuleList(layers)

        if self.JK in ["Sum", "Mean"]:
            final_embed_dim_output = gnn_embed_dim_output * 2
        elif self.JK == "Cat":
            final_embed_dim_output = gnn_embed_dim_output * 2 * (self.num_layers + 1)
        else:
            final_embed_dim_output = gnn_embed_dim_output * 2

        self.lin1 = nn.Linear(final_embed_dim_output, dim_embedding_MLP)
        self.lin2 = nn.Linear(dim_embedding_MLP, dim_embedding_MLP // 2)
        self.lin3 = nn.Linear(dim_embedding_MLP // 2, dim_target)

    def forward(self, graph_batch):
        x, edge_index, batch = graph_batch.x.float(), graph_batch.edge_index.long(), graph_batch.batch
        x, mask = to_dense_batch(x, batch=batch)
        adj = to_dense_adj(edge_index, batch=batch)

        x_all = []
        x, adj, _, _ = self.pool_layer(x, adj, mask)
        x_emd = torch.cat([torch.max(x, dim=1)[0], torch.mean(x, dim=1)], dim=1)
        x_all.append(x_emd)
        mask = None

        for i in range(self.num_layers):
            x = self.sage_layers[i](x, adj, mask)
            x_all.append(torch.cat([torch.max(x, dim=1)[0], torch.mean(x, dim=1)], dim=1))

        batch_size, num_nodes, embed_dim = x.size()
        if self.JK == "Sum":
            x = torch.cat(x_all, dim=1)
            x = torch.reshape(x, (batch_size, self.num_layers, embed_dim * 2))
            x = torch.sum(x, 1)
        elif self.JK == "Mean":
            x = torch.cat(x_all, dim=1)
            x = torch.reshape(x, (batch_size, self.num_layers, embed_dim * 2))
            x = torch.sum(x, 1) / self.num_layers
        elif self.JK == "Cat":
            x = torch.cat(x_all, dim=1)
        else:  # None and others
            x = x_all[-1]

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        x = F.log_softmax(x, dim=-1)
        return x

    def reset_parameters(self):
        self.pool_layer.reset_parameters()
        for i in range(self.num_layers):
            self.sage_layers[i].reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
