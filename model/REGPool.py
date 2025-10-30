import torch
from torch import nn
from math import ceil
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv
from torch_geometric.utils import to_dense_batch, to_dense_adj
from model.reg_pool import reg_pool, SoftTopKRetain


class SAGEConvolutions(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 normalize=True,
                 lin=True):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize=normalize)

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


class REDiffPoolLayer(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_embedding, no_new_clusters, top_k_retain=True, top_ratio=1,
                 temperature=0.1, hard=False):
        super().__init__()
        self.no_new_clusters = no_new_clusters
        self.gnn_pool = SAGEConvolutions(dim_input, dim_hidden, no_new_clusters, lin=True)
        self.gnn_embed = SAGEConvolutions(dim_input, dim_hidden, dim_embedding, lin=True)
        self.top_k_retain = top_k_retain
        self.temperature = temperature
        self.hard = hard

        if self.top_k_retain:
            k = max(no_new_clusters // top_ratio, 1)
            self.topK = SoftTopKRetain(k=k, temperature=self.temperature, hard=self.hard)
        else:
            self.topK = None

    def forward(self, x, adj, mask=None):
        s = self.gnn_pool(x, adj, mask)
        node_x = self.gnn_embed(x, adj, mask)

        x, adj, s, l, e, c, r = reg_pool(node_x, adj, s, mask, normalize=True, topk=self.topK,
                                         temperature=self.temperature)

        items = {
            "link_loss": l,
            "ent_loss": e,
            "clu_loss": c,
            "recon_loss_in_diff": r
        }
        return x, adj, s, items, node_x.clone()

    def reset_parameters(self):
        self.gnn_pool.reset_parameters()
        self.gnn_embed.reset_parameters()


class REGPool(nn.Module):
    def __init__(self, data, args):
        super().__init__()

        self.max_num_nodes = data.max_node_nums
        dim_features = data.num_features
        dim_target = data.num_classes
        self.args = args
        self.num_layers = args.num_layers
        self.dropout_ratio = args.dropout_ratio
        self.JK = args.jump_connection
        self.link_loss_coef = args.link_loss_coef
        self.ent_loss_coef = args.ent_loss_coef
        self.clu_loss_coef = args.clu_loss_coef
        self.recon_loss_in_diff_coef = args.recon_loss_in_diff_coef
        self.r_sage_coef = args.r_sage_coef
        self.r_recon_coef = args.r_recon_coef
        self.top_k_retain = args.top_k_retain
        self.top_ratio = args.top_ratio
        self.temperature = args.temperature
        self.hard = args.hard

        gnn_dim_hidden = args.hid_dim
        dim_embedding = args.hid_dim
        coarse_factor = args.pooling_ratio
        gnn_dim_input = dim_features
        gnn_embed_dim_output = dim_embedding

        no_new_clusters = ceil(coarse_factor * self.max_num_nodes)
        self.rediff_pool_layer = REDiffPoolLayer(gnn_dim_input, gnn_dim_hidden, dim_embedding, no_new_clusters,
                                                 top_k_retain=self.top_k_retain, top_ratio=self.top_ratio,
                                                 temperature=self.temperature, hard=self.hard)

        layers = []
        for i in range(self.num_layers):
            sage_layer = SAGEConvolutions(gnn_dim_hidden, gnn_dim_hidden, dim_embedding, lin=False)
            layers.append(sage_layer)

        self.sage_layers = nn.ModuleList(layers)

        if self.JK in ["Sum", "Mean"]:
            final_embed_dim_output = gnn_embed_dim_output * 2
        elif self.JK == "Cat":
            final_embed_dim_output = 2 * gnn_embed_dim_output * (self.num_layers + 1) + dim_features * 2
        elif self.JK == "ReconCat":
            final_embed_dim_output = 2 * gnn_embed_dim_output * (self.num_layers * 2 + 1) + dim_features * 4
        else:  # None and others
            final_embed_dim_output = gnn_embed_dim_output * 2

        self.recon = SAGEConvolutions(dim_embedding, gnn_dim_input, gnn_dim_input, lin=True)

        self.lin1 = nn.Linear(final_embed_dim_output, final_embed_dim_output // 2)
        self.lin2 = nn.Linear(final_embed_dim_output // 2, final_embed_dim_output // 4)
        self.lin3 = nn.Linear(final_embed_dim_output // 4, dim_target)

    def forward(self, graph_batch):
        x, edge_index, batch = graph_batch.x.float(), graph_batch.edge_index.long(), graph_batch.batch

        x, mask = to_dense_batch(x, batch=batch)
        adj = to_dense_adj(edge_index, batch=batch)
        init_x = x.detach().clone().requires_grad_(True)

        x_all = []
        x_recon_all = []
        r_sage = 0

        if self.JK in ["Cat", "ReconCat"]:
            x_emd = torch.cat([torch.max(x, dim=1)[0], torch.mean(x, dim=1)], dim=1)
            x_all.append(x_emd)

        x, adj, s, loss_items, node_x = self.rediff_pool_layer(x, adj, mask)
        x_emd = torch.cat([torch.max(x, dim=1)[0], torch.mean(x, dim=1)], dim=1)
        x_all.append(x_emd)

        mask = None
        for i in range(self.num_layers):
            x = self.sage_layers[i](x, adj, mask)
            x_emd = torch.cat([torch.max(x, dim=1)[0], torch.mean(x, dim=1)], dim=1)
            x_all.append(x_emd)

            x_recon = torch.bmm(s, x)
            recon_loss = torch.mean((node_x - x_recon) ** 2)
            r_sage = r_sage + recon_loss
            x_recon_all.append(torch.cat([torch.max(x_recon, dim=1)[0], torch.mean(x_recon, dim=1)], dim=1))

        x_recon = torch.bmm(s, x)
        adj_recon = torch.bmm(s, torch.bmm(adj, s.permute(0, 2, 1)))
        x_recon = self.recon(x_recon, adj_recon, mask)
        r_recon = torch.mean((init_x - x_recon) ** 2)

        x_emd = torch.cat([torch.max(x_recon, dim=1)[0], torch.mean(x_recon, dim=1)], dim=1)
        x_recon_all.append(x_emd)

        loss_items.update({
            "r_sage": r_sage,
            "r_recon": r_recon,
        })

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
        elif self.JK == "ReconCat":
            x_all = x_all + x_recon_all
            x = torch.cat(x_all, dim=1)
        else:  # None and others
            x = x_all[-1]

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        x = F.log_softmax(x, dim=-1)

        weighted_loss = 0
        weighted_loss += self.link_loss_coef * loss_items["link_loss"]
        weighted_loss += self.ent_loss_coef * loss_items["ent_loss"]
        weighted_loss += self.clu_loss_coef * loss_items["clu_loss"]
        weighted_loss += self.recon_loss_in_diff_coef * loss_items["recon_loss_in_diff"]
        weighted_loss += self.r_sage_coef * loss_items["r_sage"]
        weighted_loss += self.r_recon_coef * loss_items["r_recon"]

        return x, loss_items, weighted_loss

    def reset_parameters(self):
        self.rediff_pool_layer.reset_parameters()
        for i in range(self.num_layers):
            self.sage_layers[i].reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
