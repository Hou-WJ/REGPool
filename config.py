import argparse
import uuid

parser = argparse.ArgumentParser(description='Trainer Template of Graph Neural Network with Pooling')
parser.add_argument('--model', type=str, default='Backbone', help='Model name:REGPool/Backbone')
parser.add_argument('--dataset', type=str, default='PROTEINS',
                    help='Dataset name, PROTEINS/NCI1/NCI109/Mutagenicity/ENZYMES/PTC_MR/PTC_FM/IMDB-BINARY/IMDB-MULTI/MUTAG/COLLAB')

parser.add_argument('--run_name', dest="name", type=str, default='test_' + str(uuid.uuid4())[:8],
                    help='Name of the run')
parser.add_argument('--notes', type=str, default='None', help='Notes/Other Params')

parser.add_argument('--gpu', type=str, default='0', help='Cuda devices')
parser.add_argument('--folds', type=int, default=10, help='Cross validation folds')
parser.add_argument('--restore', action='store_true', help='Model restoring')
parser.add_argument('--replication', action='store_false', help='Replication with different random seeds')
parser.add_argument('--seed', type=int, default=3653, help='Random seed')
parser.add_argument('--replication_num', type=int, default=1, help='Repeat Times (Number of Random seeds)')

parser.add_argument('--epochs', dest="max_epochs", type=int, default=100, help='Max epochs')
parser.add_argument('--patience', type=int, default=50, help='Early stopping')
parser.add_argument('--num_layers', type=int, default=3, help='Number of GCN-Pooling blocks')
parser.add_argument('--dropout_ratio', type=float, default=0.1, help='Dropout ratio')

parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--hid_dim', type=int, default=8, help='Hidden embedding size')
parser.add_argument('--pooling_ratio', type=float, default=0.1, help='Pooling ratio')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--l2', type=float, default=5e-4, help='L2 regularization')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
parser.add_argument('--lr_decay_step', type=int, default=50, help='Decay step of lr scheduler')
parser.add_argument('--lr_decay_factor', type=float, default=0.5, help='Decay factor of lr scheduler')

parser.add_argument('--jump_connection', type=str, default="Cat", help='Aggregation of jump connections, '
                                                                       'None/Sum/Cat/Mean/ReconCat')
parser.add_argument('--data_normalization', type=str, default="Mean", help='None/MaxMin/Mean')

parser.add_argument('--link_loss_coef', type=float, default=10)
parser.add_argument('--recon_loss_in_diff_coef', type=float, default=0.1)
parser.add_argument('--ent_loss_coef', type=float, default=-0.1)
parser.add_argument('--clu_loss_coef', type=float, default=10)
parser.add_argument('--r_sage_coef', type=float, default=1)
parser.add_argument('--r_recon_coef', type=float, default=1)
parser.add_argument('--top_k_retain', type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument('--hard', type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--top_ratio', type=int, default=3)
parser.add_argument('--temperature', type=float, default=0.1)

args = parser.parse_args()
