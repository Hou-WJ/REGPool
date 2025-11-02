# Rebuilding to Remember: Reconstruction-Driven Regularization for Information-Preserving in Clustering Graph Pooling （REGPool）

## 1.Overview
The code for paper "Rebuilding to Remember: Reconstruction-Driven Regularization for Information-Preserving in Clustering Graph Pooling". 

![image](/data/Fig 1.PNG)

The repository is organized as follows:
* **`REGPool/`** - Contains model training scripts and parameter configuration code
* **`model/`** - Implements the core model architecture and baseline models
* **`data/`** - Stores required datasets (both raw and preprocessed data)
* **`results/`** - Saves intermediate files and output results generated during model execution

## 2.Dependencies
* python == 3.9
* numpy == 1.24.3
* pandas == 2.3.3
* scikit_learn == 1.7.2
* torch == 2.5.1+cu124
* torch_geometric == 2.6.1
* torchinfo == 1.8.0 
* torch_cluster == 1.6.3+pt25cu124
* torch_scatter == 2.1.2+pt25cu124
* torch_sparse == 0.6.18+pt25cu124
* torch_spline_conv == 1.2.2+pt25cu124
* torchvision == 0.20.1+cu124

## 3.Running REGPool
### Quick Start
Basic usage
```bash
python trainer.py --model REGPool --dataset PROTEINS
```
Available Basic Parameters

| Parameter | Description | Default | Example |
| ------ | ------ | ------ | ------ |
| --model | Model name | REGPool | Backbone |
| --dataset | Dataset name | PROTEINS | MUTAG |
| --gpu | Cuda devices | 0 | 0 |
| --hid_dim | Hidden embedding size | 8 | 32 |
| --num_layers | Number of GCN layers | 3 | 4 |
| --pooling_ratio | Pooling ratio | 0.1 | 0.4 |
| --batch_size | Batch size | 32 | 64 |
| --lr | Learning rate | 0.01 | 0.001 |
| --epochs | Max epochs | 100 | 300 |
| --patience | Early stopping | 50 | 100 |
| --jump_connection | Aggregation of jump connections | Cat | None |

### IDE/Editor Development
Step 1: Download project files
```bash
git clone https://github.com/Hou-WJ/REGPool.git
cd REGPool
```

Step 2: Configure environment
```bash
pip install -r requirements.txt
```

Step 3: Modify config.py
Edit config.py to customize your experiment parameters:
```python
parser = argparse.ArgumentParser(description='Trainer Template of Graph Neural Network with Pooling')
parser.add_argument('--model', type=str, default='REGPool', help='Model name:REGPool/Backbone')
parser.add_argument('--dataset', type=str, default='PROTEINS',
                    help='Dataset name, PROTEINS/NCI1/NCI109/Mutagenicity/ENZYMES/PTC_MR/PTC_FM/IMDB-BINARY/IMDB-MULTI/MUTAG/COLLAB')

parser.add_argument('--run_name', dest="name", type=str, default='test_' + str(uuid.uuid4())[:8],
                    help='Name of the run')
### 
### ……
### 

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
```

Step 4: Run training

Execute trainer.py directly in your IDE or via command line:
```bash
python trainer.py
```

## 4.Supported datasets
TuDataset: 
* `PROTEINS`
* `NCI1`
* `NCI109`
* `Mutagenicity`
* `ENZYMES`
* `PTC_MR`
* `PTC_FM`
* `IMDB-BINARY`
* `IMDB-MULTI`
* `MUTAG`
* `COLLAB`

Datasets mentioned above will be downloaded automatically using PyG's API when running the code.

For more information about the datasets, please refer to the [TUDataset documentation](https://chrsmrrs.github.io/datasets/).

## 5.Contacts
If you have any questions, please email Wenju Hou (wjhou23@mails.jlu.edu.cn)
