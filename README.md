# Enhancing Graph Masked Autoencoders with Dual-Level Adversarial Masking and Regularization

## Transfer learning on molecular graphs

For transfer learning on molecular graphs, please refer to the code in the `molecule/` folder.


## Usage

For quick start, you could run the scripts:

**Node classification**

```bash
sh scripts/run_transductive.sh <dataset_name> <gpu_id> # for transductive node classification
# example: sh scripts/run_transductive.sh cora/citeseer/pubmed/ogbn-arxiv 0
sh scripts/run_inductive.sh <dataset_name> <gpu_id> # for inductive node classification
# example: sh scripts/run_inductive.sh reddit/ppi 0

# Or you could run the code manually:
# for transductive node classification
python main_transductive.py --dataset wikics --seed 0 --device 0 --use_cfg
# for inductive node classification
python main_inductive.py --dataset ppi --seed 0 --device 0 --use_cfg
```
Supported datasets:
* transductive node classification:  `cora`, `citeseer`, `pubmed`, `corafull`, `wikics`,`ogbn-arxiv`,`flickr`
* inductive node classification: `ppi`, `reddit` 


**Graph classification**

```bash
sh scripts/run_graph.sh <dataset_name> <gpu_id>
# example: sh scripts/run_graph.sh mutag/imdb-b/imdb-m/proteins/... 0 

# Or you could run the code manually:
python main_graph.py --dataset COLLAB --seed 0 --device 0 --use_cfg
```
Supported datasets: 

- `IMDB-BINARY`, `IMDB-MULTI`, `PROTEINS`, `MUTAG`,  `COLLAB`, `REDDIT-BINARY`


## Requirements

- Python >= 3.9.5
- PyTorch >= 1.11.0 
- dgl >= 1.0.0
- scikit-learn >= 1.0.2
- PyYAML
- ogb
- tqdm
