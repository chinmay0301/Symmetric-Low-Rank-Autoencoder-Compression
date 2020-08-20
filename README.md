# Symmetry aware Pruning 
The repo was built over the code base for the pytorch implementation of 'Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding'  by Song Han, Huizi Mao, William J. Dally by ```tonyapplekim/deepcompressionpytorch```

``` ./vae_compress.sh``` runs the three deep compression stages Pruning, weight sharing and huffman encoding on the saved autoencoder models.

``` ./low_rank_exp.sh``` runs joint low rank compression for a number of epsilon values. 


## Relevant files - 
low_rank_quant.py - code for low rank approximation (see the parameter descriptions in the file)
pruning.py - refer to tonyapplekim repo for instructions on how to use this repo. Added some extra flags like symmetrical pruning, etc in our version. 
prune.py

## Requirements
Following packages are required for this project
- Python3.6+
- tqdm
- numpy
- pytorch, torchvision
- scipy
- scikit-learn



