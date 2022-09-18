# Neural-Fingerprints-Pytorch

## Description
Pytorch implementation of [Convolutional Networks on Graphs for Learning Molecular Fingerprints][NGF-paper]. Generate data-driven Molecular Fingerprints from SMILES.
<img src="./assets/neural_fingerprints.jpg"/>
The available structure of another kind of Molecular Fingerprints Networks in [convnet_v2.py](./neural_fingerprints/convnet_v2.py) is shown as follow:
<img src="./assets/neural_fingerprints_v2.jpg"/>

## Usage
[regression.py](regression.py) is an example script to do regression work using neural fingerprints.
+ there are a convolutional network for fingerprints generation and a basic ANN network in these scripts, they can be generated by
  
```python
from neural_fingerprints.convnet import NeuralConvNetwork
from neural_fingerprints.deepnet import DeepNetwork

model_fp = NeuralConvNetwork(**conv_params)
model_deep = DeepNetwork(**ann_params)

```
+ the params for network incloud these contents, which is familiar with those by HIPS:
```python
conv_params = {
    fp_length=50, # output of conv network
    fp_depth=4, # depth of conv network
    conv_width=20,   # node numbers of hidden layer
    h1_size=100
}

ann_params = {
    layer_sizes=[model_params['fp_length'], 
    model_params['h1_size']],
    normalize=True
}
```


## Other implementations
 - by [HIPS][1] using autograd, the direct reference for writing these script
 - by [debbiemarkslab][2] using theano
 - by [keiserlab][3] using keras
 - by [DeepChem][4] using tensorflow

## Dependencies
- RDKit
- Pytorch = 1.x
- NumPy


[NGF-paper]: https://arxiv.org/abs/1509.09292
[1]: https://github.com/HIPS/neural-fingerprint
[2]: https://github.com/debbiemarkslab/neural-fingerprint-theano
[3]: https://github.com/keiserlab/keras-neural-graph-fingerprint
[4]: https://github.com/deepchem/deepchem