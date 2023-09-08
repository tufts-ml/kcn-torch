# A PyTorch Implementation of Kriging Convolutional Networks 

**Wenrui Zhao<sup>1</sup>, Ruiqi Xu<sup>1</sup>, and Li-Ping Liu<sup>1</sup>**

1 Department of Computer Science, Tufts University

### Overview

This repo contains the PyTorch implementation of Kriging Convolutional Networks (KCNs) [1]. The original code release was an implementation with TensorFlow 1.x, which is not well supported now. In this repo, we update the implementation with PyTorch and PyTorch Geometric. We hope this new implementation will help researchers in this field. If you have any questions about this repo, please feel free to raise issues. 

### Data

The model works with datasets in the `SpatialDataset` format. In particular, it contains three fields: 

* `dataset.coords`: a tensor with shape `[n, 2]`, each row is a 2-D coordinate of an instance. Coordinates with other dimensions can also be handled by KCN.      
* `dataset.features`: a tensor with shape `[n, d]`, each row is a `d`-dimensional feature vector of an instance.    
* `dataset.y`: a tensor with shape `[n, l]`, each row is a vector of `l` labels of an instance. The current example only work for one-dimensional continuous labels.    

The current repo provides a running example of KCN on a dataset of bird counts (counts of wood thrush reported in Eeatern US during June 2014) [[download link](https://tufts.box.com/v/kcn-bird-count-dataset)]. 
  
### Model

A KCN predicts a data point's label based on data points in its neighborhood. The KCN model stores a training set internally. To make a prediction for a data point, it looks up neighbors for the data point and construct, forms an attributed graph over data points in the neighborhood, and then uses a Graph Neural Network (GNN) to predict the data point's label. During training, these graphs are computed before training to save repeated graph constructions. The general structure of KCN is similar to a k-nearest-neighbor classifier, though the former one employs a much more flexible predictive function than simple averaging.

In the implementation, a KCN model is a PyTorch module. It is initialized with a `SpatialDataset`. In the `forward` function, it takes coordinates and features of a batch of data points and then predicts their labels.    

### Run the code

#### Requirements
The code has been tested on a linux platform with the following packages installed: `python=3.10, numpy=1.24.3, torch=2.01, scikit-learn=1.2.2, pyg=2.3.0`. You can install the environment from `environment.yml` with `conda env create -f environment.yml`.  

If you want to run the Jupyter notebook `demo.ipynb`, you need also install `geoplot` and `geopandas`. 

You can try the KCN model on a single train-test split by running `python main.py`. If you want to make changes to experiment settings and model parameter, you can provide more arguments to the command according to `args.py`, or you can directly edit default values of arguments in `args.py`. 

### Reference
[1] Gabriel Appleby, Linfeng Liu, and Li-Ping Liu. "Kriging convolutional networks." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 34. No. 04. 2020.   
[2] Sullivan, B.L., C.L. Wood, M.J. Iliff, R.E. Bonney, D. Fink, and S. Kelling. 2009. eBird: a citizen-based bird observation network in the biological sciences. Biological Conservation 142: 2282-2292.


