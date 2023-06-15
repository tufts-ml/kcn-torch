import os
import numpy as np
import torch

class SpatialDataset(torch.utils.data.Dataset):
    """A dataset class for spatial data."""

    def __init__(self, coords, features, y):
        """
        Args:
            coords: tensor with shape `(n, 2)`, coordinates of `n` instances
            features: tensor with shape `(n, d)`, `d` dimensional feature vectors of `n` instances
            y: tensor with shape `(n, )`, labels of `n` instances. Please provide zeros if unknown. 
            neighbors: tensor with shape `(n, num_neighbors)`, neighbors in an external training set. 
                       It can be none and computed later.  
        """
        super(SpatialDataset, self).__init__()

        if coords.shape[0] != features.shape[0] or features.shape[0] != y.shape[0]:
            raise Exception(f"Coordinates, features, and labels have different numbers of instances: \
                             coords.shape[0]={coords.shape[0]}, features.shape[0]={features.shape[0]}, \
                             y.shape[0]={y.shape[0]}")

        
        self.coords = torch.Tensor(coords)
        self.features = torch.Tensor(features)
        self.y = torch.Tensor(y) 
        

    def __len__(self):
        return self.coords.shape[0] 

    def __getitem__(self, idx):
        
        ins = (self.coords[idx], self.features[idx], self.y[idx])

        return ins




def load_bird_count_data(args):
    """
     Load data for training and testing

    Args
    ----
    args : will use three fields, args.dataset, args.data_path, args.random_seed  

    Returns
    -------
    coords    : np.ndarray, shape (N, 2), coordinates of the data points
    features  : np.ndarray, shape (N, D), features of the data points
    y         : np.ndarray, shape (N, 1), labels of the data points
    num_total_train : int, number of training data points. The first `num_total_train` 
                of instances from three other return values should form the training set
    """

    # data file path
    datafile = os.path.join(args.data_path, args.dataset + ".npz")

    # download data if not finding it
    if not os.path.isfile(datafile):  
        raise Exception(f"Data file {datafile} not found. Please download the dataset from https://tufts.box.com/v/kcn-bird-count-dataset and save it to ./datasets/bird_count.npz")

    # load the data
    data = np.load(datafile)
    X_train = np.ndarray.astype(data['Xtrain'], np.float32)
    Y_train = data['Ytrain'].astype(np.float32)
    Y_train = Y_train[:, None]
    X_test = np.ndarray.astype(data['Xtest'], np.float32)
    Y_test = data['Ytest'].astype(np.float32)
    Y_test = Y_test[:, None]



    num_total_train = X_train.shape[0]

    # check and record shapes
    assert (X_train.shape[0] == Y_train.shape[0])
    assert (X_test.shape[0] == Y_test.shape[0])

    if args.use_default_test_set:
        print("Using the default test set from the data") 
        trainset = SpatialDataset(coords=X_train[:, 0:2], features=X_train, y=Y_train) 
        testset = SpatialDataset(coords=X_test[:, 0:2], features=X_test, y=Y_test)
    else:
        X = np.concatenate([X_train, X_test], axis=0)
        Y = np.concatenate([Y_train, Y_test], axis=0)

        perm = np.random.RandomState(seed=args.random_seed).permutation(X.shape[0])

        # include coordinates in features
        trainset = SpatialDataset(coords=X[perm[0:num_total_train], 0:2], features=X[perm[0:num_total_train]], y=Y[perm[0:num_total_train]]) 
        testset = SpatialDataset(coords=X[perm[num_total_train:], 0:2], features=X[perm[num_total_train:]], y=Y[perm[num_total_train:]])

    # feature normalization
    feature_mean = torch.mean(trainset.features, axis=0, keepdims=True)
    feature_std = torch.std(trainset.features, axis=0, keepdims=True)

    trainset.features = (trainset.features - feature_mean) / (feature_std + 0.01)
    testset.features = (testset.features - feature_mean) / (feature_std + 0.01)

    return trainset, testset




