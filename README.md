# MVA 2021-22 Data Challenge

## Usage

- Create a YAML config file 
- To test the method on a validation set : 
`python main.py --validation --config config.yaml`
- To run the method and produce a submission file :
`python main.py --config config.yaml`

Accepted arguments : 
- `name` (`str`) : name of the run (used for the name of the submission file)
- `train_size` (`int`) : number of samples for training
- `test_size` (`int`) : number of samples for training (only in validation mode)
- `n_patches` (`int`) : size of the dictionnary of patches
- `patch_size` (`int`) : size of the patches in px
- `n_nearest` (`int`) : number of nearest neighbours
- `augment_dict` (`bool`) : augment patches with their opposite
- `mean_pooling_size` (`int`) : size of the mean pooling of the features
- `use_whitening` (`bool`) : preprocess the patches with a whitening transform
- `whitening_lambda` (`bool`) : parameter lambda for the whitening transform
- `data_augmentation` (`bool`) : use data augmentation 
- `hard_assignment` (`bool`): If true, uses the hard assignment on nearest neighbours. If false, uses a sigmoid thresholding.
- `use_sklearn` (`bool`) : use sklearn for the SVM (used for debugging)
- `C` (`float`) : SVM regularization parameter
- `device` (`str`) 'cuda' or 'cpu' 
- `disable_tqdm` (`bool`) : disable tqdm progress bars
- `batch_size` (`bool`) : batch size for computing thre feature vectors