import pandas as pd
import numpy as np
import torch  ## only for linear algebra operations
from torchvision.transforms import (
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
)  ## only for data augmentations
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.svm import SVC  ## for debugging
import yaml
import logging
from svm import MultiClassSVM


def load_data():
    Xtr = np.array(
        pd.read_csv("data/Xtr.csv", header=None, sep=",", usecols=range(3072))
    )
    Ytr = np.array(pd.read_csv("data/Ytr.csv", sep=",", usecols=[1])).squeeze()
    Xtr = Xtr.reshape(5000, 3, -1).reshape(-1, 3, 32, 32)
    return Xtr, Ytr


def load_train_test_data():
    Xtr = np.array(
        pd.read_csv("data/Xtr.csv", header=None, sep=",", usecols=range(3072))
    )
    Xte = np.array(
        pd.read_csv("data/Xte.csv", header=None, sep=",", usecols=range(3072))
    )
    Ytr = np.array(pd.read_csv("data/Ytr.csv", sep=",", usecols=[1])).squeeze()
    Xtr = Xtr.reshape(5000, 3, -1).reshape(-1, 3, 32, 32)
    Xte = Xte.reshape(2000, 3, -1).reshape(-1, 3, 32, 32)
    return Xtr, Ytr, Xte


def extract_random_patch(img, probs, p):
    max_ = len(probs)
    indx, indy = np.random.choice(max_, p=probs, size=2)
    return img[:, indx : indx + p, indy : indy + p]


class Main:
    """
    Main class for running our method
    """

    def __init__(self, **params):
        self.imsize = 32
        for n, v in params.items():
            setattr(self, n, v)
        self.patch_dict = None

    def get_augmented_data(self, X_train, y_train):
        """
        Augments the data using random crops and random flips
        Returns X_train and y_train with self.train_size samples
        """
        transf = Compose(
            [
                RandomCrop(self.imsize, padding=4, padding_mode="reflect"),
                RandomHorizontalFlip(p=1),
            ]
        )
        X_train2 = torch.tensor(
            X_train, device="cpu", requires_grad=False, dtype=torch.float64
        )
        y_train2 = torch.tensor(
            y_train, device="cpu", requires_grad=False, dtype=torch.float64
        )
        Xaug = torch.vstack([X_train2, transf(X_train2)])
        yaug = torch.cat([y_train2, y_train2])
        inds = torch.randperm(len(Xaug))
        inds_train = inds[: self.train_size]
        X_train, y_train = (
            Xaug[: self.train_size, :].numpy(),
            yaug[: self.train_size].numpy(),
        )

        return X_train, y_train

    def build_patch_dictionnary(self, X, y):
        """
        Creates the dictionary of patches
        patches are sampled randomly from X
        The distribution is slightly biased towards the center of the image
        """
        n = len(X)
        npc = n // 10
        res = np.zeros((self.n_patches, 3, self.patch_size, self.patch_size))
        max_ = self.imsize - self.patch_size
        mid = max_ // 2
        probs = np.exp(-((np.arange(max_) - mid) ** 2) / 2e2)
        probs = probs / probs.sum()

        ## build dict of X per class
        # d = {c: [] for c in range(10)}
        # for x, c in zip(X, y):
        #    d[int(c)].append(x)
        # for c in d.keys():
        #    d[c] = np.array(d[c])

        # res = []
        # for c in range(10):
        #    for _ in range(npc):
        #        j = np.random.choice(len(d[c]))
        #        res.append(
        #            extract_random_patch(d[c][j, :], probs, self.patch_size)
        #        )
        # res = np.array(res)

        for i in range(self.n_patches):
            j = np.random.choice(n)
            res[i, :] = extract_random_patch(X[j, :], probs, self.patch_size)

        if self.augment_dict:
            # adds the opposite patches as well
            self.n_patches *= 2
            return np.vstack([res, -res])
        return res

    def compute_whitening_params(self):
        """
        Computes the whitening matrix W from the dict of patches
        If self.use_whitening is False, returns the Identity
        """
        if not self.use_whitening:
            self.W = torch.eye(
                3 * self.patch_size * self.patch_size,
                device=self.device,
                requires_grad=False,
                dtype=torch.float64,
            )
            return

        patch_flat = torch.tensor(
            self.patch_dict.reshape((self.n_patches, -1)),
            device=self.device,
            requires_grad=False,
        )
        mu = torch.mean(patch_flat, dim=0)
        patch_flat = patch_flat - mu
        cov = torch.mm(patch_flat.T, patch_flat)
        cov = cov + self.whitening_lambda * torch.eye(
            cov.size()[0], device=self.device, requires_grad=False
        )
        U, S, V = torch.svd(cov)
        W = U @ torch.diag((1 / torch.sqrt(S))) @ V.T
        self.W = W

    def compute_conv_params(self):
        """
        The feature extraction can be implemented like a convolution (see paper for details)
        Here the parameters (weights and bias) of the convolution are computed
        """
        dict_patches = torch.tensor(
            self.patch_dict[:, :, np.newaxis, :],
            requires_grad=False,
            device=self.device,
        )
        norms = torch.norm(
            torch.matmul(dict_patches.view(self.n_patches, -1), self.W).view(
                self.n_patches, 3, -1
            ),
            dim=-1,
        )

        weights = (self.W.T @ self.W @ dict_patches.view(self.n_patches, -1).T).T
        weights = weights.view(self.n_patches, 3, 1, self.patch_size, self.patch_size)

        return weights, norms

    def batch_feature_vectors(self, X, weights, norms):
        """
        Computes a batch of features from the batch of images X
        """
        n = len(X)
        out_size = (
            self.n_patches,
            n,
            self.imsize - self.patch_size + 1,
            self.imsize - self.patch_size + 1,
        )
        ipt = torch.tensor(X, requires_grad=False, device=self.device)
        res = torch.zeros(out_size, requires_grad=False, device=self.device)
        ## Convolution
        for i in range(self.n_patches):
            res[i] = torch.nn.functional.conv2d(
                ipt, weights[i, :], groups=3, bias=0.5 * norms[i]
            ).sum(axis=1)
        u, _ = torch.topk(res, self.n_nearest, dim=0, sorted=True, largest=False)
        if self.hard_assignment:
            res = torch.where(res <= u[-1, :], 1.0, 0.0)  ## hard K-nn assignment
        else:
            res = torch.sigmoid(u[-1, :] - res)  ## soft sigmoid assignment

        res = res.permute(1, 0, 2, 3)
        ## Average pooling
        res = torch.nn.functional.avg_pool2d(
            res, kernel_size=(self.mean_pooling_size, self.mean_pooling_size)
        )
        return self.bn(res)  ## batch norm

    def get_feature_vectors(self, X):
        """
        Computes the features from the images X
        Computation is done in batches (faster)
        """
        n = len(X)
        all_features = []
        weights, norms = self.compute_conv_params()
        for i in tqdm(range(n // self.batch_size + 1), disable=self.disable_tqdm):
            Y = X[i * self.batch_size : (i + 1) * self.batch_size]
            all_features.append(self.batch_feature_vectors(Y, weights, norms).cpu())
        all_features = torch.vstack(all_features)
        return all_features

    def train_and_predict(self, X_train, y_train, X_test):
        """
        Trains the classifier on X_train, y_train and predicts on X_test
        returns :  preds on X_test
        """
        if not self.data_augmentation:
            X_train = X_train[: self.train_size, :]
            y_train = y_train[: self.train_size]
        else:
            X_train, y_train = self.get_augmented_data(X_train, y_train)
        logging.info("Compute patch dict")
        self.patch_dict = self.build_patch_dictionnary(X_train, y_train)
        logging.info("Compute whitening")
        self.compute_whitening_params()
        self.bn = torch.nn.BatchNorm2d(self.n_patches, affine=False, device=self.device)
        logging.info("Compute training features")
        features_train = self.get_feature_vectors(X_train)
        logging.info("Compute validation features")
        features_val = self.get_feature_vectors(X_test)
        logging.info("Compute kernel matrices")
        km_train = (
            torch.einsum("njkl,mjkl->nm", features_train, features_train).cpu().numpy()
        )
        km = torch.einsum("njkl,mjkl->nm", features_val, features_train).cpu().numpy()

        if self.use_sklearn:
            svc = SVC(kernel="precomputed")
            svc.fit(km_train, y_train)
            return svc.predict(km)

        svm = MultiClassSVM(km_train, n_classes=10, C=self.C)
        svm.fit_global(y_train)
        return svm.predict_global(km)

    def run_final(self):
        """
        Trains on X_train, predicts on the test data
        Saves the predictions
        """
        X_train, Y_train, X_test = load_train_test_data()
        preds = self.train_and_predict(X_train, Y_train, X_test)
        preds = pd.DataFrame({"Prediction": preds})
        preds.index += 1
        preds.to_csv(f"{self.name}_submisison.csv", index_label="Id")

    def main(self):
        """
        Used for research / validation
        """
        logging.info("Validation script")
        logging.info("Loading data")
        Xtr, Ytr = load_data()
        X_train, X_val, y_train, y_val = train_test_split(
            Xtr, Ytr, test_size=0.3, random_state=42
        )
        X_val = X_val[: self.test_size, :]
        y_val = y_val[: self.test_size]
        preds = self.train_and_predict(X_train, y_train, X_val)
        acc = (preds == y_val).mean()
        print(f"Accuracy : {acc:.2f}")

        return preds, acc


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.INFO,
    )
    import argparse

    parser = argparse.ArgumentParser(description="Run a validation/submission script")
    parser.add_argument("--config", type=str, help="Path to config yaml file")
    parser.add_argument("--validate", action="store_true")

    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.validate:
        M = Main(**config)
        M.main()

    else:
        M = Main(**config)
        M.run_final()
