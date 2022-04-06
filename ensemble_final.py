import pandas as pd

submissions = {
    "aug_submisison": pd.read_csv("aug_submisison.csv")["Prediction"],
    "aug_whitening_submisison": pd.read_csv("aug_whitening_submisison.csv")[
        "Prediction"
    ],
    "regular_submisison": pd.read_csv("regular_submisison.csv")["Prediction"],
    "whitening_submisison": pd.read_csv("whitening_submisison.csv")["Prediction"],
}

res = pd.concat([v.rename(k) for k, v in submissions.items()], axis=1).astype(int)
final = res.T.apply(lambda x: np.bincount(x).argmax()).to_numpy()

Yte = {"Prediction": final.to_numpy()}
dataframe = pd.DataFrame(Yte)
dataframe.index += 1
dataframe.to_csv("Yte.csv", index_label="Id")
