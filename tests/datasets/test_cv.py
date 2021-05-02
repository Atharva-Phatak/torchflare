# flake8: noqa
from torchflare.datasets.cross_val import CVSplit
from torchflare.datasets.tabular import TabularDataset
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    ShuffleSplit,
    RepeatedStratifiedKFold,
    RepeatedKFold,
)
import pandas as pd

path = "tests/datasets/data/tabular_data/diabetes.csv"
df = pd.read_csv(path)
label_col = "Outcome"
input_cols = [col for col in df.columns if col != label_col]

ds = TabularDataset.from_df(df=df, feature_cols=input_cols, label_cols=label_col)


def _compare_dicts(d1, d2):

    for k1, k2 in zip(d1.keys(), d2.keys()):
        if (sorted(d1[k1]["train_idx"]) == sorted(d2[k2]["train_idx"])) and (
            sorted(d1[k1]["val_idx"]) == sorted(d2[k2]["val_idx"])
        ):
            return True
    return False


def test_kfold():
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_kf = CVSplit(dataset=ds, cv="KFold", n_splits=3, random_state=42, shuffle=True)

    fold_dict = {}
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=ds.inputs, y=ds.labels)):
        fold_dict[fold] = {"train_idx": train_idx, "val_idx": val_idx}

    val = _compare_dicts(fold_dict, cv_kf.fold_dict)
    assert val is True


def test_stratifiedKFold():
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_kf = CVSplit(dataset=ds, cv="StratifiedKFold", n_splits=3, random_state=42, shuffle=True)

    fold_dict = {}
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=ds.inputs, y=ds.labels)):
        fold_dict[fold] = {"train_idx": train_idx, "val_idx": val_idx}

    val = _compare_dicts(fold_dict, cv_kf.fold_dict)
    assert val is True


def test_shufflesplit():
    kf = ShuffleSplit(n_splits=3, train_size=0.5, test_size=0.25, random_state=42)
    cv_kf = CVSplit(dataset=ds, cv="ShuffleSplit", n_splits=3, random_state=42, train_size=0.5, test_size=0.25)

    fold_dict = {}
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=ds.inputs, y=ds.labels)):
        fold_dict[fold] = {"train_idx": train_idx, "val_idx": val_idx}

    val = _compare_dicts(fold_dict, cv_kf.fold_dict)
    assert val is True


def test_straified_shufflesplit():
    kf = StratifiedShuffleSplit(n_splits=3, train_size=0.5, test_size=0.25, random_state=42)
    cv_kf = CVSplit(
        dataset=ds, cv="StratifiedShuffleSplit", n_splits=3, random_state=42, train_size=0.5, test_size=0.25
    )

    fold_dict = {}
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=ds.inputs, y=ds.labels)):
        fold_dict[fold] = {"train_idx": train_idx, "val_idx": val_idx}

    val = _compare_dicts(fold_dict, cv_kf.fold_dict)
    assert val is True


def test_repeated_stratifiedKfold():
    kf = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=42)
    cv_kf = CVSplit(dataset=ds, cv="RepeatedStratifiedKFold", n_splits=3, random_state=42, n_repeats=2)

    fold_dict = {}
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=ds.inputs, y=ds.labels)):
        fold_dict[fold] = {"train_idx": train_idx, "val_idx": val_idx}

    val = _compare_dicts(fold_dict, cv_kf.fold_dict)
    assert val is True


def test_repeated_Kfold():
    kf = RepeatedKFold(n_splits=3, n_repeats=2, random_state=42)
    cv_kf = CVSplit(dataset=ds, cv="RepeatedKFold", n_splits=3, random_state=42, n_repeats=2)

    fold_dict = {}
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=ds.inputs, y=ds.labels)):
        fold_dict[fold] = {"train_idx": train_idx, "val_idx": val_idx}

    val = _compare_dicts(fold_dict, cv_kf.fold_dict)
    assert val is True
