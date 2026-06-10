import pandas as pd


def load_pic50(pic50_path):
    df = pd.read_csv(
        pic50_path,
        sep=r"\s+|,|\t",
        engine="python",
        header=None,
        names=["pdb_id", "pIC50"],
    )
    return dict(zip(df["pdb_id"], df["pIC50"]))
