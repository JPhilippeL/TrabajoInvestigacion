from GNNs.hyperparameter_search import run_hyperparameter_search

results = run_hyperparameter_search(
    train_sdf_dir="../../MPro-URV_Version2/MPro-URV_Version2/Ligand/Ligand_SDF/5RGV_ligand.sdf",
    target_file="../../MPro-URV_Version2/MPro-URV_Version2/pIC50.txt",
    eval_sdf_dir="../../MPro-URV_Version2/MPro-URV_Version2/Ligand/Ligand_SDF",
    eval_targets_file="../../MPro-URV_Version2/MPro-URV_Version2/pIC50.txt",
    output_root="hyperparameter_Search",
    model_names=["GIN", "GINE", "GAT", "EGAT", "GraphTransformer"],
    search_space={
        "lr": [1e-3, 5e-4],
        "batch_size": [16, 32],
        "hidden_dim": [64, 128],
        "num_layers": [2, 3],
    },
    epochs=20,
    patience=5,
    objective_metric="RMSE",
    objective_mode="min",
)



if __name__ == "__main__":
    results = run_hyperparameter_search(
        train_sdf_dir="/home/mohamed/Studies/stage/MPro-URV_Version2/MPro-URV_Version2/Ligand/Ligand_SDF",
        target_file="/home/mohamed/Studies/stage/MPro-URV_Version2/MPro-URV_Version2/pIC50.txt",
        eval_sdf_dir="/home/mohamed/Studies/stage/MPro-URV_Version2/MPro-URV_Version2/Ligand/Ligand_SDF",
        eval_targets_file="/home/mohamed/Studies/stage/MPro-URV_Version2/MPro-URV_Version2/pIC50.txt",
        output_root="hyperparameter_Search",
        model_names=["GIN", "GINE", "GAT", "EGAT", "GraphTransformer"],
        search_space={
            "lr": [1e-3, 5e-4, 1e-4],
            "batch_size": [16, 32],
            "hidden_dim": [64, 128],
            "num_layers": [2, 3],
            "atom_emb_dim": [0.4],
            "hibrid_emb_dim": [0.5],
            "bond_emb_dim": [1],
        },
        epochs=2,
        patience=1,
        valid_split=0.2,
        objective_metric="RMSE",
        objective_mode="min",
        resume=True,
        rerun_failed=True,
        seed=42,
    )

    print(results)
