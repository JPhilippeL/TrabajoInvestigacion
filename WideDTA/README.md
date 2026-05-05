# WideDTA MPro working patch

This patch replaces the Davis/KIBA-hardcoded WideDTA training path with a dynamic model and a clean trainer.

Important: old `wide.pt` checkpoints from Davis are not compatible with this dynamic MPro pipeline. Train a new checkpoint.

Required MPro folder:

```text
WideDTA/data/mpro_urv/
├── ligands_can.txt
├── proteins.txt
├── motif2.txt
└── Y
```

If you already have a DeepDTA-style MPro folder with `ligands_can.txt`, `proteins.txt`, and `Y`, create the WideDTA folder with:

```bash
python -m WideDTA.Core.widedta_mpro_urv_converter \
  --source-deepdta-dir DeepDTA/data/mpro_urv \
  --output-dir WideDTA/data/mpro_urv \
  --use-protein-as-motif
```

Using the protein sequence as motif is a technical baseline, not the original WideDTA biological motif representation.

Smoke test:

```bash
python -m WideDTA.train --dataset mpro_urv --epochs 1 --batch-size 1 --max-train-batches 2
```
