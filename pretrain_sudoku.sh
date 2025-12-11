OMP_NUM_THREADS=8 torchrun  --standalone --nproc-per-node 8 pretrain_sr2.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=60000 eval_interval=2000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0

