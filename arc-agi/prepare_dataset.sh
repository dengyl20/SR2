# Dataset Preparation
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/input_arc2/arc-agi \
  --output-dir data/arc-2-aug-1000 \
  --subsets training evaluation \
  --test-set-name evaluation