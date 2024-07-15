#!usr/bin/env bash

preprocess_resolution=100
threshold=0.01
max_convex_hull=-1
mcts_iterations=100
mcts_max_depth=3
mcts_nodes=20
resolution=10000
seed=0

obj2mjcf --obj-dir ./preprocessed/phantom-elastrat/ --decompose --compile-model --save-mjcf \
  --coacd-args.preprocess-resolution $preprocess_resolution \
  --coacd-args.threshold $threshold \
  --coacd-args.max-convex-hull $max_convex_hull \
  --coacd-args.mcts-iterations $mcts_iterations \
  --coacd-args.mcts-max-depth $mcts_max_depth \
  --coacd-args.mcts-nodes $mcts_nodes \
  --coacd-args.resolution $resolution \
  --coacd-args.pca \
  --coacd-args.seed $seed
