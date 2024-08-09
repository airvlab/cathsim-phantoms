#!/bin/bash

# Iterate over each subfolder in the phantoms directory
for subfolder in phantoms/*; do
  if [ -d "$subfolder" ]; then
    echo "Decompiling $subfolder"

    preprocess_resolution=100
    threshold=0.01
    max_convex_hull=-1
    mcts_iterations=100
    mcts_max_depth=3
    mcts_nodes=20
    resolution=10000
    seed=0

    obj2mjcf --obj-dir $subfolder --decompose --compile-model --save-mjcf \
      --coacd-args.preprocess-resolution $preprocess_resolution \
      --coacd-args.threshold $threshold \
      --coacd-args.max-convex-hull $max_convex_hull \
      --coacd-args.mcts-iterations $mcts_iterations \
      --coacd-args.mcts-max-depth $mcts_max_depth \
      --coacd-args.mcts-nodes $mcts_nodes \
      --coacd-args.resolution $resolution \
      --coacd-args.seed $seed

    # Extract the name of the .obj file (without extension) to determine the resulting folder name

    obj_file=$(ls $subfolder | grep -E "*.obj" | sed 's/.obj$//')
    result_folder="$obj_file"

    # Move the resulting folder to the parent directory (cwd)
    mv "$subfolder/$result_folder" ./

    # Copy the README.md from the subfolder to the corresponding result folder
    cp "$subfolder/README.md" "./$result_folder"

  fi
done
