#!/bin/bash

uv run cs336_systems/benchmark.py --vocab_size 10000 \
              --context_length 512 \
              --rope_theta 10000 \
              --d_model 1280 \
              --d_ff 5120 \
              --num_layers 36 \
              --num_heads 20 \
              --batch_size 4