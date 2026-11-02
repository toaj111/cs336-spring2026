#!/bin/bash

uv run cs336_systems/benchmark.py --vocab_size 10000 \
              --context_length 512 \
              --rope_theta 10000 \
              --d_model 1024 \
              --d_ff 4096 \
              --num_layers 24 \
              --num_heads 16 \
              --batch_size 4