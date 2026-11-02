#!/bin/bash

uv run cs336_systems/benchmark.py --vocab_size 10000 \
              --context_length 512 \
              --rope_theta 10000 \
              --d_model 768 \
              --d_ff 3072 \
              --num_layers 12 \
              --num_heads 12 \
              --batch_size 4