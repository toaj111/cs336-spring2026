#!/bin/bash

uv run cs336_systems/benchmark.py --vocab_size 10000 \
              --context_length 512 \
              --rope_theta 10000 \
              --d_model 4608 \
              --d_ff 12288 \
              --num_layers 50 \
              --num_heads 36 \
              --batch_size 4