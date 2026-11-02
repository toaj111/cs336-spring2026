#!/bin/bash

uv run cs336_systems/benchmark.py --vocab_size 10000 \
              --context_length 512 \
              --rope_theta 10000 \
              --d_model 2560 \
              --d_ff 10240 \
              --num_layers 32 \
              --num_heads 32 \
              --batch_size 4