#!/bin/sh
export PYTHONPATH=./atarashi-paddle/:./bin/atarashi/protos/:$PYTHONPATH

python=./app/bin/python3

#python test.py

echo "Using gpu env ${CUDA_VISIBLE_DEVICES}"
#${python} ./dataset_test.py \

${python} ./paddle_toy3.py \
    --train_data_dir ./data_ernie/train_gz/ \
    --eval_data_dir  ./data_ernie/dev_gz/ \
    --vocab_file ./data_2/vocab \
    --max_seqlen 128 \
    --run_config '{
        "batch_size": 25,
        "model_dir": "./models/toy_8",
        "max_steps": 10000000,
        "save_steps": 1000,
        "log_steps": 100,
        "skip_steps": 100, # comment
        "eval_steps": 1000,
        "shit": 0
    }' \
    --hparam '{
        "hidden_size": 256,
        "vocab_size": 300000,
        "embedding_size": 256,
        "num_layers": 3,
        "learning_rate": 0.001
    }' \
    --vocab_size 300000 

