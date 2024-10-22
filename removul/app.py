from __future__ import absolute_import, division, print_function
import os
from flask import Flask, jsonify, request
import argparse
import logging
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
import pandas as pd
import pytorch_lightning as pl
from linevul import Model,UCC_Data_Module
from linelevel import LineLevel
from filevul import Filevul,Directoryvul
import random
import numpy as np
import torch

logging.basicConfig(filename='linelevel.log', level=logging.DEBUG)

parser = argparse.ArgumentParser(description='Script so useful.')
parser.add_argument('--model_name_or_path', type=str, default='microsoft/codebert-base', help='pretrained model name')
parser.add_argument('--train_data_file', type=str, default='../../train.csv', help='train data file')
parser.add_argument('--eval_data_file', type=str, default='../../val.csv', help='eval data file')
parser.add_argument('--test_data_file', type=str, default='../../test.csv', help='test data file')
parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='adam epsilon')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
parser.add_argument('--do_train', action='store_true', help='do train')
parser.add_argument('--do_test', action='store_true', help='do test')
parser.add_argument('--epochs', type=int, default=5, help='epochs')
parser.add_argument('--block_size', type=int, default=512, help='block size')
parser.add_argument('--train_batch_size', type=int, default=9, help='train batch size')
parser.add_argument('--eval_batch_size', type=int, default=9, help='eval batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='test batch size')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='learning rate')
parser.add_argument('--max_steps', type=int, default=1, help='max steps')
parser.add_argument('--file_name', type=str, default=None, help='file name to check vulnerability in it')
parser.add_argument('--do_linelevel', action='store_true', help='do line level')
parser.add_argument('--warmup_steps', type=int, default=0, help='warmup steps')
parser.add_argument('--directory_name', type=str, default=None, help='directory name to check vulnerability in it')
parser.add_argument('--num_labels', type=int, default=2, help='number of labels')
parser.add_argument('--function_column', type=str, default='processed_func', help='function column name')
parser.add_argument('--target_column', type=str, default='target', help='target column name')
parser.add_argument('--checkpoint_path', type=str, default="./lightning_logs/version_11/checkpoints/epoch=4-step=83840.ckpt", help='checkpoint path')
parser.add_argument('--hparams_file', type=str, default="./lightning_logs/version_11/hparams.yaml", help='hparams file')
# class weight tensor
parser.add_argument('--class_weight', type=list, default=[0.5307, 8.6371], help='class weight tensor')


# parse arguments
args = parser.parse_args()
# get config and tokenizer
config = RobertaConfig.from_pretrained(args.model_name_or_path)
config.num_labels = args.num_labels
config.num_attention_heads = 12

tokenizer = RobertaTokenizer(vocab_file="../bpe_tokenizer/bpe_tokenizer-vocab.json",
                                    merges_file="../bpe_tokenizer/bpe_tokenizer-merges.txt")
checkpoint_path=args.checkpoint_path
hparams_file=args.hparams_file

def set_seed(num):
    """ Set all seeds to make results reproducible """
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed_all(num)

set_seed(42)
app = Flask(__name__)

@app.route('/api/list_numbers', methods=['GET'])
def list_numbers():
    directory = request.args.get('directory')
    
    # Check if the directory exists
    if not os.path.isdir(directory):
        return jsonify({'error': 'Invalid directory path'})

    # Get a list of numbers in the directory
    model = Model(config, tokenizer, args)
    model = model.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        hparams_file=hparams_file,
        map_location=None,
        config=config,
        tokenizer=tokenizer,
        args=args
    )   
    vul_lines = Directoryvul(dirname=directory,model=model,tokenizer=tokenizer,args=args)

    return jsonify({'vul_lines': vul_lines})

if __name__ == '__main__':
    app.run()
