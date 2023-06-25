from __future__ import absolute_import, division, print_function
import argparse
import logging
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
import pandas as pd
import pytorch_lightning as pl
from removul.linevul import Model,UCC_Data_Module
from removul.linelevel import LineLevel
from removul.filevul import Filevul,Directoryvul
import random
import numpy as np
import torch
from dotenv import load_dotenv
import os

# Set the logging level to ignore warnings
logging.getLogger("transformers").setLevel(logging.ERROR)



logging.basicConfig(filename='linelevel.log', level=logging.DEBUG)
# Load the environment variables from the .env file
load_dotenv()

# Get the base directory from the environment variable
base_dir = os.environ['BASE_DIR']

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
parser.add_argument('--checkpoint_path', type=str, default=f"{base_dir}/removul/lightning_logs/version_11/checkpoints/epoch=4-step=83840.ckpt", help='checkpoint path')
parser.add_argument('--hparams_file', type=str, default=f"{base_dir}/removul/lightning_logs/version_11/hparams.yaml", help='hparams file')
# class weight tensor
parser.add_argument('--class_weight', type=list, default=[0.5307, 8.6371], help='class weight tensor')



def set_seed(num):
    """ Set all seeds to make results reproducible """
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed_all(num)

def main():
    # parse arguments
    args = parser.parse_args()
    # get config and tokenizer
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    config.num_labels = args.num_labels
    config.num_attention_heads = 12

    tokenizer = RobertaTokenizer(vocab_file=f"{base_dir}/bpe_tokenizer/bpe_tokenizer-vocab.json",
                                        merges_file=f"{base_dir}/bpe_tokenizer/bpe_tokenizer-merges.txt")
    set_seed(42)
    checkpoint_path=args.checkpoint_path
    hparams_file=args.hparams_file

    # Training
    if args.do_train:
        model = Model(config, tokenizer, args)
        logging.info("Training/evaluation parameters %s", args)
        ucc_data_module = UCC_Data_Module(args,tokenizer)
        trainer = pl.Trainer(gpus=[1],max_epochs=args.epochs, num_sanity_val_steps=50,gradient_clip_val=1.0)
        trainer.fit(model, ucc_data_module)



    # Testing
    if args.do_test:
        model = Model(config, tokenizer, args)
        model = model.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            hparams_file=hparams_file,
            map_location=None,
            config=config,
            tokenizer=tokenizer,
            args=args
        )
        ucc_data_module = UCC_Data_Module(args,tokenizer)
        trainer = pl.Trainer(gpus=[1],max_epochs=args.epochs, num_sanity_val_steps=50,gradient_clip_val=1.0)
        trainer.test(model, ucc_data_module)

    # line level vurnerability detection evaluation
    if args.do_linelevel:
        model = Model(config, tokenizer, args)
        model = model.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            hparams_file=hparams_file,
            map_location=None,
            config=config,
            tokenizer=tokenizer,
            args=args
        )
        test_df=pd.read_csv(args.test_data_file)
        ucc_data_module = UCC_Data_Module(args,tokenizer)
        ucc_data_module.setup(stage = 'test')
        dataloader=ucc_data_module.test_dataloader()
        LineLevel(model=model,dataloader=dataloader,tokenizer=tokenizer,test_df=test_df)

    # get vulnerable lines in a file
    if args.file_name:
        model = Model(config, tokenizer, args)
        model = model.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            hparams_file=hparams_file,
            map_location=None,
            config=config,
            tokenizer=tokenizer,
            args=args
        )   
        vul_lines = Filevul(filename=args.file_name,model=model,tokenizer=tokenizer,args=args)
        logging.info("Vulnerable lines in %s file: %s", args.file_name, vul_lines)

    # get vulnerable lines in each file in a directory
    if args.directory_name:
        model = Model(config, tokenizer, args)
        model = model.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            hparams_file=hparams_file,
            map_location=None,
            config=config,
            tokenizer=tokenizer,
            args=args
        )   
        vul_lines = Directoryvul(dirname=args.directory_name,model=model,tokenizer=tokenizer,args=args)
        for file,vul_lines_infile in vul_lines:
            print((file,vul_lines_infile))
            logging.info("Vulnerable lines in %s file: %s", file, vul_lines_infile)  



if __name__ == "__main__":
    main()

    
