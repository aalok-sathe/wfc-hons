#!/usr/bin/env python3

import random
import yaml
from pathlib import Path
from argparse import ArgumentParser

from colorama import Fore, Back, Style
from blessings import Terminal; T = Terminal()

import pandas as pd
import torch

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

import datasets# import load_dataset

from transformers import (
    BertForMaskedLM, BertConfig, BertTokenizerFast, 
    RobertaForMaskedLM, RobertaConfig, RobertaTokenizerFast,
    AutoModelForMaskedLM, AutoConfig, AutoTokenizer,
)

from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

torch.manual_seed(3141)
random.seed(3141)

model_class_dict = {
    'auto': (AutoModelForMaskedLM, AutoConfig, AutoTokenizer),
    'bert': (BertForMaskedLM, BertConfig, BertTokenizerFast),
    'roberta': (RobertaForMaskedLM, RobertaConfig, RobertaTokenizerFast),
}


def main():

    parser = ArgumentParser()
    parser.add_argument('--model_class', type=str, default='bert',
                        help='which model class to use ([bert]/roberta/auto)')
    parser.add_argument('--model_name_or_path', type=str, default='bert-base-uncased',
                        help='path to or unique identifier of pretrained model [bert-base-uncased]')
    parser.add_argument('--max_len', type=int, default=128,
                        help='maximum length of the input sequence, in tokens [128]')
    parser.add_argument('--num_heads', type=int, default=6,
                        help='number of heads on language model [6]')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='number of layers in language model [6]')
#     parser.add_argument('--tokenizer', type=str, default='./wfctokenizer',
#                         )
    
    parser.add_argument('--num_epochs', type=int, default=1, help='[1]')
    parser.add_argument('--batch_size', type=int, default=8, help='[8]')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='[1e-5]')
    parser.add_argument('--save_steps', type=int, default=10_000, 
                        help='save every X steps [10_000]')
    parser.add_argument('--eval_steps', type=int, default=5_000,
                        help='evaluate every X steps [1_000]')
    parser.add_argument('--logging_steps', type=int, default=1_000,
                        help='log every X steps [500]')
    parser.add_argument('--num_data_files', type=int, default=5_000,
                        help='how many evidence files to use for training (-1 for all) [5_000]')
    
    parser.add_argument('--overwrite_cache', action='store_true')
    
#     parser.add_argument('--train_size', type=float, default=.8)
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='location of the data [./data]')
    parser.add_argument('--evidence_dir', type=str, default='./utf-refdata',
                        help='location of the evidence files [./utf-refdata]')
    parser.add_argument('--out_dir', type=str, default='./wfclm',
                        help='where to store the trained model and checkpoints [./wfclm]')
    parser.add_argument('--cache_dir', type=str, default='/sw/mcs/wfc',
                        help='location to use for caching of data [/sw/mcs/wfc]')
    
    args = parser.parse_args()
    if args.num_data_files == -1:
        args.num_data_files = None
    
    print(args)
    
    model_class, config_class, tokenizer_class = model_class_dict[args.model_class]

    print(f'{T.bold_yellow_on_black("INFO:")} loading pretrained tokenizer from', 
          args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, 
                                                max_len=args.max_len)

    config = config_class(
#         vocab_size=50_000,
#         max_position_embeddings=(args.max_len+2),
        max_len=args.max_len,
        num_attention_heads=args.num_heads,
        num_hidden_layers=args.num_layers,
        type_vocab_size=1,
    )
    
    model = model_class(config=config).cuda()
    print(f'{model_class} model num params:', model.num_parameters())
    
    train = pd.read_csv(f'{args.data_dir}/wfc_train.tsv', sep='\t')
    dev = pd.read_csv(f'{args.data_dir}/wfc_dev.tsv', sep='\t')
    
    md5s = map(lambda name: args.evidence_dir + '/' + name, list(train['url_md5']))
    dev_md5s = map(lambda name: args.evidence_dir + '/' + name, list(dev['url_md5']))
    
    reffiles = [*sorted(md5s)][:args.num_data_files]
    dev_reffiles = [*dev_md5s]#[:args.num_data_files]
    
    random.seed(3142)
    random.shuffle(reffiles)
    
    print(f'found records for {len(train)} evidence files in the train set')
    print(f'retaining records for {len(reffiles)} evidence files in the train set')
    print(f'found records for {len(dev_reffiles)} evidence files in the dev set')

    saved = Path(f'{args.cache_dir}/datasets/{args.num_data_files}_encoded.dat')
    saved_dev = Path(f'{args.cache_dir}/datasets/{args.num_data_files}_encoded_dev.dat')
    if saved.exists() and not args.overwrite_cache:
        dataset = datasets.load_from_disk(str(saved))
        dev_dataset = datasets.load_from_disk(str(saved_dev))
    else:
        dataset = datasets.load_dataset('text', 
                                        data_files=reffiles[:],
                                        cache_dir=args.cache_dir)
        dev_dataset = datasets.load_dataset('text', 
                                        data_files=dev_reffiles[:],
                                        cache_dir=args.cache_dir)
    
        def tokenize(e):
            return tokenizer(e['text'], truncation=True, padding=True)
    
        dataset = dataset.map(tokenize, batched=True)
        dev_dataset = dev_dataset.map(tokenize, batched=True)
        
        dataset.save_to_disk(f'{args.cache_dir}/datasets/{args.num_data_files}_encoded.dat')
        dev_dataset.save_to_disk(f'{args.cache_dir}/datasets/{args.num_data_files}_encoded_dev.dat')
    
    print(dataset)
    
#     split = dataset.train_test_split(train_size=args.train_size, shuffle=True, seed=3141)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    
    out_dir = f"./{args.out_dir}/{args.model_class}_{args.num_data_files}_lr={args.learning_rate}"
            
    training_args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        do_train=True, do_eval=True, #evaluate_during_training=True,
        
        dataloader_num_workers=4,
        
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        
        logging_steps=args.logging_steps,
        logging_first_step=True,
        
        save_steps=args.save_steps,
        save_total_limit=6,
        
        learning_rate=args.learning_rate,
        
#         eval_steps=args.eval_steps,
#         evaluation_strategy='steps'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dev_dataset['train'],
#         prediction_loss_only=True,
    )
    
    try:
        trainer.train()
    except KeyboardInterrupt as k:
        print('INFO: training aborted by user')
        
    with open(out_dir + '/' + 'files_used.yml', 'w+') as f:
        print(*reffiles, sep='\n', file=f)
    
main()