#!/usr/bin/env python3

import random
from pathlib import Path
from argparse import ArgumentParser

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

    parser = ArgumentParser('wfclm')
    parser.add_argument('--model_class', type=str, default='bert')
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--tokenizer', type=str, default='./wfctokenizer')
    
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--eval_steps', type=int, default=50)
    parser.add_argument('--logging_steps', type=int, default=50)
    parser.add_argument('--num_data_files', type=int, default=None)
    
    parser.add_argument('--overwrite_cache', action='store_true')
    
    parser.add_argument('--train_size', type=float, default=.8)
    parser.add_argument('--data_dir', type=str, default='./utf-refdata')
    
    args = parser.parse_args()
    print(args)
    
    model_class, config_class, tokenizer_class = model_class_dict[args.model_class]
    
#     tokenizer = ByteLevelBPETokenizer(
#         (args.tokenizer) + '/' "vocab.json",
#         (args.tokenizer) + '/' + "merges.txt",
#     )
#     tokenizer._tokenizer.post_processor = BertProcessing(
#         ("</s>", tokenizer.token_to_id("</s>")),
#         ("<s>", tokenizer.token_to_id("<s>")),
#     )
#     tokenizer.enable_truncation(max_length=args.max_len)

    print('INFO: loading tokenizer from', args.tokenizer)
    tokenizer = tokenizer_class.from_pretrained('bert-base-uncased', 
                                                max_len=args.max_len)

    config = config_class(
#         vocab_size=50_000,
#         max_position_embeddings=(args.max_len+2),
#         max_len=args.max_len,
        num_attention_heads=args.num_heads,
        num_hidden_layers=args.num_layers,
        type_vocab_size=1,
    )
    
    model = model_class(config=config).cuda()
    print(f'{model_class} model num params:', model.num_parameters())
    
    train = pd.read_csv('data/wfc_train.tsv', sep='\t')
    md5s = map(lambda name: args.data_dir + '/' + name, list(train['url_md5']))
    reffiles = [*md5s][:args.num_data_files]
    
    random.seed(3142)
    random.shuffle(reffiles)
    
    print(f'found records for {len(train)} evidence files in the train set')
    print(f'retaining records for {len(reffiles)} evidence files in the train set')

    saved = Path(f'/sw/mcs/wfc/datasets/{args.num_data_files}_encoded.dat')
    if saved.exists() and not args.overwrite_cache:
        dataset = datasets.load_from_disk(str(saved))
    else:
        dataset = datasets.load_dataset('text', 
                                        data_files=reffiles[:],
                                        cache_dir='/sw/mcs/wfc')
#         dataset = dataset.filter(lambda e: len(e['text'])>0)['train']
    
        def tokenize(e):
            return tokenizer(e['text'], truncation=True, padding=True)
    
        dataset = dataset.map(tokenize, batched=True, #with_indices=True
                              #new_fingerprint=f'{args.num_data_files}_encoded'
                             )
        
        dataset.save_to_disk(f'/sw/mcs/wfc/datasets/{args.num_data_files}_encoded.dat')
    
#     dataset.set_format(type='torch', 
#                        columns=['input_ids', 'token_type_ids', 'attention_mask'])#, 'label'])
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
    print(dataset)
    
#     split = dataset.train_test_split(train_size=args.train_size, shuffle=True, seed=3141)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    
    training_args = TrainingArguments(
        output_dir="./wfclm",
        overwrite_output_dir=True,
        do_train=True, #do_eval=True, evaluate_during_training=True,
        
        num_train_epochs=args.num_epochs,
        per_gpu_train_batch_size=args.batch_size,
        
        logging_steps=args.logging_steps,
        logging_first_step=True,
        
        save_steps=args.save_steps,
        save_total_limit=5,
        
#         eval_steps=args.eval_steps,
#         evaluation_strategy='steps'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
#         tokenizer=tokenizer,
        train_dataset=dataset['train'],
#         eval_dataset=split['test'],
#         prediction_loss_only=True,
    )
    
    trainer.train()
    
main()