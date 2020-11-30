#!/usr/bin/env python3

import random
from pathlib import Path
from argparse import ArgumentParser

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

from datasets import load_dataset

from transformers import (
    BertForMaskedLM, BertConfig, BertTokenizer, 
    RobertaForMaskedLM, RobertaConfig, RobertaTokenizer,
    AutoModelForMaskedLM, AutoConfig, AutoTokenizer,
)

from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling


random.seed(3141)

model_class_dict = {
    'auto': (AutoModelForMaskedLM, AutoConfig, AutoTokenizer),
    'bert': (BertForMaskedLM, BertConfig, BertTokenizer),
    'roberta': (RobertaForMaskedLM, RobertaConfig, RobertaTokenizer),
}


def main():

    parser = ArgumentParser('wfclm')
    parser.add_argument('--model_class', type=str, default='bert')
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--tokenizer', type=str, default='./wfctokenizer')
    
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--num_data_files', type=int, default=None)
    
    parser.add_argument('--train')
    
    args = parser.parse_args()
    
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
                                                max_len=args.max_len,
                                               )

    config = config_class(
#         vocab_size=50_000,
#         max_position_embeddings=(args.max_len+2),
#         max_len=args.max_len,
        num_attention_heads=args.num_heads,
        num_hidden_layers=args.num_layers,
        type_vocab_size=1,
    )
    
    model = model_class(config=config)
    
    print(f'{model_class} model num params:', model.num_parameters())
    
    with open('ls_refdata', 'r') as f:
        reffiles = [*map(lambda name: 'utf-refdata/'+name[:-1], f.readlines())]
        random.shuffle(reffiles)
        
    dataset = load_dataset('text', 
                           data_files=reffiles[:args.num_data_files])
    dataset = dataset.filter(lambda e: len(e['text'])>0)['train']
    dataset = dataset.map(lambda e: tokenizer(e['text'], truncation=True,
                          padding=True), batched=True)
    
    print(dataset[0])
    
    split = dataset.train_test_split(train_size=.8, shuffle=True, seed=3141)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    
    training_args = TrainingArguments(
        output_dir="./wfclm",
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_gpu_train_batch_size=args.batch_size,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=10,
        eval_steps=args.eval_steps,
        do_eval=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=split['train'],
        eval_dataset=split['test'],
#         prediction_loss_only=True,
    )
    
    trainer.train()
    
main()