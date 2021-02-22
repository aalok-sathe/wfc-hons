#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning multi-lingual models on XNLI (e.g. Bert, DistilBERT, XLM).
    Adapted from `examples/text-classification/run_glue.py`"""

import pdb

import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
# from torch import linalg as LA
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    BertForFactChecking,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    #
)
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import xnli_compute_metrics as compute_metrics
# from transformers import xnli_output_modes as output_modes
# from transformers import xnli_processors as processors

from datasets import load_dataset, load_from_disk
from data.wfc_processor import wfc_processors as processors
from data.wfc_processor import wfc_output_modes as output_modes
from data.wfc_processor import wfc_convert_examples_to_features

from sentence_transformers import CrossEncoder, SentenceTransformer, util

from colorama import Fore, Back, Style
from blessings import Terminal; T = Terminal()

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


################################################################################
######## begin TRAIN FUNCTION ##################################################
################################################################################
def train(args, train_dataset, model, tokenizer, sbert=None, config=None):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    #!
    ##############################
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=1 or args.train_batch_size)
    ##############################
    print("LEN train data", len([*train_dataloader]))
#     return None, None
        
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility
    
    ############################################################################
    ######## train L00p ########################################################
    
    # total epochs loop
    for epoch_number in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])

        torch.autograd.set_detect_anomaly(True)
        # batches within an epoch loop
        for step, batch in enumerate(epoch_iterator):
        
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            #                                                         1, |E|, 768
            #print(f'{T.bold_yellow_on_black("INFO:")} batch length: {(batch[0].size())}\r', end='')
            
            #### TODO here for DEBUGGING purposes
            #if step >= 10: raise
                
            attentions = []
            instance_outputs = []
            hidden_states = []
            instance_label = None
            
            running_logits = [] #torch.zeros(size=(1,2)).to(args.device)
                
            ######## batched loop
#             minibatch_size = args.batch_size
#             for j in range(0, batch[0].shape[1], minibatch_size):
#                 limits = slice(j, min(j+minibatch_size, batch[0].shape[1]))
                
#                 print(f'batch contains: {[t.shape for t in batch]}')
#                 minibatch = tuple(t[:, limits] for t in batch)
            
            #!!!!!!
            minibatch = batch
#                 print(f'{T.bold_yellow_on_black("Minibatch shape:")} {minibatch[0].shape}')

            model.train()
            # batch = tuple(t.to(args.device) for t in batch) # contains *one* claim

            labels = minibatch[3].view(-1, 1).to(args.device)
            inputs = {"input_ids": minibatch[0].view(-1, 128).to(args.device), 
                      "attention_mask": minibatch[1].view(-1, 128).to(args.device), 
#                           "labels": labels,
                     }
            
            outputs = model(**inputs, output_hidden_states=True)
            # b,2   b,1   12,b,n,768
            logits, attn, hidden = outputs[:3] # model outputs are always tuple in transformers
            # CLS = hidden[-1][:, 0, :]

            weighted = logits * attn.view(-1, 1)
            logits = torch.sum(weighted, dim=0)
                
            instance_label = labels[0]
                            
            loss_fct = torch.nn.functional.cross_entropy
            loss = loss_fct(logits.view(-1, 2), instance_label.view(-1))
    
#             pdb.set_trace()
                
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if False or args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
                
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if False or args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logger.info("Train logging loss %.3f", (tr_loss - logging_loss) / args.logging_steps)
                    logging_loss = tr_loss
                                       

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

################################################################################
######## end TRAIN FUNCTION ####################################################
################################################################################




################################################################################
######## begin EVAL FUNCTION ###################################################
################################################################################

def evaluate(args, model, tokenizer, prefix=""):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1 or args.eval_batch_size)

        [*eval_dataloader]
#         return {"None": None}
    
        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            # batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                labels = batch[3].view(-1, 1).to(args.device); instance_label = labels[0]
                
                inputs = {"input_ids": batch[0].view(-1, 128).to(args.device), 
                          "attention_mask": batch[1].view(-1, 128).to(args.device), 
#                           "labels": batch[3].view(-1, 1).to(args.device),
                         }
#                 if args.model_type != "distilbert":
#                     inputs["token_type_ids"] = (
#                         batch[2] if args.model_type in ["bert"] else None
#                     )  # XLM and DistilBERT don't use segment_ids
                outputs = model(**inputs, output_hidden_states=True)
                
#                 pdb.set_trace()
                logits, attn, hidden = outputs[:3] # model outputs are always tuple in transformers (see doc)
                CLS = hidden[-1][:, 0, :]
            
                weighted = logits * (attn ).view(-1, 1)
                logits = torch.sum(weighted, dim=0)
            
                loss_fct = torch.nn.functional.cross_entropy
                tmp_eval_loss = loss_fct(logits.view(-1, 2), instance_label.view(-1))

                eval_loss = eval_loss + tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            
            
            if preds is None:
                preds = logits.detach().cpu().view(1,-1).numpy()
                out_label_ids = labels.detach().cpu()[0].view(1,-1).numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().view(1,-1).numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu()[0].view(1,-1).numpy(), axis=0)

        pdb.set_trace()

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        else:
            raise ValueError("No other `output_mode` for WFC-en.")
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    pdb.set_trace()            
    
    return results

################################################################################
######## end EVAL FUNCTION #####################################################
################################################################################



################################################################################
######## begin CACHE FUNCTION ##################################################
################################################################################
def load_and_cache_examples(args, task, tokenizer, evaluate=False, sbert_model_name_or_path:str=None):
    '''
        sbert: string model name to pass to sentence_transformers.CrossEncoder
            this instance will be used to rank sentences in the evidence file by similarity
    '''
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    if sbert_model_name_or_path is None:
        sbert = CrossEncoder('cross-encoder/stsb-roberta-large', device=args.device)
    else:
        sbert = CrossEncoder(sbert_model_name_or_path, device=args.device)
        
    processor = processors[task]()
        #(language=args.language, train_language=args.train_language)
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.cache_dir,
        "cached_split={}_tokenizer={}_len={}_task={}_topk={}_sbert={}".format(
            "test" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
            str(args.top_k),
            str(sbert_model_name_or_path)
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        examples = (
            processor.get_test_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        )
        features = wfc_convert_examples_to_features(
            examples,
            tokenizer,
            sbert=sbert,
            top_k=args.top_k,
            max_length=args.max_seq_length,
            label_list=label_list,
            output_mode=output_mode,
        ) # -> list of (list of features with text_a=claim, text_b=ith evidence sentence)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = [torch.tensor([f.input_ids for f in inst], dtype=torch.long) 
                     for inst in features] # each 'inst' is a claim,evidence example
    all_attention_mask = [torch.tensor([f.attention_mask for f in inst], dtype=torch.long) 
                          for inst in features]
    all_token_type_ids = [torch.tensor([f.token_type_ids for f in inst], dtype=torch.long) 
                          for inst in features]
    if output_mode == "classification":
        all_labels = [torch.tensor([f.label for f in inst], dtype=torch.long) 
                      for inst in features]
    else:
        raise ValueError("No other `output_mode` for XNLI.")

#     dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
#     return dataset

    return [*zip(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)]
    ##############
        
#     dataset = map(lambda tup: TensorDataset(*tup), 
#                   zip(all_input_ids, all_attention_mask, all_token_type_ids, all_labels))
#     return list(dataset)


################################################################################
######## end CACHE FUNCTION ####################################################
################################################################################




################################################################################
######## MAIN METHOD ###########################################################
################################################################################

def main():
    
    ############################################################################
    ######## argument parser block #############################################

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default='./data',
        type=str,
#         required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
#     parser.add_argument(
#         "--language",
#         default=None,
#         type=str,
#         required=True,
#         help="Evaluation language. Also train language if `train_language` is set to None.",
#     )
#     parser.add_argument(
#         "--train_language", default=None, type=str, help="Train language if is different of the evaluation language."
#     )
    parser.add_argument(
        "--output_dir",
        default='out',
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default='/scratch/as9kc/wfc',
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", default=True, help="(def. True) unset this using 'False' if using uncased model."
    )

    parser.add_argument("--batch_size", default=8, type=int, help="Batch size")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=5, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
#     parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
#     parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--top_k", type=int, default=3, help='number of sentences to retrieve based on similarity with claim')
    args = parser.parse_args()
    
    ######## argument parser block #############################################
    ############################################################################

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16,
    )

    ############################################################################    
    ######## auxilliary block ##################################################
    
    # Set seed
    set_seed(args)

    # Prepare WFC task
    args.task_name = "wfc"
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir,
    )
    args.model_type = config.model_type
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir,
    )
#     model = AutoModelForSequenceClassification.from_pretrained(
    model = BertForFactChecking.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    
    ############################################################################    
    ######## action items ######################################################
    
    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, config=config)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForFactChecking.from_pretrained(args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = BertForFactChecking.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
