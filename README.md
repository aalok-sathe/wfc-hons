# wfc-hons

Continuation of the work [Automated Fact-Checking of Claims from Wikipedia](https://www.aclweb.org/anthology/2020.lrec-1.849/)

- `utf-refdata` contains all UTF-8 formatted evidence files stored as `[md5sum_of_url].txt`
- `transformers_root` contains a frozen copy of Huggingface Transformers repository
- `transformers` is a symlink to `transformers_root/src/transformers`
- `data` contains data obtained from [wikifactcheck-english](https://github.com/wikifactcheck-english/wikifactcheck-english)


# train language model on evidence files
`run_lm_train.py`

```python
usage: run_lm_train.py
             [-h] [--model_class MODEL_CLASS] [--max_len MAX_LEN]
             [--num_heads NUM_HEADS] [--num_layers NUM_LAYERS]
             [--tokenizer TOKENIZER] [--num_epochs NUM_EPOCHS]
             [--batch_size BATCH_SIZE] [--save_steps SAVE_STEPS]
             [--eval_steps EVAL_STEPS] [--logging_steps LOGGING_STEPS]
             [--num_data_files NUM_DATA_FILES] [--train TRAIN]

optional arguments:
  -h, --help            show this help message and exit
  --model_class MODEL_CLASS
  --max_len MAX_LEN
  --num_heads NUM_HEADS
  --num_layers NUM_LAYERS
  --tokenizer TOKENIZER
  --num_epochs NUM_EPOCHS
  --batch_size BATCH_SIZE
  --save_steps SAVE_STEPS
  --eval_steps EVAL_STEPS
  --logging_steps LOGGING_STEPS
  --num_data_files NUM_DATA_FILES
  --train TRAIN```
