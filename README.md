# wfc-hons

Continuation of the work [Automated Fact-Checking of Claims from Wikipedia](https://www.aclweb.org/anthology/2020.lrec-1.849/)

- `utf-refdata` contains all UTF-8 formatted evidence files stored as `[md5sum_of_url].txt`
- `transformers_root` contains a frozen copy of Huggingface Transformers repository
- `transformers` is a symlink to `transformers_root/src/transformers`
- `data` contains data obtained from [wikifactcheck-english](https://github.com/wikifactcheck-english/wikifactcheck-english)


# train language model on evidence files
`run_lm_train.py`
