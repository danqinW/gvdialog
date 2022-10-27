## GVDialog
----
**GVDialog**: Modeling Global Latent Semantic in Multi-turn Conversations with  Random Context Reconstruction

### Requirement
 - pytorch >= 1.11
 - rouge >= 1.0.1
 - nltk >= 3.6.5
 - transformers >= 4.19.2

### data preparation
prepare three input files: train.txt, valid.txt, test.txt <br>
prepare data in file like this: <br>
> lieutenant? \_\_eou\_\_ i came to say goodbye. \_\_eou\_\_ you just missed them.<br>
>Hello How are you? \_\_eou\_\_ Good, you? \_\_eou\_\_ I'm fine, what's new?

### run code
run training:
```shell
python train.py
    --model gvdialog \
    --task gv \
    --corpus-path path_to_your_corpus \
    --cache_dir path_to_your_cache  \
    --use-latent \
    --num-encoder-blocks 4 \
    --num-decoder-blocks 6
```

run evaluating:
```shell
python infer_gv.py
    --model gvdialog \
    --task gv \
    --vocab-path path_to_your_vocab \
    --cache_dir path_to_your_cache  \
    --use-latent \
    --num-encoder-blocks 4 \
    --num-decoder-blocks 6
```