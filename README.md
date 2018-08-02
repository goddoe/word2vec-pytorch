# Word2Vec Implementation with PyTorch 

Simple implementation of word2vec with pytorch

## Usage
```
usage: main.py [-h] [--corpus_path CORPUS_PATH] [--context_size CONTEXT_SIZE]
               [--min_word MIN_WORD] [--embed_dim EMBED_DIM]
               [--n_epoch N_EPOCH] [--batch_size BATCH_SIZE]
               [--learning_rate LEARNING_RATE] [--shuffle SHUFFLE]
               [--verbose_iterval VERBOSE_ITERVAL] [--vis VIS]
               [--dim_reduction_type DIM_REDUCTION_TYPE]

preprocess data

optional arguments:
  -h, --help            show this help message and exit
  --corpus_path CORPUS_PATH
                        corpus path
  --context_size CONTEXT_SIZE
                        context size
  --min_word MIN_WORD   minimum word
  --embed_dim EMBED_DIM
                        embedding dimension
  --n_epoch N_EPOCH     context_size
  --batch_size BATCH_SIZE
                        context_size
  --learning_rate LEARNING_RATE
                        learning rate
  --shuffle SHUFFLE     flag shuffle mini batches
  --verbose_iterval VERBOSE_ITERVAL
                        Interval for print loss.
  --vis VIS             Interval for print loss.
  --dim_reduction_type DIM_REDUCTION_TYPE
                        Interval for print loss.
```

## TODO
- [x] CBOW
- [ ] Skip-gram



