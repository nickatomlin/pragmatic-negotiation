Pragmatic Dialogue Agents for Negotiation
=======================

Project based on Facebook AI Research's [Deal or no deal?](https://code.fb.com/applied-machine-learning/deal-or-no-deal-training-ai-bots-to-negotiate/) negotiation dataset. The goal is to build dialog agents
for a multi-issue bargaining task. This code uses pragmatic reasoning based on the rational speech acts
(RSA) model to infer information about the opponent while bargaining.

Written in Dynet. Tensorflow code available in [branch](https://github.com/Designist/Negotiation/tree/tensorflow).

## Directory structure

```
.
├── data        			
│   ├── action         # agreement space only
│   ├── clusters       # cluster assignments 
│   ├── full           # agreement space and goal vector
│   ├── processed
│   ├── raw
│   └── tmp
├── models             # saved Tensorflow models  
├── notebooks
├── reports
├── src                           
│   ├── data           # data processing scripts
│   ├── experiments
│   └── models         # seq2seq and RNN models
└── ...
```

## Run instructions
To run clustering locally:

    $ python src/experiments/main.py --dynet-autobatch 1
    
To run on GPU machine:

    $ source activate dy3-ntomlin
    $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64
    $ nlprun -a dy-ntomlin -m jagupard18 -n clusters -p high -w /u/scr/ntomlin/Negotiation/ "PYTHONPATH=. python src/experiments/main.py --dynet-gpu --dynet-mem 12000"
    
Note that `--dynet-autobatch 1` turns on [automatic minibatching](https://dynet.readthedocs.io/en/latest/minibatch.html), and `--dynet-mem 12000` sets the memory consumption to 12GB. See [more command line options](https://dynet.readthedocs.io/en/latest/commandline.html).
