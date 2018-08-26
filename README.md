Pragmatic Dialogue Agents for Negotiation
=======================

Project based on Facebook Research's [Deal or no deal?](https://code.fb.com/applied-machine-learning/deal-or-no-deal-training-ai-bots-to-negotiate/) negotiation dataset. The goal is to build dialog agents
for a multi-issue bargaining task. This code uses pragmatic reasoning based on the rational speech acts
(RSA) model to infer information about the opponent while bargaining.

Sequence model code based on [Stanford CS224U](https://github.com/cgpotts/cs224u).

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
│     	└── agents     # derived model classes for negotiation
└── ...
```
