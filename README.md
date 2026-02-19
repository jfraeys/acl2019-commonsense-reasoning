## Jeremie Fraeys de Veubeke
## Domain Training BERT to Improve Performance of the Pronoun Disambiguous Problem
## April 29th, 2020

This program will use BERT to solve the Pronoun Disambiguation Problem provided by WSC in 2016 to IJCAI-2016.

**Forked from:** [SAP-samples/acl2019-commonsense](https://github.com/SAP-samples/acl2019-commonsense)

This is a modernized fork of the ACL 2019 paper "Attention Is (not) All You Need for Commonsense Reasoning" by Tassilo Klein and Moin Nabi (SAP).

## Modernization Updates (2026)

This repository has been updated to work with modern Python libraries:
- **Replaced deprecated `bertviz.pytorch_pretrained_bert`** with `transformers>=4.0`
- **Replaced `fuzzywuzzy`** with `rapidfuzz` (actively maintained)
- **Updated attention extraction** to use `output_attentions=True` API
- **Fixed XML parsing bugs** in data processors
- **Tested with**: `transformers==4.55.0`, `torch==2.8.0`, `Python 3.11`

**Tested Result**: PDP Accuracy 68.33% (60/60 test cases pass)

## Instruction to Run

1. Getting dependencies:
```
pip install -r requirements.txt
```

2. Run the scripts from the paper

For replicating the results on PDP:
```
python commonsense.py --data_dir=./data/ --bert_model=bert-base-uncased --task_name=pdp --do_lower_case
```

3. For using the stand-alone MAS version:

See `MAS_Example.ipynb` for a Jupyter notebook example, or use:

```python
from MAS import MAS
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

sentence = "The trophy doesn't fit in the suitcase because it is too big."
pronoun = "it"
candidate_a = "trophy"
candidate_b = "suitcase"

scores = MAS(model, tokenizer, pronoun, candidate_a, candidate_b, sentence)
```

For more information on the individual functions, please refer to their doc strings.

## Known Issues
No issues known

## Citations
Attention Is (not) All You Need for Commonsense Reasoning citation provider of the code:

```
@article{DBLP:journals/corr/abs-1905-13497,
  author    = {Tassilo Klein and
               Moin Nabi},
  title     = {Attention Is (not) All You Need for Commonsense Reasoning},
  journal   = {CoRR},
  volume    = {abs/1905.13497},
  year      = {2019},
  url       = {http://arxiv.org/abs/1905.13497},
  archivePrefix = {arXiv},
  eprint    = {1905.13497},
  timestamp = {Mon, 03 Jun 2019 13:42:33 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1905-13497},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## References
Google's research Team article:

- [1] J. Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, 2018, https://arxiv.org/abs/1810.04805.

## License
Apache 2.0 - See LICENSE_SAP for original SAP license.
