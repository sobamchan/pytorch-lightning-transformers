# pytorch-lightning-transformers

## Setup
```bash
$ git clone https://github.com/sobamchan/pytorch-lightning-transformers.git
$ cd pytorch-lightning-transformers
$ pipenv install
$ pipenv shell
```

## Usage

### Fine-tune for [MRPC](https://www.microsoft.com/en-us/download/details.aspx?id=52398)
```bash
$ python mrpc.py
```
This will load pre-trained BERT and fine-tune it with putting classification layer on top on MRPC task (paraphrase identification).
