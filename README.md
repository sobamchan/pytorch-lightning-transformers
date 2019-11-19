# pytorch-lightning-transformers [Medium article](https://towardsdatascience.com/how-to-fine-tune-bert-with-pytorch-lightning-ba3ad2f928d2)

## Setup
```bash
$ git clone https://github.com/sobamchan/pytorch-lightning-transformers.git
$ cd pytorch-lightning-transformers
$ pipenv install
$ pipenv shell
```

## Usage

### Fine-tune for [CommonsenseQA](https://www.tau-nlp.org/commonsenseqa) with 2 gpus.
```bash
CUDA_VISIBLE_DEVICES=0,1 python csqa.py --gpus 2
```

### Fine-tune for [MRPC](https://www.microsoft.com/en-us/download/details.aspx?id=52398)
```bash
$ python mrpc.py
```
This will load pre-trained BERT and fine-tune it with putting classification layer on top on MRPC task (paraphrase identification).
