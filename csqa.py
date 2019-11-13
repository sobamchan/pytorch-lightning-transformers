from typing import Dict
from functools import partial
from collections import OrderedDict
from argparse import ArgumentParser
import random

import numpy as np

import lineflow.datasets as lfds
from transformers import BertModel, BertTokenizer, RobertaTokenizer, RobertaModel
from transformers import AdamW, WarmupLinearSchedule

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping


MAX_LEN = 128
NUM_LABELS = 5
label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

model_class_dict = {
        "bert-base-uncased": BertModel,
        "roberta-base": RobertaModel
        }
tokenizer_dict = {
        "bert-base-uncased": BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True),
        "roberta-base": RobertaTokenizer.from_pretrained("roberta-base")
        }


class BERT(nn.Module):

    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def preprocess(tokenizer: BertTokenizer, x: Dict) -> Dict:

    choices_features = []

    for key, option in x["options"].items():
        text_a = x["stem"]
        text_b = option
        inputs = tokenizer.encode_plus(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=MAX_LEN,
                )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1] * len(input_ids)

        pad_token_id = tokenizer.pad_token_id
        padding_length = MAX_LEN - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_id] * padding_length)

        assert len(input_ids) == MAX_LEN, "Error with input length {} vs {}".format(len(input_ids), MAX_LEN)
        assert len(attention_mask) == MAX_LEN, "Error with input length {} vs {}".format(len(attention_mask), MAX_LEN)
        assert len(token_type_ids) == MAX_LEN, "Error with input length {} vs {}".format(len(token_type_ids), MAX_LEN)

        choices_features.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            })

    label = label_map.get(x["answer_key"], -1)
    label = torch.tensor(label).long()

    return {
            "id": x["id"],
            "label": label,
            "input_ids": torch.tensor([cf["input_ids"] for cf in choices_features]),
            "attention_mask": torch.tensor([cf["attention_mask"] for cf in choices_features]),
            "token_type_ids": torch.tensor([cf["token_type_ids"] for cf in choices_features]),
            }


def get_dataloader(model_type, batch_size):
    train = lfds.CommonsenseQA("train")
    val = lfds.CommonsenseQA("dev")

    tokenizer = tokenizer_dict[model_type]
    preprocessor = partial(preprocess, tokenizer)

    train_dataloader = DataLoader(
            train.map(preprocessor),
            sampler=RandomSampler(train),
            batch_size=batch_size
            )
    val_dataloader = DataLoader(
            val.map(preprocessor),
            sampler=SequentialSampler(val),
            batch_size=batch_size
            )

    return train_dataloader, val_dataloader


class Model(pl.LightningModule):

    def __init__(self, args):
        super(Model, self).__init__()
        self.hparams = args

        self.device = torch.device("cuda")

        model = model_class_dict[args.model_type].from_pretrained(args.model_type, num_labels=NUM_LABELS)
        bert = BERT(model)

        self.model = bert.to(self.device)

        train_dataloader, val_dataloader = get_dataloader(args.model_type, args.batch_size)
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.hparams.weight_decay
                },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                }
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_eps)
        scheduler = WarmupLinearSchedule(optimizer, self.hparams.warmup_steps, -1)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        labels = batch["label"]

        logits = self.model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids if self.hparams.model_type != "roberta-base" else None,
                )
        num_choices = input_ids.shape[1]
        reshaped_logits = logits.view(-1, num_choices)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(reshaped_logits, labels)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        tqdm_dict = {"train_loss": loss}
        output = OrderedDict({
                "loss": loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict
                })

        return output

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        labels = batch["label"]

        logits = self.model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids if self.hparams.model_type != "roberta-base" else None,
                )
        num_choices = input_ids.shape[1]
        reshaped_logits = logits.view(-1, num_choices)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(reshaped_logits, labels)

        labels_hat = torch.argmax(reshaped_logits, dim=1)
        correct_count = torch.sum(labels == labels_hat)

        if self.on_gpu:
            correct_count = correct_count.cuda(loss.device.index)
            batch_size = torch.tensor(len(labels)).cuda(loss.device.index)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)
            correct_count = correct_count.unsqueeze(0)
            batch_size = batch_size.unsqueeze(0)

        output = OrderedDict({
                "val_loss": loss,
                "correct_count": correct_count,
                "batch_size": batch_size
                })

        return output

    def validation_end(self, outputs):
        if self.trainer.use_dp:
            val_acc = sum([torch.mean(out["correct_count"].float()) for out in outputs]).float()\
                    /\
                    sum(torch.mean(out["batch_size"].float()) for out in outputs)
            val_loss = sum([torch.mean(out["val_loss"].float()) for out in outputs]) / len(outputs)
        else:
            val_acc = sum([out["correct_count"] for out in outputs]).float() / sum(out["batch_size"] for out in outputs)
            val_loss = sum([out["val_loss"] for out in outputs]) / len(outputs)
        tqdm_dict = {
                "val_loss": val_loss,
                "val_acc": val_acc,
                }
        return {"progress_bar": tqdm_dict, "log": tqdm_dict, "val_loss": val_loss}

    @pl.data_loader
    def train_dataloader(self):
        return self._train_dataloader

    @pl.data_loader
    def val_dataloader(self):
        return self._val_dataloader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--note", type=str, default=None)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--test-run", action="store_true")
    parser.add_argument("--distributed-backend", type=str, default="dp")
    parser.add_argument("--model-type", type=str, default="roberta-base")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--val-check-interval", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-nb-epochs", type=int, default=3000)
    parser.add_argument("--min-nb-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adam-eps", type=float, default=1e-06)
    parser.add_argument("--warmup-steps", type=int, default=150)

    args = parser.parse_args()

    seed = args.seed if args.seed else random.randint(1, 100)
    args.seed = seed
    set_seed(args.seed)

    early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.0,
            patience=args.patience,
            verbose=True,
            mode="min",
            )

    if args.test_run:
        trainer = pl.Trainer(
                gpus=args.gpus,
                early_stop_callback=early_stop_callback,
                val_check_interval=args.val_check_interval,
                distributed_backend=args.distributed_backend,
                default_save_path="./test_run_logs",
                train_percent_check=0.001,
                val_percent_check=0.001,
                max_nb_epochs=1
                )
    else:
        trainer = pl.Trainer(
                gpus=args.gpus,
                early_stop_callback=early_stop_callback,
                val_check_interval=0.5,
                distributed_backend=args.distributed_backend,
                min_nb_epochs=args.min_nb_epochs
                )

    model = Model(args)

    trainer.fit(model)
