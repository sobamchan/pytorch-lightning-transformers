from typing import Dict
from collections import OrderedDict
from functools import partial

import lineflow as lf
import lineflow.datasets as lfds
import lineflow.cross_validation as lfcv

import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from transformers import BertForSequenceClassification, BertTokenizer, AdamW


MAX_LEN = 256
NUM_LABELS = 2


def preprocess(tokenizer: BertTokenizer, x: Dict) -> Dict:
    # Given two sentences, x["string1"] and x["string2"], this function returns BERT ready inputs.
    inputs = tokenizer.encode_plus(
            x["string1"],
            x["string2"],
            add_special_tokens=True,
            max_length=MAX_LEN,
            )

    # First `input_ids` is a sequence of id-type representation of input string.
    # Second `token_type_ids` is sequence identifier to show model the span of "string1" and "string2" individually.
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
    attention_mask = [1] * len(input_ids)

    # BERT requires sequences in the same batch to have same length, so let's pad!
    padding_length = MAX_LEN - len(input_ids)

    pad_id = tokenizer.pad_token_id
    input_ids = input_ids + ([pad_id] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([pad_id] * padding_length)

    # Super simple validation.
    assert len(input_ids) == MAX_LEN, "Error with input length {} vs {}".format(len(input_ids), MAX_LEN)
    assert len(attention_mask) == MAX_LEN, "Error with input length {} vs {}".format(len(attention_mask), MAX_LEN)
    assert len(token_type_ids) == MAX_LEN, "Error with input length {} vs {}".format(len(token_type_ids), MAX_LEN)

    # Convert them into PyTorch format.
    label = torch.tensor(int(x["quality"])).long()
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)

    # DONE!
    return {
            "label": label,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
            }


def nonefilter(dataset):
    filtered = []
    for x in dataset:
        if x["string1"] is None:
            continue
        if x["string2"] is None:
            continue
        filtered.append(x)
    return lf.Dataset(filtered)


def get_dataloader():
    train = lfds.MsrParaphrase("train")
    train = nonefilter(train)
    test = lfds.MsrParaphrase("test")
    test = nonefilter(test)
    train, val = lfcv.split_dataset_random(train, int(len(train) * 0.8), seed=42)
    batch_size = 8

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
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
    test_dataloader = DataLoader(
            test.map(preprocessor),
            sampler=SequentialSampler(test),
            batch_size=batch_size
            )

    return train_dataloader, val_dataloader, test_dataloader


class Model(pl.LightningModule):

    def __init__(self):
        super(Model, self).__init__()
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=NUM_LABELS)
        self.model = model

        train_dataloader, val_dataloader, test_dataloader = get_dataloader()
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader

    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    "weight_decay_rate": 0.01
                    },
                {
                    "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    "weight_decay_rate": 0.0
                    },
                ]
        optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=2e-5,
                )
        return optimizer

    def training_step(self, batch, batch_idx):
        labels = batch["label"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        loss, _ = self.model(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels
                )

        tqdm_dict = {"train_loss": loss}
        output = OrderedDict({
            "loss": loss,
            "progress_bar": tqdm_dict,
            "log": tqdm_dict
            })

        return output

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        loss, logits = self.model(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels
                )
        labels_hat = torch.argmax(logits, dim=1)

        correct_count = torch.sum(labels == labels_hat)

        if self.on_gpu:
            correct_count = correct_count.cuda(loss.device.index)

        output = OrderedDict({
            "val_loss": loss,
            "correct_count": correct_count,
            "batch_size": len(labels)
            })
        return output

    def validation_end(self, outputs):
        val_acc = sum([out["correct_count"] for out in outputs]).float() / sum(out["batch_size"] for out in outputs)
        val_loss = sum([out["val_loss"] for out in outputs]) / len(outputs)
        tqdm_dict = {
                "val_loss": val_loss,
                "val_acc": val_acc,
                }
        result = {"progress_bar": tqdm_dict, "log": tqdm_dict, "val_loss": val_loss}
        return result

    def test_step(self, batch, batch_idx):
        labels = batch["label"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        loss, logits = self.model(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels
                )
        labels_hat = torch.argmax(logits, dim=1)

        correct_count = torch.sum(labels == labels_hat)

        if self.on_gpu:
            correct_count = correct_count.cuda(loss.device.index)

        output = OrderedDict({
            "test_loss": loss,
            "correct_count": correct_count,
            "batch_size": len(labels)
            })

        return output

    def test_end(self, outputs):
        test_acc = sum([out["correct_count"] for out in outputs]).float() / sum(out["batch_size"] for out in outputs)
        test_loss = sum([out["test_loss"] for out in outputs]) / len(outputs)
        tqdm_dict = {
                "test_loss": test_loss,
                "test_acc": test_acc,
                }
        result = {"progress_bar": tqdm_dict, "log": tqdm_dict}
        return result

    @pl.data_loader
    def train_dataloader(self):
        return self._train_dataloader

    @pl.data_loader
    def val_dataloader(self):
        return self._val_dataloader

    @pl.data_loader
    def test_dataloader(self):
        return self._test_dataloader


if __name__ == "__main__":
    early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.0,
            patience=3,
            verbose=True,
            mode="min"
            )

    trainer = pl.Trainer(
            gpus=1,
            early_stop_callback=early_stop_callback,
            )

    model = Model()

    trainer.fit(model)
    trainer.test()
