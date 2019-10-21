from typing import Dict
from functools import partial
from collections import OrderedDict

import lineflow.datasets as lfds
from transformers import BertForMultipleChoice, BertTokenizer, AdamW

import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping


MAX_LEN = 128
NUM_LABELS = 5
label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


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
            "label": label,
            "input_ids": torch.tensor([cf["input_ids"] for cf in choices_features]),
            "attention_mask": torch.tensor([cf["attention_mask"] for cf in choices_features]),
            "token_type_ids": torch.tensor([cf["token_type_ids"] for cf in choices_features]),
            }


def get_dataloader():
    train = lfds.CommonsenseQA("train")
    val = lfds.CommonsenseQA("dev")
    test = lfds.CommonsenseQA("test")

    batch_size = 4

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
        model = BertForMultipleChoice.from_pretrained("bert-base-uncased", num_labels=NUM_LABELS)
        self.model = model

        train_dataloader, val_dataloader, test_dataloader = get_dataloader()
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        weight_decay = 0.0
        adam_epsilon = 1e-8

        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay
                },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                }
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=adam_epsilon)

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
            "log": tqdm_dict,
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
        return {"progress_bar": tqdm_dict, "log": tqdm_dict, "val_loss": val_loss}

    # def test_step(self, batch, batch_idx):
    #     pass

    # def test_end(self, outputs):
    #     pass

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
            mode="min",
            )

    trainer = pl.Trainer(
            gpus=1,
            early_stop_callback=early_stop_callback,
            # train_percent_check=0.1,
            # val_percent_check=0.1
            )

    model = Model()

    trainer.fit(model)
