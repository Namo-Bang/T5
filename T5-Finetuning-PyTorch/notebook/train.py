import os
import json
import re
import argparse
import time
from collections import Counter
from copy import deepcopy

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch import cuda

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

from rich.table import Column, Table
from rich import box
from rich.console import Console
from rich.live import Live

from utils import *
from metric import calculateF1, split


def display_df(df):
    """display dataframe in ASCII format"""

    console = Console()
    table = Table(Column("source_text", justify="center"), Column("target_text", justify="center"), title="Sample Data",
                  pad_edge=False, box=box.ASCII)

    for i, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])

    console.print(table)


class YourDataSetClass(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the neural network for finetuning the model

    """

    def __init__(self, dataframe, tokenizer, source_len, target_len, source_text, target_text):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = ' '.join(source_text.split())
        target_text = ' '.join(target_text.split())

        source = self.tokenizer.batch_encode_plus([source_text], max_length=self.source_len, pad_to_max_length=True,
                                                  truncation=True, padding="max_length", return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([target_text], max_length=self.summ_len, pad_to_max_length=True,
                                                  truncation=True, padding="max_length", return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


def generate_table() -> Table:
    """Make a new table."""
    table = Table(Column("Epoch", justify="center"),
                  Column("Steps", justify="center"),
                  Column("Loss", justify="center"),
                  title="Training Status", pad_edge=False, box=box.ASCII)
    return table


def train(epoch, tokenizer, model, device, loader, optimizer):
    """
    Function to be called for training with the parameters passed from main function

    """
    model.train()
    with Live(generate_table(), refresh_per_second=4) as live:
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype=torch.long)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, labels=lm_labels)
            loss = outputs[0]

            if _ % 10 == 0:
                training_logger.add_row(str(epoch), str(_), str(loss))
                table = generate_table()
                table.add_row(str(epoch), str(_), str(loss))
                live.update(table)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def validate(epoch, tokenizer, model, device, loader):
    """
    Function to evaluate model for predictions

    """

    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=150,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                     generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
            if _ % 10 == 0:
                console.print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals


def T5Trainer(dataframe, dev_dataframe, source_text, target_text, model_params, output_dir="outputs"):
    """
    T5 trainer

    """

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # logging
    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    # tokenizer.add_tokens(["[SEP]","LC", "QT", "DT", "OG"]) #uncomment to add entity tokens not in vocab
    # model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    # logging
    console.log(f"[Data]: Reading data...\n")

    # Importing the raw dataset
    dataframe = dataframe[[source_text, target_text]]
    dev_dataframe = dev_dataframe[[source_text, target_text]]
    console.log("sleep 2sec for check data input and output\n")
    display_df(dataframe.head(2))
    print(dataframe[source_text].values[0])
    time.sleep(2)

    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest for validation.
    train_size = 0.8
    train_dataset = dataframe
    val_dataset = dev_dataframe

    console.print(f"FULL Dataset: {dataframe.shape}")
    console.print(f"TRAIN Dataset: {train_dataset.shape}")
    console.print(f"TEST Dataset: {val_dataset.shape}\n")

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = YourDataSetClass(train_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"],
                                    model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)
    val_set = YourDataSetClass(val_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"],
                               model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)

    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': model_params["TRAIN_BATCH_SIZE"],
        'shuffle': True,
        'num_workers': 0
    }

    val_params = {
        'batch_size': model_params["VALID_BATCH_SIZE"],
        'shuffle': False,
        'num_workers': 0
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(params=model.parameters(), lr=model_params["LEARNING_RATE"])

    # Training loop
    console.log(f'[Initiating Fine Tuning]...\n')

    pattern = '<[^<>]+:[^<>]+>'
    best_f1 = 0
    best_path = os.path.join(output_dir, "best_model")
    console.log(f'Best Model will be saved at {best_path}\n')
    for epoch in range(model_params["TRAIN_EPOCHS"]):
        train(epoch, tokenizer, model, device, training_loader, optimizer)
        for ep in range(model_params["VAL_EPOCHS"]):
            predictions, actuals = validate(ep, tokenizer, model, device, val_loader)
            final_df = pd.DataFrame({'Generated Text': predictions, 'Actual Text': actuals})
            final_df.to_csv(os.path.join(output_dir, f'predictions{epoch}.csv'))

        # 예측한 Entity set 추출
        pred = final_df['Generated Text'].apply(
            lambda x: re.findall(pattern, x.replace('<extra_id_0>', '').replace('<pad>', ''))).values.tolist()
        label = final_df['Actual Text'].apply(lambda x: re.findall(pattern, x.replace('<pad>', ''))).values.tolist()

        pred2 = [list(map(split, i)) for i in pred]
        label2 = [list(map(split, i)) for i in label]

        predict_golden = zip(pred2, label2)
        p, r, f1 = calculateF1(predict_golden)
        console.log(('-' * 20) + f'Epoch{epoch}' + ('-' * 20) + '\n' + f"precision:{p}\nrecall:{r}\nf1:{f1}")
        if f1 > best_f1:
            console.log(f'Best F1 is updated from {best_f1} to {f1}\n')
            best_f1 = f1
            console.log(f'[Saving Best Model]...\n')
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            with open(output_dir + 'best_val_result.txt', 'w') as f:
                f.write(('-' * 20) + f'Epoch{epoch}' + ('-' * 20) + '\n' + f"precision:{p}\nrecall:{r}\nf1:{f1}")

        console.log(f"[Saving Model]...\n")
        # Saving the model after training
        path = os.path.join(output_dir, f"model_files{epoch}")
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)

    # evaluating test dataset
    console.log(f"[Initiating Validation]...\n")

    console.save_text(os.path.join(output_dir, 'logs.txt'))

    console.log(f"[Validation Completed.]\n")
    console.print(f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n""")
    console.print(
        f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir, 'predictions2.csv')}\n""")
    console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir, 'logs.txt')}\n""")


if __name__ == '__main__':
    # Setting up the device for GPU usage

    device = 'cuda' if cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default='cins')
    parser.add_argument("--output_dir", type=str, default='outputs')
    parser.add_argument("--target_type", type=str, default='target')
    parser.add_argument("--example_num", type=int, default=0)
    parser.add_argument("--example_type", type=str, default='easy')
    parser.add_argument("--model_size", type=str, default='base')
    parser.add_argument("--use_cleaning", action="store_true")
    parser.add_argument("--ner_position", type=str, default="input")
    args = parser.parse_args()

    list_dir = os.listdir()
    if args.output_dir not in list_dir:
        os.mkdir('./' + args.output_dir)

    output_dir = './' + args.output_dir + '/'
    console = Console(record=True)
    if args.use_cleaning:
        with open('./data_learn/klue_ner_train_80_cleaning.t', 'r', encoding='utf-8') as f:
            train_data = f.readlines()
    else:
        with open('./data_learn/klue_ner_train_80.t', 'r', encoding='utf-8') as f:
            train_data = f.readlines()

    train_processed = preprocessing(train_data)
    result = train_test_split(train_processed, seed=42, return_split=True)

    train_data, dev = result['train'], result['dev']
    # augmentation is not permitted
    # augmented = entity_mixing(train,leq_standard=5, rounds=10, seed=42)
    print(f"# of train data: {len(train_data)}")
    print(f"# of dev data: {len(dev)}")
    # print(f"# of augmented data: {len(augmented)}")
    # print(f" sum of train and augmented: {len(train)+len(augmented)}")

    # train += augmented # uncomment this line to use augmented data
    if args.example_num != 0:
        if args.example_type == 'easy':
            example = deepcopy(train_data[:args.example_num])
            train_data = train_data[args.example_num:]
        elif args.example_type == 'hard':
            example = deepcopy(train_data[-args.example_num:])
            train_data = train_data[:-args.example_num]
        example = [(i['input'], i['text']) if args.target_type == 'target' else (i['input'], i['target']) for i in
                   example]
        with open(output_dir + 'example.json', 'w') as f:
            console.log("Example\n")
            for a, b in example:
                console.log(f"{a.strip()} -> {b.strip()}\n")
            json.dump(example, f, ensure_ascii=False)
            console.log("Example saved!\n")
        pm = PromptMaker(example=example, args=args)
    else:
        pm = PromptMaker(args=args)
    df = pm.get_df(args.prompt, train_data)
    dev_df = pm.get_df(args.prompt, dev)

    # define a rich console logger

    training_logger = Table(Column("Epoch", justify="center"),
                            Column("Steps", justify="center"),
                            Column("Loss", justify="center"),
                            title="Training Status", pad_edge=False, box=box.ASCII)

    model_params = {
        "MODEL": "./model/kt-ulm-base/" if args.model_size == 'base' else "./model/kt-ulm-small/",
        # model_type: kt-ulm-base
        "TRAIN_BATCH_SIZE": 8,  # training batch size
        "VALID_BATCH_SIZE": 8,  # validation batch size
        "TRAIN_EPOCHS": 3,  # number of training epochs
        "VAL_EPOCHS": 1,  # number of validation epochs
        "LEARNING_RATE": 1e-4,  # learning rate
        "MAX_SOURCE_TEXT_LENGTH": 256 if args.prompt == 'driven' else 768 if args.example_type == 'hard' else 512,
        # max length of source text
        "MAX_TARGET_TEXT_LENGTH": 256,  # max length of target text
        "SEED": 42  # set seed for reproducibility

    }
    console.log(f"{'-' * 20}model_params{'-' * 20}\n{model_params}")
    T5Trainer(dataframe=df, dev_dataframe=dev_df, source_text="prompt", target_text=args.target_type,
              model_params=model_params,
              output_dir=output_dir)