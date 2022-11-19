import pandas as pd
import torch
from tqdm import tqdm
from transformers.adapters.composition import Stack
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoAdapterModel
from transformers import AdapterConfig
from transformers import TrainingArguments, AdapterTrainer
from datasets import concatenate_datasets

import numpy as np
from transformers import EvalPrediction


def compute_accuracy(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": (preds == p.label_ids).mean()}


def get_data(lang, data):
    data = data[data["language"] == lang]
    return data


def get_languages(data):
    # data = pd.read_csv("data/train.tsv",sep="\t")
    return data["language"].unique()


def encode_batch(examples):
    """Encodes a batch of input data using the model tokenizer."""
    all_encoded = {"input_ids": [], "attention_mask": []}
    # Iterate through all examples in this batch
    for premise, hypothesis in zip(examples["premise"], examples["hypothesis"]):

        # sep_token (str, optional, defaults to "</s>") — The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.'

        premise = [str(premise) + " " + str(hypothesis) for _ in range(3)]
        choices = ["0", "1", "2"]
        encoded = tokenizer(
            premise,
            choices,
            max_length=60,
            truncation=True,
            padding="max_length",
        )
        all_encoded["input_ids"].append(encoded["input_ids"])
        all_encoded["attention_mask"].append(encoded["attention_mask"])
    return all_encoded


def preprocess_dataset(dataset):
    # Encode the input data
    dataset = dataset.map(encode_batch, batched=True)
    # The transformers model expects the target class column to be named "labels"
    dataset = dataset.rename_column("gold_label", "labels")
    # Transform to pytorch tensors and only output the required columns
    dataset.set_format(columns=["input_ids", "attention_mask", "labels"])
    return dataset


# main function

if __name__ == "__main__":

    print("Tokenizing data")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    config = AutoConfig.from_pretrained(
        "xlm-roberta-base",
    )
    model = AutoAdapterModel.from_pretrained(
        "xlm-roberta-base",
        config=config,
    )

    # Load the language adapters
    lang_adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=2)

    # load all language adapters
    lang_adapters = {
        "en": "en/wiki@ukp",
        "vi": "vi/wiki@ukp",
        "de": "de/wiki@ukp",
        "ar": "ar/wiki@ukp",
        "es": "es/wiki@ukp",
        "bg": "bg/wiki@ukp",
        "el": "el/wiki@ukp",
        "th": "th/wiki@ukp",
        "ru": "ru/wiki@ukp",
        "tr": "tr/wiki@ukp",
        "sw": "sw/wiki@ukp",
        "ur": "ur/wiki@ukp",
        "zh": "zh/wiki@ukp",
        "hi": "hi/wiki@ukp",
        "fr": "fr/wiki@ukp",
    }

    print("Reading data")
    df = pd.read_csv("data/train.tsv", sep="\t")
    # sample 0.1
    df = df.sample(frac=0.001, random_state=42)

    print("Size of data", df.shape)

    # Add a classification head for our target task
    model.add_multiple_choice_head("nli", num_choices=3)

    # Add a new task adapter
    model.add_adapter("nli")

    languages = get_languages(df)

    for lang in languages:

        en_data = get_data(lang, df)
        print("Size of data for language", lang, en_data.shape)

        labels = en_data["gold_label"].values
        labels = [0 if label == "entailment" else 1 if label == "neutral" else 2 for label in labels]
        en_data["gold_label"] = labels
        en_data = Dataset.from_pandas(en_data)

        dataset_en = preprocess_dataset(en_data)
        dataset_en = dataset_en.remove_columns(["language", "premise", "hypothesis", "__index_level_0__"])
        print("Loading language adapter for {}".format(lang))
        try:
            model.load_adapter(lang_adapters[lang], config=lang_adapter_config)
            model.train_adapter(["nli"])
            model.active_adapters = Stack(lang, "nli")
        except:
            model.add_adapter(lang_adapters[lang], config=lang_adapter_config)
            model.train_adapter([lang_adapters[lang], "nli"])
            model.save_adapter("adapters_trained/", lang_adapters[lang])
            model.train_adapter(["nli"])
            model.active_adapters = Stack(lang_adapters[lang], "nli")

        training_args = TrainingArguments(
            learning_rate=1e-4,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            logging_steps=100,
            output_dir="./training_output",
            overwrite_output_dir=True,
            # The next line is important to ensure the dataset labels are properly passed to the model
            remove_unused_columns=False,
        )

        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset_en,
        )

        print(f"Training on {lang}")

        trainer.train()

        eval_trainer = AdapterTrainer(
            model=model,
            args=TrainingArguments(
                output_dir="./eval_output",
                remove_unused_columns=False,
            ),
            eval_dataset=dataset_en,
            compute_metrics=compute_accuracy,
        )
        eval_trainer.evaluate()

    model.save_adapter("adapters_trained/", "nli")
