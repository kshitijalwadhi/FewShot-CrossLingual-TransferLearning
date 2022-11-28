from datasets import Dataset
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import numpy as np
from transformers import AutoTokenizer


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average="weighted")
    precision = precision_score(y_true=labels, y_pred=pred, average="weighted")
    f1 = f1_score(y_true=labels, y_pred=pred, average="weighted")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def encode_batch(examples):
    all_encoded = {"input_ids": [], "attention_mask": []}
    print("Tokenizing data")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    for premise, hypothesis in zip(examples["premise"], examples["hypothesis"]):
        # encode separately to maintain CLS and SEP tokens
        encoded = tokenizer.batch_encode_plus([premise, hypothesis], add_special_tokens=True, pad_to_max_length=True, max_length=100, return_tensors="pt", truncation=True)

        all_encoded["input_ids"].append(encoded["input_ids"].flatten())
        all_encoded["attention_mask"].append(encoded["attention_mask"].flatten())

    return all_encoded


def get_data(lang, data):
    data = data[data["language"] == lang]
    return data


def get_languages(data):
    # data = pd.read_csv("data/train.tsv",sep="\t")
    return data["language"].unique()


def preprocess_dataset(dataset):
    # Encode the input data
    dataset = dataset.map(encode_batch, batched=True)
    # The transformers model expects the target class column to be named "labels"
    dataset = dataset.rename_column("gold_label", "labels")
    # Transform to pytorch tensors and only output the required columns

    # print dataset column names
    print(dataset.column_names)

    dataset.set_format(columns=["input_ids", "attention_mask", "labels"])
    return dataset


def make_dataset(df, lang):
    en_data = get_data(lang, df)
    en_data = en_data.dropna()
    print("Size of data for language", lang, en_data.shape)

    labels = en_data["gold_label"].values
    labels = [0 if label == "entailment" else 1 if label == "neutral" else 2 for label in labels]
    en_data["gold_label"] = labels
    en_data = Dataset.from_pandas(en_data)
    dataset_en = preprocess_dataset(en_data)
    dataset_en = dataset_en.remove_columns(["language", "premise", "hypothesis", "__index_level_0__"])

    return dataset_en


def translate(text_list, lang, translate_model, translate_tokenizer):
    translate_tokenizer.src_lang = lang
    encoded_lang = translate_tokenizer(text_list, return_tensors="pt", padding=True)
    generated_tokens = translate_model.generate(**encoded_lang, forced_bos_token_id=translate_tokenizer.get_lang_id("en"))
    output = translate_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return output


def make_dataset_with_translate(df, lang, translate_model, translate_tokenizer):
    en_data = get_data(lang, df)
    en_data = en_data.dropna()
    print("Size of data for language", lang, en_data.shape)

    labels = en_data["gold_label"].values
    labels = [0 if label == "entailment" else 1 if label == "neutral" else 2 for label in labels]
    en_data["gold_label"] = labels

    premise_list = en_data["premise"].to_list()
    hypothesis_list = en_data["hypothesis"].to_list()

    translated_premise_list = []
    translated_hypothesis_list = []

    BATCH_SIZE = 8
    for i in range(0, len(premise_list), BATCH_SIZE):
        translated_premise_list.extend(translate(premise_list[i : i + BATCH_SIZE], lang, translate_model, translate_tokenizer))
        translated_hypothesis_list.extend(translate(hypothesis_list[i : i + BATCH_SIZE], lang, translate_model, translate_tokenizer))

    en_data["premise"] = translated_premise_list
    en_data["hypothesis"] = translated_hypothesis_list

    en_data = Dataset.from_pandas(en_data)
    dataset_en = preprocess_dataset(en_data)
    dataset_en = dataset_en.remove_columns(["language", "premise", "hypothesis", "__index_level_0__"])

    return dataset_en


def split_dataset(df):
    # split data into 80 and 20 in each language

    languages = df.language.unique()
    train_data = None
    for lang in languages:
        lang_data = df[df.language == lang]
        if train_data is None:
            train_data = lang_data.sample(frac=0.8, random_state=42)
            test_data = lang_data.drop(train_data.index)
        else:
            temp = lang_data.sample(frac=0.8, random_state=42)
            train_data = pd.concat([train_data, temp])
            test_data = pd.concat([test_data, lang_data.drop(temp.index)])

    print("Size of Train data", train_data.shape)
    print("Size of Test data", test_data.shape)

    return train_data, test_data
