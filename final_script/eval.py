from trainer import *
from dataset import *
from tokenizer import *
from pipeline import *


def eval_one_lang(model, lang, train_data, test_data, lang_adapters, lang_adapter_config):
    en_data_test = make_dataset(test_data, lang)

    print("Loading language adapter for {}".format(lang))
    try:
        model.load_adapter(lang_adapters[lang], config=lang_adapter_config, model_name="xlm-roberta-base")
        model.active_adapters = Stack(lang, "nli")
    except:
        model.load_adapter(lang_adapters[lang], config=lang_adapter_config, model_name="xlm-roberta-base")
        model.active_adapters = Stack(lang_adapters[lang], "nli")

    eval_trainer = AdapterTrainer(
        model=model,
        args=TrainingArguments(
            output_dir="./eval_output",
            remove_unused_columns=False,
        ),
        eval_dataset=en_data_test,
        compute_metrics=compute_metrics,
    )

    acc = eval_trainer.evaluate()["eval_acc"]

    print(f"couldn't do it {lang}")

    return acc


def eval_one_lang_translate(model, lang, test_data, translate_model, translate_tokenizer):
    en_data_test = make_dataset_with_translate(test_data, lang, translate_model, translate_tokenizer)

    translated_lang = "en"
    print("Loading language adapter for {}".format(translated_lang))
    try:
        model.load_adapter(lang_adapters[translated_lang], config=lang_adapter_config, model_name="xlm-roberta-base")
        model.active_adapters = Stack(translated_lang, "nli")
    except:
        model.load_adapter(lang_adapters[translated_lang], config=lang_adapter_config, model_name="xlm-roberta-base")
        model.active_adapters = Stack(lang_adapters[translated_lang], "nli")

    eval_trainer = AdapterTrainer(
        model=model,
        args=TrainingArguments(
            output_dir="./eval_output",
            remove_unused_columns=False,
        ),
        eval_dataset=en_data_test,
        compute_metrics=compute_metrics,
    )

    acc = eval_trainer.evaluate()["eval_acc"]

    print(f"couldn't do it {lang}")

    return acc
