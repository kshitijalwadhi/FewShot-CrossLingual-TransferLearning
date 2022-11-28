from dataset import *
from tokenizer import *
from modules import *


def train_on_one_lang(
    model,
    train_data,
    test_data,
    lang_adapters,
    lang_adapter_config,
    lang,
    learning_rate=3e-4,
    epochs=5,
    batch_size=64,
):

    try:
        # load nli adapter
        # load task adapter
        model.load_adapter("saved_adapters", config=lang_adapter_config, model_name="xlm-roberta-base", load_as="nli")

    except:
        # Add a classification head for our target task
        model.add_classification_head("nli", num_labels=3)

        # Add a new task adapter
        model.add_adapter("nli")

    en_data_train = make_dataset(train_data, lang)
    en_data_test = make_dataset(test_data, lang)

    print("Size of train data", en_data_train.shape)
    print("Size of test data", en_data_test.shape)

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

    training_args = TrainingArguments(learning_rate=learning_rate, num_train_epochs=epochs, per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size, output_dir="./training_output", overwrite_output_dir=True, remove_unused_columns=False, evaluation_strategy="epoch", save_strategy="epoch", metric_for_best_model="f1", load_best_model_at_end=True)

    trainer = AdapterTrainer(model=model, args=training_args, train_dataset=en_data_train, eval_dataset=en_data_test, compute_metrics=compute_metrics, callbacks=[EarlyStoppingCallback(early_stopping_patience=5)])

    print(f"Training on {lang}")

    trainer.train()

    print(f"Finished training on {lang}")

    print("Saving adapter for {}".format(lang))

    model.save_adapter("saved_adapters/", "nli")

    return model
