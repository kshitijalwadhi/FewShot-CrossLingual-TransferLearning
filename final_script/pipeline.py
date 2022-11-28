from tokenizer import *
from dataset import *
from trainer import *
from eval import *
from modules import *

if __name__ == "__main__":

    config = AutoConfig.from_pretrained(
        "xlm-roberta-base",
    )
    model = AutoAdapterModel.from_pretrained(
        "xlm-roberta-base",
        config=config,
    )

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

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

    DATA_PATH = "/Users/navyajain/Desktop/IIT/Sem-7/COL772/A3/FewShot-CrossLingual-TransferLearning/data/train.tsv"

    print("Reading data")
    df = pd.read_csv(DATA_PATH, sep="\t")
    df = df.sample(frac=0.01, random_state=42)

    train_data, test_data = split_dataset(df)

    # TRAINING LOOP

    languages = get_languages(train_data)
    print("Languages", languages)

    # VALUES
    BATCH_SIZE = 64
    EPOCHS = 5
    LEARNING_RATE = 3e-4

    for lang in languages:

        model = train_on_one_lang(
            model,
            train_data,
            test_data,
            lang_adapters,
            lang_adapter_config,
            lang,
            learning_rate=LEARNING_RATE,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
        )

    model.save_pretrained("tuned_model")

    # EVALUATION LOOP

    print("Evaluating on test data")

    config = AutoConfig.from_pretrained(
        "xlm-roberta-base",
    )

    model = AutoAdapterModel.from_pretrained(
        "tuned_model",
        config=config,
    )

    # load task adapter
    model.load_adapter("saved_adapters", config=lang_adapter_config, model_name="xlm-roberta-base", load_as="nli")

    model.active_adapters = Stack("nli")

    acc = {}

    for lang in languages:
        acc[lang] = eval_one_lang(model, lang, train_data, test_data, lang_adapters, lang_adapter_config)

    print(acc)
