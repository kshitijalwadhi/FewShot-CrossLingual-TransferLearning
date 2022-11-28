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
from transformers import EarlyStoppingCallback, IntervalStrategy
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
