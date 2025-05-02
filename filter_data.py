# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import pdb
import logging
import os
import sys

import datasets
import torch
import transformers
import numpy as np

from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from accelerate import PartialState
from trl import ModelConfig, TrlParser, get_peft_config, ScriptArguments
from trl.data_utils import maybe_apply_chat_template
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding
)

from transformers.data.data_collator import DataCollatorMixin
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from torch.utils.data import DataLoader
from math_verify import parse


from src.open_r1.configs import GRPOScriptArguments
from src.open_r1.rewards import get_reward_funcs
from src.open_r1.utils import get_tokenizer
from src.open_r1.utils.callbacks import get_callbacks
from src.open_r1.utils.wandb_logging import init_wandb_training
from src.open_r1.trainer import RLOOTrainer, RLOOConfig

logger = logging.getLogger(__name__)

#copy code from handbook run_dpo.py
class RlooDataCollator(DataCollatorMixin):
    """
    """
    def __init__(
        self,
        pad_token_id: int=0,
        return_tensors : str='pt', 
        ):

        self.pad_token_id = pad_token_id
        self.return_tensors = return_tensors

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        new_output=[]
        for example in examples:
            solution = example['answer']

            #verify solution field
            gold_parsed = parse(
                solution,
                extraction_mode="first_match",
            )
            if len(gold_parsed) == 0 :
                print(f"verify failed, gold: {gold_parsed},{solution}")
                continue

            new_output.append(example)

        return new_output

def load_custom_data(script_args, training_args, model_args, tokenizer) :

    # Load the dataset
    print(script_args)
    if 'SimpleRL-Zoo-Data' in script_args.dataset_name:
        def load_simplerl_zoo_data(datadir):
            dataset = load_dataset(
                    "parquet", 
                    data_files={'train': datadir}, 
                    )

            return dataset

        #qwen2.5-0.5B model just using level1to4 dataset
        datadir =  script_args.dataset_name + '/simplelr_qwen_level1to4/train.parquet'
        dataset = load_simplerl_zoo_data(datadir)
        print(dataset)
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    print(dataset['train'][0])
    print(dataset['train'][1])
    print(dataset['train'][2])

    return dataset

def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    ################
    # Load tokenizer
    # Load poliy model and reference policy model
    ################
    #tokenizer = get_tokenizer(model_args, training_args)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    print("tokenizer.pad_token_id")
    print(tokenizer.pad_token_id)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    print(tokenizer.pad_token_id)

    # Load the dataset
    dataset = load_custom_data(script_args, training_args, model_args, tokenizer)
    print(dataset)
    print(dataset['train'][0])
    print(dataset['train'][1])
    print(dataset['train'][2])

    data_collator = RlooDataCollator(pad_token_id=tokenizer.pad_token_id)

    dataloader = DataLoader(
        dataset['train'],
        batch_size=128,
        collate_fn=data_collator,
    )

    mydict = {}
    mydict['answer'] = []
    mydict['question'] = []
    cnt = 0
    #pdb.set_trace()
    for data in dataloader:
        answer = [x['answer'] for x in data]
        question = [x['question'] for x in data]
        
        mydict['answer'].extend(answer)
        mydict['question'].extend(question)
        answer_len = len(answer)
        question_len = len(question)

        cnt += answer_len

    print(len(mydict['answer']))
    print(cnt)
    df = pd.DataFrame(mydict)
    df.to_parquet("data.parquet", engine="pyarrow")

if __name__ == "__main__":
    #parser = TrlParser((ScriptArguments, RLOOConfig, ModelConfig))
    parser = TrlParser((GRPOScriptArguments, RLOOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
