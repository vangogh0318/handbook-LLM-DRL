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

from src.open_r1.configs import GRPOScriptArguments
from src.open_r1.rewards import get_reward_funcs
from src.open_r1.utils import get_tokenizer
from src.open_r1.utils.callbacks import get_callbacks
from src.open_r1.utils.wandb_logging import init_wandb_training
from src.open_r1.trainer import RLOOTrainer, RLOOConfig

logger = logging.getLogger(__name__)

def pad(tensors: List[torch.Tensor], padding_value: int = 0, padding_side: str = "right") -> torch.Tensor:
    """
    Pads a list of tensors to the same shape along the first dimension.

    Args:
        tensors (`List[torch.Tensor]`):
            List of input tensors to pad.
        padding_value (`int`):
            Value to use for padding. Default is 0.
        padding_side (`str`):
            Side on which to add padding. Must be 'left' or 'right'. Default is 'right'.

    Returns:
        `torch.Tensor`:
            A single tensor containing the padded tensors.

    Examples:
        >>> import torch
        >>> pad([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
        tensor([[1, 2, 3],
                [4, 5, 0]])
        >>> pad([torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])])
        tensor([[[1, 2],
                [3, 4]],

                [[5, 6],
                [0, 0]]])
    """
    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    # Create an output tensor filled with the padding value
    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        # Determine the slice for the sequence dimension
        if padding_side == "left":
            seq_slice = slice(output_shape[0] - t.shape[0], output_shape[0])
        elif padding_side == "right":
            seq_slice = slice(0, t.shape[0])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output

#copy code from handbook run_dpo.py
class RlooDataCollatorOld(DataCollatorMixin):
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
        # Convert to tensor
        prompt_input_ids = [torch.tensor(example["prompt_input_ids"]) for example in examples]
        prompt_attention_mask = [torch.ones_like(input_ids) for input_ids in prompt_input_ids]
        
        solution = [example["solution"] for example in examples]
        prompt = [example["prompt"] for example in examples]

        # Pad
        output = {}
        output["prompt_input_ids"] = pad(prompt_input_ids, padding_value=self.pad_token_id, padding_side="left")
        output["prompt_attention_mask"] = pad(prompt_attention_mask, padding_value=0, padding_side="left")

        output["solution_ids"] = solution
        output["prompt"] = prompt

        return output

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
        # Convert to tensor
        prompt_input_ids = [torch.tensor(example["prompt_input_ids"]) for example in examples]
        prompt_attention_mask = [torch.ones_like(input_ids) for input_ids in prompt_input_ids]
        
        solution = [example["solution"] for example in examples]
        prompt = [example["prompt"] for example in examples]

        # Pad
        output = {}
        output["prompt_input_ids"] = pad(prompt_input_ids, padding_value=self.pad_token_id, padding_side="left")
        output["prompt_attention_mask"] = pad(prompt_attention_mask, padding_value=0, padding_side="left")

        output["solution"] = solution
        output["prompt"] = prompt

        new_output = []
        for prompt_input_ids, prompt_attention_mask, solution, prompt in zip(output["prompt_input_ids"], 
                        output["prompt_attention_mask"],
                        output["solution"],
                        output["prompt"]):
            current_dict = {}
            current_dict["prompt_input_ids"] = prompt_input_ids
            current_dict["prompt_attention_mask"] = prompt_attention_mask
            current_dict["solution"] = solution
            current_dict["prompt"] = prompt
            new_output.append(current_dict)

        return new_output

def load_custom_data(script_args, training_args, model_args, tokenizer) :

    # Load the dataset
    print(script_args)
    if 'SimpleRL-Zoo-Data' in script_args.dataset_name:
        def load_simplerl_zoo_data(datadir):
            dataset = load_dataset(
                    "parquet", 
                    data_files={'train': datadir}, 
                    #streaming=True, 
                    #columns=['answer', 'question']
                    )
            rm_cols = ['gt_answer', 'subject', 'level', 'target', 'data_source', 'prompt', 'ability', 'reward_model', 'extra_info']

            current_dict = {}
            current_dict["prompt_input_ids"] = pad(prompt_input_ids, padding_value=self.pad_token_id, padding_side="left")
            current_dict["prompt_attention_mask"] = pad(prompt_attention_mask, padding_value=0, padding_side="left")

            current_dict["solution_ids"] = solution
            current_dict["prompt"] = prompt
            output.append(current_dict)

        return output

def load_custom_data(script_args, training_args, model_args, tokenizer) :

    # Load the dataset
    print(script_args)
    if 'SimpleRL-Zoo-Data' in script_args.dataset_name:
        def load_simplerl_zoo_data(datadir):
            dataset = load_dataset(
                    "parquet", 
                    data_files={'train': datadir}, 
                    #streaming=True, 
                    #columns=['answer', 'question']
                    )
            rm_cols = ['gt_answer', 'subject', 'level', 'target', 'data_source', 'prompt', 'ability', 'reward_model', 'extra_info']

            dataset=dataset.remove_columns(rm_cols)
            dataset=dataset.rename_column("answer", "solution")
            dataset=dataset.rename_column("question", "problem")

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

    # Format into conversation
    def make_conversation(example, prompt_column: str = script_args.dataset_prompt_column):
        prompt = []

        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})

        if prompt_column not in example:
            raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")

        prompt.append({"role": "user", "content": example[prompt_column]})
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)
    print("add prompt")
    print("add prompt")
    print("add prompt")
    print(dataset['train'][0])
    print(dataset['train'][1])
    print(dataset['train'][2])

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    #add prepare_dataset funtion, to tokenize inputs before invoking trainer.
    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            prompt_text = maybe_apply_chat_template(element, tokenizer)['prompt']
            prompt_inputs = tokenizer(
                text=prompt_text,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False
            )
            prompt_ids, prompt_mask = prompt_inputs["input_ids"][0], prompt_inputs["attention_mask"][0]

            prompt_len = len(prompt_ids)
            if prompt_len > training_args.max_prompt_length:
                prompt_len = training_args.max_prompt_length
                prompt_ids = prompt_ids[:training_args.max_prompt_length]

            return {"prompt_input_ids": prompt_ids}

        return dataset.map(
            tokenize,
            num_proc=training_args.dataset_num_proc,
        )

    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        dataset = prepare_dataset(dataset, tokenizer)
        #dataset = dataset.filter(lambda x: x["lengths"] <= 512, num_proc=training_args.dataset_num_proc)

    #assert dataset[0]["input_ids"][-1] != tokenizer.eos_token_id, "The last token should not be an EOS token"
    print("after tokenizer")
    print("after tokenizer")
    print("after tokenizer")
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

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    ################
    # Load tokenizer
    # Load poliy model and reference policy model
    ################
    #tokenizer = get_tokenizer(model_args, training_args)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    #
    ref_policy = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    policy = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )

    # Load the dataset
    dataset = load_custom_data(script_args, training_args, model_args, tokenizer)

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        #use_cache=False if training_args.gradient_checkpointing else True,
        use_cache=False
    )
    training_args.model_init_kwargs = model_kwargs

    # Get reward functions from the registry
    reward_funcs = get_reward_funcs(script_args)

    #############################
    # Initialize the GRPO trainer
    #############################
    #print(training_args)
    #print(model_kwargs)

    #def data_collator(features):  # No data collation is needed in GRPO
    #    return features
    data_collator = RlooDataCollator(pad_token_id=tokenizer.pad_token_id)

    trainer = RLOOTrainer(
        config=training_args,
        processing_class=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_funcs,
        data_collator=data_collator,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

if __name__ == "__main__":
    #parser = TrlParser((ScriptArguments, RLOOConfig, ModelConfig))
    parser = TrlParser((GRPOScriptArguments, RLOOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
