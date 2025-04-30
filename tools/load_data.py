
import datasets
import torch
import transformers
from datasets import load_dataset


#dataset = load_dataset("parquet", data_dir="datasets/SimpleRL-Zoo-Data/simplelr_qwen_level1to4", data_files=data_files, trust_remote_code=True)
#dataset = load_dataset("parquet", data_dir="datasets/SimpleRL-Zoo-Data/simplelr_qwen_level1to4", trust_remote_code=True)
#dataset = load_dataset("parquet", data_dir="datasets/SimpleRL-Zoo-Data/simplelr_qwen_level1to4")
#dataset = load_dataset("parquet", data_files={'test': 'datasets/SimpleRL-Zoo-Data/simplelr_qwen_level1to4/test.parquet'})
#print(dataset)

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

datadir =  '../datasets/SimpleRL-Zoo-Data/simplelr_qwen_level1to4/train.parquet'
dataset = load_simplerl_zoo_data(datadir)
print(dataset)
print(dataset['train'][:5])
