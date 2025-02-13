import json
import logging
import os
import argparse
import random
import subprocess
import sys

import datasets
import pandas as pd
import torch
import yaml
from datasets import load_dataset
from lm_eval.utils import apply_template
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import utils

# 设置日志
logging.basicConfig(level=logging.INFO)

# 设置路径
DATA_PATH = f"./data"

# 创建必要的目录
os.makedirs(f"{DATA_PATH}/json", exist_ok=True)
os.makedirs(f"{DATA_PATH}/yaml", exist_ok=True)
os.makedirs(f"{DATA_PATH}/json", exist_ok=True)
os.makedirs(f"{DATA_PATH}/loss", exist_ok=True)
os.makedirs(f"{DATA_PATH}/models", exist_ok=True)
os.makedirs(f"{DATA_PATH}/partitioned", exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def generate_training_data(train_data_dic, total_samples, dataset_alias):
    all_data = []
    for source_name,data_lst in train_data_dic.items():
        all_data.extend(data_lst)

    if len(all_data) < total_samples:
        logging.warning(f"Not enough data. Using all available {len(all_data)} samples.")
        selected_samples = all_data
    else:
        selected_samples = random.sample(all_data, total_samples)

    training_data = selected_samples
    random.shuffle(training_data)

    # json_path = f"data/json/{dataset_alias}.json"
    # with open(json_path, 'w', encoding='utf-8') as f:
    #     json.dump(training_data, f, indent=4, ensure_ascii=False)

    # 处理成jsonl格式
    json_path = f"data/json/{dataset_alias}.jsonl"
    with open(json_path, 'w', encoding='utf-8') as f:
        for sample in training_data:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')

    return training_data, f"json/{dataset_alias}.jsonl"


def update_dataset_info(json_path, dataset_alias):
    dataset_info_path = f"data/dataset_info.json"
    if not os.path.exists(dataset_info_path):
        with open(dataset_info_path, 'w', encoding='utf-8') as f:
            json.dump({}, f, indent=4, ensure_ascii=False)
    with open(dataset_info_path, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)

    logging.info(f"Updating dataset info for {dataset_alias} with file path {json_path}")
    dataset_info[dataset_alias] = {
        "file_name": json_path,
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        }
    }

    with open(dataset_info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=4, ensure_ascii=False)


def create_full_yaml_config(dataset_alias, model_path, output_path):
    if "llama-3" in model_path.lower():
        template = "llama3"
    elif "llama-2" in model_path.lower():
        template = "llama2"
    else:
        template = "chatml"

    config = {
        "model_name_or_path": model_path,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "full",
        # "finetuning_type": "lora",
        # "lora_target": "all",
        "deepspeed": "./config/ds_z3_config.json",
        "dataset": dataset_alias,
        "template": template,
        "cutoff_len": 4096,
        "overwrite_cache": True,
        "preprocessing_num_workers": 64,
        "output_dir": output_path,
        "logging_steps": 10,
        "save_total_limit": 1,
        "save_strategy": "epoch",
        # "save_steps": 10000,
        "plot_loss": True,
        "overwrite_output_dir": True,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 128//(torch.cuda.device_count() * 4),
        "learning_rate": 2e-6,
        "num_train_epochs": 1.0,
        "lr_scheduler_type": "linear",
        # "lr_scheduler_kwargs": {'min_lr': 1.0e-6},
        "warmup_ratio": 0.03,
        "bf16": True,
        "ddp_timeout": 180000000,
        "report_to": "wandb",
        "run_name": dataset_alias + "_" + model_path.replace("/", "-"),
        "save_only_model": True,
    }
    return config


def train_model(json_path, model_path, output_path, dataset_alias):
    update_dataset_info(json_path, dataset_alias)

    config_yaml_path = f"{DATA_PATH}/yaml/training_config_{dataset_alias}_{model_path.replace('/', '-')}.yaml"
    config = create_full_yaml_config(dataset_alias, model_path, output_path)
    with open(config_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)

    logging.info(f"Training model cmd: DISABLE_VERSION_CHECK=1 MKL_THREADING_LAYER=GNU  llamafactory-cli train {config_yaml_path}")
    cmd = f"FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 llamafactory-cli train {config_yaml_path}"

    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end='')
        sys.stdout.flush()
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)



def calculate_losses(model_path, val_data, batch_size, lora_path=None):
    logging.info(f"Loading model from {model_path}")
    if not lora_path:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        model.load_adapter(lora_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    logging.info("Model loaded successfully")

    avg_losses = {}
    for task_name, task_data in tqdm(val_data.items(), desc="Task"):
        logging.info(f"Processing task {task_name}")
        task_losses = []

        for i in tqdm(range(0, len(task_data), batch_size), desc=f"Calculating losses for {task_name}"):
            batch_data = task_data[i:i + batch_size]

            inputs = []
            answer_starts = []
            for data in batch_data:
                full_input = data["data"]
                inputs.append(full_input)
                answer_start = full_input.find("Answer:")
                answer_starts.append(answer_start if answer_start != -1 else 0)

            tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
            input_ids = tokenized_inputs.input_ids.to(model.device)
            attention_mask = tokenized_inputs.attention_mask.to(model.device)

            labels = input_ids.clone()
            for j, start in enumerate(answer_starts):
                answer_token_start = \
                    tokenizer(inputs[j][:start], return_tensors="pt", add_special_tokens=False).input_ids.shape[1]
                labels[j, :answer_token_start] = -100

            with torch.inference_mode():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logging.info(f"Batch {i}: {loss}")
                task_losses.append(loss.item() * input_ids.shape[0])

        avg_losses[task_name] = sum(task_losses) / len(task_data)
        logging.info(f"Task {task_name} average loss: {avg_losses[task_name]}")

    del model
    torch.cuda.empty_cache()  # Clear CUDA cache after finishing all tasks
    return avg_losses


def load_specific(dataname, dic):
    if dataname == "science":
        file_name = f"./data/science_train.jsonl"
        with open(file_name, 'r', encoding='utf-8') as f:
            dic["science"] = []
            for line in f:
                line = json.loads(line)
                dic["science"].append(
                    {
                        "conversations": [
                            {"role": "user", "content": line["input"]},
                            {"role": "assistant", "content": line["output"]}
                        ]
                    }
                )
        return dic

    elif dataname == "OpenOrca":
        file_name = f"./data/1M-GPT4-Augmented.parquet"
        df = pd.read_parquet(file_name)
        dic["OpenOrca"] = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing OpenOrca data"):
            dic["OpenOrca"].append(
                {
                    "conversations": [
                        {"role": "system", "content": row["system_prompt"]},
                        {"role": "user", "content": row["question"]},
                        {"role": "assistant", "content": row["response"]}
                    ]
                }
            )
        return dic
    elif dataname == "tulu":
        # allenai/tulu-v2-sft-mixture
        # 用dataset加载数据到本地
        dataset = load_dataset("allenai/tulu-v2-sft-mixture")

        # Process each example in the dataset
        for example in tqdm(dataset['train'], desc="Processing Tulu data"):
            sub_name = example["dataset"]
            conversation = example["messages"]
            if sub_name not in dic:
                dic[sub_name] = []
            dic[sub_name].append(
                {
                    "conversations": conversation
                }
            )
        return dic


    elif dataname == "tulu3":
        # allenai/tulu-v3-sft-mixture
        # 用dataset加载数据到本地
        dataset = load_dataset("allenai/tulu-3-sft-mixture")

        # Process each example in the dataset
        for example in tqdm(dataset['train'], desc="Processing Tulu data"):
            sub_name = example["source"]
            conversation = example["messages"]
            if sub_name not in dic:
                dic[sub_name] = []
            dic[sub_name].append(
                {
                    "conversations": conversation
                }
            )
        return dic

    else:
        raise ValueError(f"Unknown training data {dataname}")


def main(args):
    set_seed(42)
    logging.info("Loading data...")
    logging.info("Training and validation complete. Loss changes saved.")
    if args.apply_template:
        apply_template = True
    else:
        apply_template = False
    utils.run_lm_eval(args.pretrained, lora_path=None, apply_template = apply_template)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="random")
    # parser.add_argument("--pretrained", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--pretrained", type=str, default="meta-llama/Llama-3.1-8B")
    # parser.add_argument("--pretrained", type=str, default="EleutherAI/pythia-6.9b")
    # parser.add_argument("--pretrained", type=str, default="togethercomputer/RedPajama-INCITE-7B-Base")
    # parser.add_argument("--pretrained", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--total_samples", type=int, default=5 * 1000, help="Number of samples")
    parser.add_argument("--round_number", type=int, default=10, help="Number of rounds to train")
    parser.add_argument("--output_model_path", type=str, default=f"{DATA_PATH}/models")
    # Action apply_template = False
    parser.add_argument("--apply_template", action="store_true", help="Apply chat template")

    args = parser.parse_args()

    args.dataset_alias = f"{args.task}_{args.total_samples // 1000}k_{args.round_number}round"
    args.output_model_path = f"{args.output_model_path}/{args.dataset_alias}_{args.pretrained.replace('/', '-')}"
    main(args)
