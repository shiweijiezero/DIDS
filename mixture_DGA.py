import glob
import hashlib
import json
import logging
import os
import argparse
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path
from pprint import pprint

import datasets
import numpy as np
import pandas as pd
import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

import load_eval_data
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


def get_cache_path(source_name, cache_dir="./cache/embeddings"):
    """
    基于数据源名称生成cache路径
    source_name: 数据源名称，如 "tulu3", "science" 等
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return Path(cache_dir) / f"{source_name}_embeddings.pkl"


def get_text_embeddings(data_dict, model_name="all-MiniLM-L6-v2", cache_dir="./cache/embeddings"):
    """
    获取文本嵌入，基于数据源进行缓存
    """
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    # 提取所有文本
    texts = []
    text_to_source = {}  # 记录每个文本属于哪个源
    source_to_examples = {}  # 保存原始数据结构
    all_conversations = []

    for source, examples in data_dict.items():
        source_to_examples[source] = examples
        for ex in tqdm(examples, desc=f"Processing {source} data", total=len(examples)):
            prompt = tokenizer.apply_chat_template(
                ex["conversations"],
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(prompt)
            all_conversations.append(ex)
            text_to_source[prompt] = source

    # 检查每个源的缓存
    all_embeddings = []
    all_cached = True
    source_embeddings = {}

    for source in data_dict.keys():
        cache_path = get_cache_path(source)
        if not cache_path.exists():
            all_cached = False
            break
        with open(cache_path, 'rb') as f:
            source_embeddings[source] = pickle.load(f)

    # 如果所有源都有缓存，直接组合
    if all_cached:
        logging.info("Using cached embeddings for all sources")
        for text in texts:
            source = text_to_source[text]
            all_embeddings.append(source_embeddings[source][text])
        return np.array(all_embeddings), all_conversations

    # 否则重新计算所有embeddings
    logging.info("Computing embeddings for all texts...")
    embedder = SentenceTransformer(model_name)
    all_embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=64)

    # 按源分组并缓存
    for source in data_dict.keys():
        source_texts = [text for text in texts if text_to_source[text] == source]
        source_indices = [i for i, text in enumerate(texts) if text_to_source[text] == source]
        source_emb = {text: all_embeddings[i] for text, i in zip(source_texts, source_indices)}

        cache_path = get_cache_path(source)
        os.makedirs(cache_path.parent, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(source_emb, f)

    return all_embeddings, all_conversations


def hierarchical_clustering(data_dict, initial_clusters=64, cache_dir="./cache"):
    """
    层次聚类: 64 -> 4096，使用基于数据源的缓存

    Args:
        data_dict: 包含各个域数据的字典
        initial_clusters: 第一层聚类的簇数,默认64
        cache_dir: 缓存目录

    Returns:
    """
    # 寻找缓存的聚类结果
    cache_path = Path(cache_dir) / f"cluster_results_{initial_clusters}.pkl"
    if cache_path.exists():
        logging.info("Loading cached cluster results")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    # 获取embeddings
    embeddings, all_conversations = get_text_embeddings(data_dict, cache_dir=cache_dir)

    # 第一层聚类 (64个簇)
    start_time = time.time()
    logging.info("Performing first level clustering (k=64)")
    kmeans_l1 = KMeans(n_clusters=initial_clusters, random_state=42)
    clusters_l1 = kmeans_l1.fit_predict(embeddings)
    logging.info(f"First level clustering took {time.time() - start_time:.2f} seconds")

    # 收集每个cluster中的conversations
    l1_clusters = {i: [] for i in range(initial_clusters)}
    for i, cluster in enumerate(clusters_l1):
        l1_clusters[cluster].append(all_conversations[i])

    # 保存聚类结果
    with open(cache_path, 'wb') as f:
        pickle.dump(l1_clusters, f)

    return l1_clusters


def generate_training_data(train_data_dic, round_samples, dataset_alias, mixture_dic, round_number=0):
    all_data = []
    mixture_info = {
        "original_mixture": mixture_dic.copy(),
        "actual_samples": {},
        "supplementary_samples": {}
    }

    # 记录原始混合采样情况
    for source_name, data_lst in train_data_dic.items():
        random.shuffle(data_lst)
        ratio = mixture_dic[source_name]
        sample_num = int(round_samples * ratio)
        # 如果数据不足，全部采样
        if len(data_lst) < sample_num:
            logging.warning(f"Insufficient data for source {source_name}. Sampling all data.")
            if len(data_lst) == 0:
                logging.warning(f"Source {source_name} has no data left.")
                continue
            else:
                sampled_data = data_lst
        else:
            sampled_data = random.sample(data_lst, sample_num)
        all_data.extend(sampled_data)

        # 记录实际采样数量
        mixture_info["actual_samples"][source_name] = len(sampled_data)

        # 更新剩余数据
        if len(data_lst) <= sample_num:
            train_data_dic[source_name] = []
        else:
            train_data_dic[source_name] = data_lst[sample_num:]

    # 补充采样部分
    if len(all_data) < round_samples:
        logging.warning(f"Not enough data. Sample from other sources.")
        remaining_samples = round_samples - len(all_data)

        available_sources = {
            source: data_lst
            for source, data_lst in train_data_dic.items()
            if len(data_lst) > 0
        }

        if available_sources:
            sources_count = len(available_sources)
            samples_per_source = remaining_samples // sources_count
            extra_samples = remaining_samples % sources_count

            for i, (source, data_lst) in enumerate(available_sources.items()):
                current_samples = samples_per_source + (extra_samples if i == sources_count - 1 else 0)
                current_samples = min(current_samples, len(data_lst))

                sampled_data = random.sample(data_lst, current_samples)
                all_data.extend(sampled_data)

                # 记录补充采样数量
                mixture_info["supplementary_samples"][source] = len(sampled_data)

                if len(data_lst) == current_samples:
                    train_data_dic[source] = []
                else:
                    train_data_dic[source] = data_lst[current_samples:]
        else:
            logging.warning("No available sources for additional sampling")
            mixture_info["supplementary_samples"] = {}

    # 添加总体统计信息
    mixture_info["total_samples"] = len(all_data)
    mixture_info["target_samples"] = round_samples
    mixture_info["final_distribution"] = {
        source: (mixture_info["actual_samples"].get(source, 0) +
                 mixture_info["supplementary_samples"].get(source, 0))
        for source in set(mixture_dic.keys())
    }

    # 保存采样信息
    os.makedirs(f"{DATA_PATH}/mixture_info", exist_ok=True)
    # with open(f"{DATA_PATH}/mixture_info/{dataset_alias}_{round_number}.json", 'w', encoding='utf-8') as f:
    #     json.dump(mixture_info, f, indent=4, ensure_ascii=False)

    pprint(f"Round {round_number} mixture info: {mixture_info}")

    # 保存训练数据
    training_data = all_data
    random.shuffle(training_data)
    json_path = f"data/json/{dataset_alias}_{round_number}.json"
    # with open(json_path, 'w', encoding='utf-8') as f:
    #     json.dump(training_data, f, indent=4, ensure_ascii=False)

    # 保存JSONL文件
    with open(json_path, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    return training_data, f"json/{dataset_alias}_{round_number}.json", train_data_dic


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


def create_full_yaml_config(dataset_alias, model_path, output_path, round_number=0, lr=2e-6):
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
        "per_device_train_batch_size": 1,
        # 128/(GPU数量*4)
        "gradient_accumulation_steps": 128 // (torch.cuda.device_count() * 1),
        "learning_rate": lr,
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


def train_model(json_path, model_path, output_path, dataset_alias, round_number=0, lr=2e-6):
    update_dataset_info(json_path, dataset_alias)

    config_yaml_path = f"{DATA_PATH}/yaml/training_config_{dataset_alias}_{model_path.replace('/', '-')}.yaml"
    config = create_full_yaml_config(dataset_alias, model_path, output_path, round_number=round_number, lr=lr)
    with open(config_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)

    logging.info(
        f"Training model cmd: DISABLE_VERSION_CHECK=1 MKL_THREADING_LAYER=GNU  llamafactory-cli train {config_yaml_path}"
    )
    cmd = f"FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 llamafactory-cli train {config_yaml_path}"

    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end='')
        sys.stdout.flush()
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)


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
    dic = {}

    # 1. 加载训练数据
    # 如果文件已经存在，直接读取
    dataset_alias = args.dataset_alias

    logging.info("loading data from scratch")
    dic = load_specific(dataname="tulu3", dic=dic)

    train_data_dic = dic
    total_samples = args.total_samples
    total_samples = min(total_samples, sum([len(data) for data in train_data_dic.values()]))
    logging.info(f"Generating training data for {total_samples} samples...")

    # 2. 加载评估数据集
    eval_datasets = load_eval_data.load_eval_datasets()

    # 3. 对训练集执行聚类
    cluster_results = hierarchical_clustering(dic, initial_clusters=args.num_clusters)
    # 使用clustered_data替代原始的train_data_dic
    train_data_dic = cluster_results

    # 每一轮训练的数据
    round_samples = total_samples // args.round_number
    mixture_dic = {domain: 1.0 / len(train_data_dic) for domain in train_data_dic}
    for round_number in range(args.round_number):
        logging.info(f"Round {round_number + 1} of {args.round_number}")
        initial_lr = 2e-6
        min_lr = 0
        beta = 0.2
        decay = (args.round_number - round_number) / args.round_number
        lr = initial_lr * decay + min_lr * (1 - decay)

        if round_number == 0:
            # 初始化使用uniform采样
            mixture_dic = {key: 1 / len(train_data_dic) for key in train_data_dic.keys()}
            mixture_dic = utils.DGA_mixture(train_data_dic, eval_datasets, round_samples, args.num_clusters,
                                            round_number, args.pretrained, outer_lr=lr, beta=beta,
                                            task_name=args.task, alpha=mixture_dic)
            # random采样
            # all_num = sum([len(value) for value in train_data_dic.values()])
            # mixture_dic = {key: len(value) / all_num for key, value in train_data_dic.items()}
            training_data, data_path, train_data_dic = generate_training_data(train_data_dic, round_samples,
                                                                              args.dataset_alias, mixture_dic,
                                                                              round_number)
            model_path, output_path = args.pretrained, args.output_model_path
        else:
            # 从上一轮的剩余数据中使用DGA算法采样
            # def DGA_mixture(train_data_dic, eval_datasets, round_samples, num_clusters, round_number, model_path,
            #                 outer_lr=0.01, beta=0.1, task_name="mixture_DGA"):
            mixture_dic = utils.DGA_mixture(train_data_dic, eval_datasets, round_samples, args.num_clusters,
                                            round_number, args.output_model_path, outer_lr=lr, beta=beta,
                                            task_name=args.task, alpha=mixture_dic)

            training_data, data_path, train_data_dic = generate_training_data(train_data_dic, round_samples,
                                                                              args.dataset_alias, mixture_dic,
                                                                              round_number)
            model_path, output_path = args.output_model_path, args.output_model_path
        logging.info(f"Round {round_number} Training model...")

        train_model(data_path, model_path, output_path, dataset_alias, round_number, lr=lr)
    # train_model(train_json_path, args.pretrained, output_model_path, dataset_alias)

    logging.info(f"Eval model...")
    utils.run_lm_eval(args.output_model_path, lora_path=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="mixture_DGA_mean")
    # parser.add_argument("--pretrained", type=str, default="meta-llama/Llama-3.2-1B")
    # parser.add_argument("--pretrained", type=str, default="meta-llama/Llama-3.1-8B")
    # parser.add_argument("--pretrained", type=str, default="EleutherAI/pythia-6.9b")
    # parser.add_argument("--pretrained", type=str, default="togethercomputer/RedPajama-INCITE-7B-Base")
    parser.add_argument("--pretrained", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--total_samples", type=int, default=int(0.5 * 1000), help="Number of samples")
    parser.add_argument("--round_number", type=int, default=25, help="Number of rounds to train")
    parser.add_argument("--output_model_path", type=str, default=f"{DATA_PATH}/models")
    parser.add_argument("--num_clusters", type=int, default=64, choices=[64,128,256,512,1024,2048,4096])

    args = parser.parse_args()

    args.dataset_alias = f"{args.task}_{args.total_samples // 1000}k_{args.round_number}round_{args.num_clusters}clusters"
    args.output_model_path = f"{args.output_model_path}/{args.dataset_alias}_{args.pretrained.replace('/', '-')}"
    main(args)
