import glob
import hashlib
import json
import logging
import math
import multiprocessing
import os
import argparse
import pickle
import random
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Any

import datasets
import numpy as np
import pandas as pd
import torch
import yaml
from datasets import load_dataset
from scipy.optimize import curve_fit
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from sklearn.cluster import KMeans

import load_eval_data
import utils
from trak.projectors import BasicProjector, CudaProjector, ProjectionType

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


def get_trak_projector(device: torch.device):
    """ Get trak projectors (see https://github.com/MadryLab/trak for details) """
    try:
        num_sms = torch.cuda.get_device_properties(
            device.index).multi_processor_count
        import fast_jl

        # test run to catch at init time if projection goes through
        fast_jl.project_rademacher_8(torch.zeros(
            8, 1_000, device=device), 512, 0, num_sms)
        projector = CudaProjector
        print("Using CudaProjector")
    except:
        projector = BasicProjector
        print("Using BasicProjector")
    return projector


def get_text_embeddings(data_dict, model_name="Qwen/Qwen2.5-0.5B-Instruct", target_dim=512,
                        sample_dim=2000000, cache_dir="./cache/embeddings", gpu_id=0, param_ratio=0.1):
    """
    获取文本嵌入，只基于模型最后N%层参数的梯度进行计算，使用batch方式进行随机投影降维

    Args:
        data_dict: 包含各个域数据的字典
        model_name: 使用的模型名称
        target_dim: Random Projection的目标维度,默认512
        sample_dim: 梯度采样维度,默认2000000
        cache_dir: 缓存目录
        gpu_id: 指定使用的GPU ID，默认为0
        param_ratio: 使用最后多少比例的层，默认0.1表示最后10%的层

    Returns:
        tuple: (所有文本的embeddings, 所有对话)
    """
    # 检查缓存
    cache_path = Path(cache_dir) / f"embeddings_{target_dim}d_{sample_dim}s_{model_name.replace('/', '_')}_{param_ratio}_last.pkl"
    if cache_path.exists():
        logging.info(f"Loading cached embeddings from {cache_path}")
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
            return cached_data['embeddings'], cached_data['conversations']

    # 初始化模型和tokenizer
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    logging.info(f"Using device: {device}, model.device: {model.device}")

    # 获取最后N%的可训练参数
    all_params = list(model.parameters())
    num_params = len(all_params)
    start_idx = int(num_params * (1 - param_ratio))
    selected_params = all_params[start_idx:]

    total_params = sum(p.numel() for p in selected_params)
    logging.info(f"Using last {param_ratio*100}% layers, total parameters: {total_params}")

    # 初始化TRAK projector
    projector = get_trak_projector(model.device)
    proj = projector(
        grad_dim=total_params,
        proj_dim=target_dim,
        seed=42,
        proj_type=ProjectionType.rademacher,
        device=model.device,
        dtype=torch.float16,
        block_size=128,
        max_batch_size=128
    )

    # 收集所有对话
    all_conversations = []
    for source, examples in data_dict.items():
        all_conversations.extend(examples)

    # 获取采样维度信息
    feature_dim = total_params
    logging.info(f"Original feature dimension: {feature_dim}")
    logging.info(f"Sampled feature dimension: {sample_dim}")

    # 计算梯度特征
    gradient_features = []
    batch_size = 32  # 可以根据GPU显存调整

    for i in tqdm(range(0, len(all_conversations), batch_size), desc="Computing gradients"):
        batch_convs = all_conversations[i:i + batch_size]
        batch_grads = []

        for conv in batch_convs:
            # 准备输入
            messages = conv["conversations"]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=300)
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # 只计算选定参数的梯度
            with torch.enable_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                grads = torch.autograd.grad(loss, selected_params)

            # 处理梯度并展平
            grad_list = []
            for grad, param in zip(grads, selected_params):
                if grad is None:
                    grad = torch.zeros_like(param)
                grad_list.append(grad.view(-1).to(device))

            # 合并所有参数的梯度
            grad_feature = torch.cat(grad_list)
            grad_feature = grad_feature.view(1, -1).to(torch.float16)
            batch_grads.append(grad_feature)

        # 将batch的梯度堆叠在一起并进行随机投影降维
        batch_grads = torch.cat(batch_grads, dim=0)
        reduced_features = proj.project(batch_grads, model_id=42)
        gradient_features.extend(reduced_features.cpu().numpy())

    # 转换为numpy数组
    gradient_features = np.stack(gradient_features)

    # 缓存结果
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'embeddings': gradient_features,
            'conversations': all_conversations
        }, f)

    return gradient_features, all_conversations

def process_embeddings(args):
    """
    多进程处理函数，用于在特定GPU上获取embeddings
    """
    data_dict, gpu_id, model_name, target_dim, sample_dim, cache_dir = args
    return get_text_embeddings(
        data_dict,
        model_name=model_name,
        target_dim=target_dim,
        sample_dim=sample_dim,
        cache_dir=cache_dir,
        gpu_id=gpu_id
    )


def hierarchical_clustering(data_dict, initial_clusters=0, target_dim=1024, sample_dim=2000000,
                            cache_dir="./cache", model_name="Qwen/Qwen2.5-0.5B-Instruct",
                            num_gpus=None):
    """
    基于LLM最后的梯度的层次聚类，使用多进程并行处理

    Args:
        data_dict: 包含各个域数据的字典
        initial_clusters: 对原始簇后进行簇内再聚类的数量，如果为0则不进行簇内聚类，默认为0
        target_dim: Random Projection的目标维度,默认384
        sample_dim: 梯度采样维度,默认10000
        cache_dir: 缓存目录
        model_name: 使用的模型名称
        num_gpus: 使用的GPU数量，默认为None，表示使用所有可用GPU

    Returns:
        dict: 聚类结果字典
    """
    if initial_clusters <= 0:
        logging.info("Skipping hierarchical clustering")
        return data_dict

    # 检查缓存
    cache_path = Path(
        cache_dir) / f"DIDS_cluster_results_{initial_clusters}_{target_dim}d_{sample_dim}s_{model_name.replace('/', '_')}_mlp_JL.pkl"
    if cache_path.exists():
        logging.info("Loading cached cluster results")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    # 确定可用的GPU数量
    available_gpus = torch.cuda.device_count()
    if num_gpus is None:
        num_gpus = available_gpus
    num_gpus = min(num_gpus, available_gpus)

    if num_gpus == 0:
        raise RuntimeError("No GPU available for clustering")

    logging.info(f"Using {num_gpus} GPUs for parallel processing")

    # 先将所有数据合并，同时记录每个domain的数据范围
    all_data = []
    domain_ranges = {}
    start_idx = 0

    for domain, examples in data_dict.items():
        all_data.extend(examples)
        end_idx = start_idx + len(examples)
        domain_ranges[domain] = (start_idx, end_idx)
        start_idx = end_idx

    # 计算每个GPU处理的数据量
    total_samples = len(all_data)
    samples_per_gpu = total_samples // num_gpus

    # 均匀切分数据
    data_splits = []
    for i in range(num_gpus):
        start_idx = i * samples_per_gpu
        end_idx = min((i + 1) * samples_per_gpu, total_samples)
        split_data = all_data[start_idx:end_idx]

        # 创建一个新的字典，使用一个key来存储切分的数据
        split_dict = {"split_data": split_data}

        data_splits.append((
            split_dict,
            i,  # gpu_id
            model_name,
            target_dim,
            sample_dim,
            cache_dir
        ))

    logging.info(f"Data split sizes: {[len(split[0]['split_data']) for split in data_splits]}")

    # 使用多进程并行处理
    with Pool(num_gpus) as pool:
        results = pool.map(process_embeddings, data_splits)

    # 合并所有结果
    all_embeddings = []
    all_conversations = []
    for emb, conv in results:
        all_embeddings.append(emb)
        all_conversations.extend(conv)

    # 合并embeddings
    embeddings = np.concatenate(all_embeddings, axis=0)

    # 标准化特征
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # embeddings = scaler.fit_transform(embeddings)

    # 创建最终的聚类结果字典
    final_clusters = {}

    # 对每个domain分别进行聚类
    for domain, (start_idx, end_idx) in domain_ranges.items():
        logging.info(f"Processing domain: {domain}")

        # 获取当前domain的embeddings和对话
        domain_embeddings = embeddings[start_idx:end_idx]
        domain_conversations = all_conversations[start_idx:end_idx]

        # 执行domain内的聚类
        start_time = time.time()
        logging.info(f"Performing clustering for domain {domain} (k={initial_clusters})")
        kmeans = KMeans(n_clusters=initial_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(domain_embeddings)
        logging.info(f"Clustering for domain {domain} took {time.time() - start_time:.2f} seconds")

        # 将聚类结果添加到最终字典中
        for cluster_id in range(initial_clusters):
            cluster_key = f"{domain}-{cluster_id + 1}"
            cluster_examples = [domain_conversations[i] for i in range(len(clusters)) if clusters[i] == cluster_id]
            final_clusters[cluster_key] = cluster_examples

    # 缓存结果
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(final_clusters, f)

    return final_clusters

    # # 执行聚类
    # start_time = time.time()
    # logging.info(f"Performing clustering (k={initial_clusters})")
    # kmeans = KMeans(n_clusters=initial_clusters, random_state=42, n_init=10)
    # clusters = kmeans.fit_predict(embeddings)
    # logging.info(f"Clustering took {time.time() - start_time:.2f} seconds")

    # # 整理聚类结果
    # cluster_results = {i: [] for i in range(initial_clusters)}
    # for idx, cluster in enumerate(clusters):
    #     cluster_results[cluster].append(all_conversations[idx])

    # # 缓存结果
    # os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    # with open(cache_path, 'wb') as f:
    #     pickle.dump(cluster_results, f)

    # return cluster_results


def generate_training_data(train_data_dic, round_samples, dataset_alias, mixture_dic, round_number=0):
    all_data = []
    mixture_info = {
        "original_mixture": mixture_dic.copy(),
        "actual_samples": {},
        "supplementary_samples": {}
    }

    # 记录原始混合采样情况
    delete_keys = []
    for source_name, data_lst in train_data_dic.items():
        random.shuffle(data_lst)
        ratio = mixture_dic.get(source_name, 0)
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
            if sample_num == 0:
                logging.warning(f"Sample number is 0 for source {source_name}. Skipping.")
                sampled_data = []
            else:
                sampled_data = random.sample(data_lst, sample_num)
        all_data.extend(sampled_data)

        # 记录实际采样数量
        mixture_info["actual_samples"][source_name] = len(sampled_data)

        # 更新剩余数据
        if len(data_lst) <= sample_num:
            train_data_dic[source_name] = []
            # 把source_name这个键删除
            # del train_data_dic[source_name]
            # 记录要删除的键
            delete_keys.append(source_name)
        else:
            train_data_dic[source_name] = data_lst[sample_num:]

    # 删除已经采样完的数据
    for key in delete_keys:
        train_data_dic.pop(key)

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


def create_full_yaml_config(dataset_alias, model_path, output_path, round_number=0, lr=2e-6, min_lr=0):
    template = "llama3"

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
        "cutoff_len": 8192,
        "overwrite_cache": True,
        "preprocessing_num_workers": 64,
        "output_dir": output_path,
        "logging_steps": 1,
        "save_total_limit": 1,
        "save_strategy": "epoch",
        # "save_steps": 10000,
        "plot_loss": True,
        "overwrite_output_dir": True,
        "per_device_train_batch_size": 1,
        # 128/(GPU数量*4)
        "gradient_accumulation_steps": 32 // (torch.cuda.device_count() * 1),
        "learning_rate": lr,
        "num_train_epochs": 1.0,
        "lr_scheduler_type": "cosine_with_min_lr",
        "lr_scheduler_kwargs": {'min_lr': min_lr},
        # "warmup_ratio": 0,
        "bf16": True,
        "ddp_timeout": 180000000,
        # "report_to": "wandb",
        "report_to": "none",
        "run_name": dataset_alias + "_" + model_path.replace("/", "-"),
        "save_only_model": True,
    }
    return config


def train_model(json_path, model_path, output_path, dataset_alias, round_number=0, lr=2e-6, min_lr=0):
    update_dataset_info(json_path, dataset_alias)

    config_yaml_path = f"{DATA_PATH}/yaml/training_config_{dataset_alias}_{model_path.replace('/', '-')}.yaml"
    config = create_full_yaml_config(dataset_alias, model_path, output_path, round_number=round_number, lr=lr, min_lr=min_lr)
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
    elif dataname == "relevant_trains":
        dic = load_eval_data.load_relevant_trains()
        return dic
    else:
        raise ValueError(f"Unknown training data {dataname}")

# def compute_validation_loss(model, tokenizer, eval_data, device, batch_size=10):
#     """
#     批量计算验证集loss
#
#     Args:
#         model: 模型
#         tokenizer: 分词器
#         eval_data: 验证数据列表
#         device: 计算设备
#         batch_size: 批次大小
#
#     Returns:
#         float: 平均loss
#     """
#     start_time = time.time()
#     model.eval()
#     total_loss = 0
#     total_samples = 0
#
#     llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
#     llama_tokenizer.pad_token = llama_tokenizer.eos_token
#
#     # 创建数据加载器
#     def collate_fn(batch):
#         texts = [
#             llama_tokenizer.apply_chat_template(
#                 item,
#                 tokenize=False,
#                 add_generation_prompt=False,
#             )
#             for item in batch
#         ]
#         tokenizer.pad_token = tokenizer.eos_token
#         encodings = tokenizer(
#             texts,
#             padding=True,
#             truncation=True,
#             max_length=384,
#             return_tensors="pt"
#         )
#         return encodings
#
#     dataloader = DataLoader(
#         eval_data,
#         batch_size=batch_size,
#         collate_fn=collate_fn,
#         shuffle=False
#     )
#
#     with torch.no_grad():
#         for batch_inputs in dataloader:
#             # 将输入移到设备上
#             batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
#
#             # # 创建attention mask,标记非padding位置
#             # labels = batch_inputs["input_ids"].clone()
#             # # 将padding位置的label设为-100
#             # labels[batch_inputs["attention_mask"] == 0] = -100
#
#             # 计算loss
#             outputs = model(**batch_inputs, labels=batch_inputs["input_ids"].clone())
#
#             # 累加loss * batch大小，以便正确计算加权平均
#             batch_size = batch_inputs["input_ids"].size(0)
#             total_loss += outputs.loss.item() * batch_size
#             total_samples += batch_size
#
#     # 返回加权平均loss
#     logging.info(f"Validation loss computation took {time.time() - start_time:.2f} seconds")
#     return total_loss / total_samples

def compute_validation_loss(model, tokenizer, eval_data, device, batch_size=10):
    """
    批量计算验证集中assistant部分的loss

    Args:
        model: 模型
        tokenizer: 分词器
        eval_data: 验证数据列表
        device: 计算设备
        batch_size: 批次大小

    Returns:
        float: assistant部分的平均loss
    """
    start_time = time.time()
    model.eval()
    total_loss = 0
    total_samples = 0

    llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    llama_tokenizer.pad_token = llama_tokenizer.eos_token

    # 创建数据加载器
    def collate_fn(batch):
        # 对每个样本应用聊天模板并标记assistant部分
        processed_texts = []
        for item in batch:
            # 获取完整对话文本
            full_text = llama_tokenizer.apply_chat_template(
                item,
                tokenize=False,
                add_generation_prompt=False
            )
            # 记录assistant部分在文本中的位置
            assistant_spans = []
            current_pos = 0
            for message in item:
                if message['role'] == 'assistant':
                    # 在完整文本中定位assistant的回复
                    start = full_text.find(message['content'], current_pos)
                    if start != -1:
                        end = start + len(message['content'])
                        assistant_spans.append((start, end))
                        current_pos = end

            processed_texts.append({
                'text': full_text,
                'assistant_spans': assistant_spans
            })

        # 对文本进行编码
        tokenizer.pad_token = tokenizer.eos_token
        encodings = tokenizer(
            [item['text'] for item in processed_texts],
            padding=True,
            truncation=True,
            max_length=384,
            return_tensors="pt"
        )

        # 构建只包含assistant部分的labels
        labels = encodings['input_ids'].clone()
        labels[:] = -100  # 默认所有位置都不计算loss

        # 对每个样本，将assistant部分的token标记为计算loss
        for i, (encoding, item) in enumerate(zip(encodings['input_ids'], processed_texts)):
            # 获取每个assistant span对应的token范围
            for start_char, end_char in item['assistant_spans']:
                # 将字符位置转换为token位置
                token_spans = tokenizer(item['text'][start_char:end_char],
                                        add_special_tokens=False,
                                        return_tensors="pt")
                num_tokens = token_spans['input_ids'].size(1)

                # 在完整序列中定位这些token
                full_tokens = encoding.tolist()
                target_tokens = token_spans['input_ids'][0].tolist()

                # 查找匹配的token序列
                for pos in range(len(full_tokens) - num_tokens + 1):
                    if full_tokens[pos:pos + num_tokens] == target_tokens:
                        labels[i, pos:pos + num_tokens] = encoding[pos:pos + num_tokens]
                        break

        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        }

    dataloader = DataLoader(
        eval_data,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False
    )

    with torch.no_grad():
        for batch_inputs in dataloader:
            # 将输入移到设备上
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

            # 计算loss
            outputs = model(**batch_inputs)

            # 只统计非-100位置的loss
            valid_positions = (batch_inputs['labels'] != -100).sum().item()
            if valid_positions > 0:  # 确保有计算loss的位置
                total_loss += outputs.loss.item() * valid_positions
                total_samples += valid_positions

    # 返回加权平均loss
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    logging.info(f"Validation loss computation took {time.time() - start_time:.2f} seconds")
    logging.info(f"Processed {total_samples} valid positions from assistant responses")

    return avg_loss


def calculate_potential_factor(loss_history: List[float]):
    """
    计算domain的潜力因子，基于loss轨迹预测收敛值

    Args:
        loss_history: loss的历史记录
    """
    window = 5  # 固定窗口大小

    if len(loss_history) < window:
        return 1.0  # 历史数据不足时默认最大潜力

    current_loss = loss_history[-1]

    try:
        # 准备拟合数据
        y = np.array(loss_history)
        x = np.arange(len(y))

        # 指数衰减模型: y = a * exp(-b * x) + c
        def exp_decay(x, a, b, c):
            return a * np.exp(-b * x) + c

        # 拟合模型
        popt, pcov = curve_fit(
            exp_decay,
            x, y,
            p0=[y[0] - y[-1], 0.1, y[-1]],  # 初始参数估计
            bounds=(-np.inf, np.inf),  # 默认不设限制
            maxfev=2000,  # 增加最大迭代次数
        )

        # 预测future_window轮后的值
        predicted_loss = exp_decay(len(loss_history) + window, *popt)

        # 计算预计loss和当前loss的提升量
        future_improvement = current_loss - predicted_loss

    except Exception as e:
        logging.warning(f"Loss prediction failed: {e}")
        future_improvement = 0

    return future_improvement


def DIDS_mixture_multi_task(
        train_data_dic: Dict[str, List[Any]],
        eval_datasets: Dict[str, List[Any]],
        round_samples: int,
        num_clusters: int,
        round_number: int,
        model_path: str,
        outer_lr: float = 0.01,
        beta: float = 0.1,
        task_name: str = "DIDS",
        alpha: Dict[str, float] = None,
        k: int = 100,
        eval_losses: Dict[str, List[float]] = None
) -> Dict[str, float]:
    """
    改进的DIDS实现，基于KL散度、loss变化和未来潜力计算domain混合比例

    Args:
        train_data_dic: 训练数据字典
        eval_datasets: 评估数据集字典
        round_samples: 每轮训练的样本数
        num_clusters: 聚类数量
        round_number: 当前训练轮数
        model_path: 模型路径
        outer_lr: 外层学习率η
        beta: EMA参数β
        task_name: 任务名称
        alpha: 上一轮的混合比例字典
        k: 每个domain/task采样的样本数
        eval_losses: 评估数据集的loss历史记录
    """
    logging.info("Starting improved DIDS mixture calculation")

    # 加载模型和tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token

    # 初始化权重
    domains = list(train_data_dic.keys())
    if alpha is None:
        alpha = {domain: 1.0 / len(domains) for domain in domains}
    alpha_ema = alpha.copy()

    # 采样评估数据集和训练数据
    eval_domain_samples = {}
    for dataset_name, dataset in eval_datasets.items():
        if len(dataset) >= k:
            eval_domain_samples[dataset_name] = random.sample(dataset, k)
        else:
            eval_domain_samples[dataset_name] = random.choices(dataset, k=k)

    domain_samples = {}
    delete_domains = []
    for domain, data in train_data_dic.items():
        if len(data) > 0:
            if len(data) >= k:
                domain_samples[domain] = random.sample(data, k)
            else:
                domain_samples[domain] = random.choices(data, k=k)
        else:
            logging.warning(f"Empty data for domain {domain}")
            # 记录空数据的domain
            delete_domains.append(domain)
    for domain in delete_domains:
        train_data_dic.pop(domain)
        domains.remove(domain)

    # 计算KL散度
    kl_alignments = compute_DIDS_FIM(
        model,
        tokenizer,
        domain_samples,
        eval_domain_samples,
        device,
        k=k
    )
    # logging.info(f"KL divergences: {kl_alignments}")
    # kl_alignments: {domain: {eval_domain: kl_score}}

    # 处理KL散度值：对每个eval_domain分别进行标准化
    normalized_similarities = {}
    for eval_domain in next(iter(kl_alignments.values())).keys():
        # 收集所有domain对当前eval_domain的KL散度
        domain_kls = {domain: kls[eval_domain] for domain, kls in kl_alignments.items()}

        # 找到最大KL散度值
        max_kl = max(domain_kls.values())

        # 计算差值并归一化
        kl_diffs = {domain: max_kl - kl for domain, kl in domain_kls.items()}
        total_diff = sum(kl_diffs.values())

        # 进行归一化
        for domain in domain_kls:
            if domain not in normalized_similarities:
                normalized_similarities[domain] = {}
            if total_diff > 0:
                normalized_similarities[domain][eval_domain] = kl_diffs[domain] / total_diff
            else:
                normalized_similarities[domain][eval_domain] = 0
    logging.info(f"Normalized similarities : {normalized_similarities}")

    # 计算每个domain对每个下游任务loss变化的贡献
    domain_contributions = {domain: {eval_domain: 0.0 for eval_domain in eval_domain_samples.keys()}
                            for domain in domains}
    loss_changes = {}  # 存储loss变化情况
    potentials = {}  # 存储每个eval_domain的潜力因子

    # 计算每个domain对每个下游任务的效用值
    domain_utilities = {domain: {} for domain in domains}

    for eval_domain, loss_history in eval_losses.items():
        if len(loss_history) >= 2:
            # 计算loss improvement
            prev_loss = loss_history[-2]
            curr_loss = loss_history[-1]
            improvement = prev_loss - curr_loss

            # 计算potential
            future_improvement = calculate_potential_factor(loss_history)

            # 结合improvement和potential
            effective_improvement = improvement + future_improvement
            loss_changes[eval_domain] = effective_improvement
            potentials[eval_domain] = future_improvement

            # 计算每个domain的utility
            for domain in domains:
                # normalized_similarities[domain][eval_domain]就是Impact矩阵
                impact = normalized_similarities[domain][eval_domain]

                # 计算utility: Impact * Loss_improvement / p(domain)
                if alpha[domain] > 0:  # 避免除以0
                    utility = (impact * effective_improvement) / alpha[domain]
                else:
                    utility = 0

                domain_utilities[domain][eval_domain] = utility

    logging.info(f"potentials: {potentials}")
    logging.info(f"loss_changes: {loss_changes}")
    logging.info(f"domain_utilities: {domain_utilities}")

    # 基于utility计算新的混合比例
    new_weights = {}
    if len(eval_losses) > 0 and len(potentials) > 0:
        total_eval_domains = len(eval_domain_samples)

        for domain in domains:
            # 计算domain的平均utility
            domain_total_utility = sum(domain_utilities[domain].values()) / total_eval_domains

            # 转换为非负权重
            if domain_total_utility < 0:
                new_weights[domain] = 1e-7  # 很小的正数
            else:
                new_weights[domain] = domain_total_utility

            logging.info(f"Total utility for {domain}: {domain_total_utility:.4f}")
    else:
        # 如果没有足够的历史数据，使用impact作为权重
        for domain in domains:
            new_weights[domain] = sum(normalized_similarities[domain].values())

    # 归一化权重
    total_weight = sum(new_weights.values())
    if total_weight > 0:
        alpha_new = {domain: w / total_weight for domain, w in new_weights.items()}
    else:
        # 如果所有权重都是0，使用均匀分布
        alpha_new = {domain: 1.0 / len(domains) for domain in domains}

    # 应用EMA更新
    logging.info(f"previous weights: {alpha}")
    alpha_ema = {
        domain: beta * alpha_ema[domain] + (1 - beta) * alpha_new[domain]
        for domain in domains
    }
    logging.info(f"updated EMA weights: {alpha_ema}")

    # 保存详细信息用于分析
    save_info = {
        "round": round_number,
        "kl_alignments": kl_alignments,
        "normalized_similarities": normalized_similarities,
        "domain_contributions": domain_contributions,
        "losses": eval_losses,
        "loss_changes": loss_changes,
        "potentials": potentials if eval_losses else {},
        "weights": new_weights,
        "final_weights": alpha_new,
        "ema_weights": alpha_ema,
    }

    os.makedirs(f"./data/mixture_ratio/{task_name}", exist_ok=True)
    with open(f"./data/mixture_ratio/{task_name}/round_{round_number}.json", "w") as f:
        json.dump(save_info, f, indent=4)

    # 释放GPU内存
    del model
    torch.cuda.empty_cache()

    return alpha_ema


def compute_fisher_matrix_diagonal(model, tokenizer, domain_samples, eval_domain_samples, device,
                                   param_sample_ratio=0.1, batch_size=10):
    """
    Compute diagonal Fisher Information Matrix using empirical Fisher with proper batching.
    只计算最后N%层的参数的对角线元素来降低内存使用
    F ≈ diag(E[∇log p(x|θ)⊙∇log p(x|θ)])

    Args:
        param_sample_ratio: 使用最后多少比例的层，默认0.1表示最后10%的层
    """
    logging.info("Computing diagonal empirical Fisher Information Matrix")

    # Get all trainable parameters
    trainable_params = [param for param in model.parameters() if param.requires_grad]

    # Select last N% layers
    total_layers = len(trainable_params)
    num_selected_layers = max(1, int(total_layers * param_sample_ratio))
    selected_params = trainable_params[-num_selected_layers:]

    logging.info(f"Using last {param_sample_ratio * 100:.1f}% layers ({num_selected_layers} layers)")
    # 打印参数总数，以MB为单位
    logging.info(f"Total trainable parameters: {sum(param.numel() for param in trainable_params) / 1e6:.2f} MB")
    # 打印选择的参数总数，以MB为单位
    logging.info(f"Selected parameters: {sum(param.numel() for param in selected_params) / 1e6:.2f} MB")

    # Initialize fisher diagonal for selected parameters on CPU
    fisher_diags = [torch.zeros_like(param.flatten(), device='cpu') for param in selected_params]
    sample_count = 0

    # Collect all samples from domain_samples
    all_samples = []
    for domain, batches in domain_samples.items():
        for batch in batches:
            input_text = tokenizer.apply_chat_template(
                batch["conversations"],
                tokenize=False,
                add_generation_prompt=False,
            )
            all_samples.append((domain, input_text))

    if len(all_samples) > 15:
        all_samples = random.sample(all_samples, 15)

    logging.info(f"Processing {len(all_samples)} domain samples")

    # Process samples in batches
    for i in range(0, len(all_samples), batch_size):
        batch_samples = all_samples[i:i + batch_size]
        batch_inputs = [sample[1] for sample in batch_samples]
        batch_domains = [sample[0] for sample in batch_samples]

        model_input = tokenizer(
            batch_inputs,
            padding=True,
            max_length=384,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        # Compute gradients on CUDA
        with torch.enable_grad():
            outputs = model(**model_input, labels=model_input["input_ids"])
            loss = outputs.loss
            for domain in batch_domains:
                logging.info(f"Loss for domain {domain}: {loss.item():.4f}")

            gradient = torch.autograd.grad(loss, selected_params)

            # Move to CPU and accumulate squared gradients
            for grad_idx, grad in enumerate(gradient):
                fisher_diags[grad_idx] += grad.detach().cpu().flatten().pow(2)

            sample_count += len(batch_samples)

            del outputs
            del gradient
            torch.cuda.empty_cache()

    # Process eval samples
    all_eval_samples = []
    for eval_domain, eval_batches in eval_domain_samples.items():
        for eval_batch in eval_batches:
            eval_input = tokenizer.apply_chat_template(
                eval_batch,
                tokenize=False,
                add_generation_prompt=False,
            )
            all_eval_samples.append((eval_domain, eval_input))

    if len(all_eval_samples) > 15:
        all_eval_samples = random.sample(all_eval_samples, 15)

    logging.info(f"Processing {len(all_eval_samples)} eval domain samples")

    for i in range(0, len(all_eval_samples), batch_size):
        batch_eval_samples = all_eval_samples[i:i + batch_size]
        batch_eval_inputs = [sample[1] for sample in batch_eval_samples]
        batch_eval_domains = [sample[0] for sample in batch_eval_samples]

        eval_input = tokenizer(
            batch_eval_inputs,
            padding=True,
            max_length=384,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.enable_grad():
            outputs = model(**eval_input, labels=eval_input["input_ids"])
            loss = outputs.loss
            for eval_domain in batch_eval_domains:
                logging.info(f"Loss for eval domain {eval_domain}: {loss.item():.4f}")

            gradient = torch.autograd.grad(loss, selected_params)

            for grad_idx, grad in enumerate(gradient):
                fisher_diags[grad_idx] += grad.detach().cpu().flatten().pow(2)

            sample_count += len(batch_eval_samples)

            del outputs
            del gradient
            torch.cuda.empty_cache()

    # Average the accumulated squares and add small term for numerical stability
    fisher_diags = [diag / sample_count + 1e-10 for diag in fisher_diags]

    return fisher_diags, selected_params


def compute_DIDS_FIM(model, tokenizer, domain_samples, eval_domain_samples, device='cuda', k=5, param_sample_ratio=0.1,
                     batch_size=20):
    """
    使用对角Fisher矩阵计算KL散度，支持批处理，只使用最后N%的层
    KL[p(x|θ+∇ℓ_Di) | p(x|θ+∇ℓ_Sj)] ≈ 1/2(∇ℓ_Sj-∇ℓ_Di)^T diag(F) ⊙ (∇ℓ_Sj-∇ℓ_Di)

    Args:
        param_sample_ratio: 使用最后多少比例的层，默认0.1表示最后10%的层
    """
    model.eval()
    kl_divergences = {}

    logging.info("Starting optimized KL divergence computation")

    # Compute diagonal Fisher Information Matrix and get selected parameters
    logging.info("Computing diagonal Fisher Information Matrix")
    fisher_diags, selected_params = compute_fisher_matrix_diagonal(
        model,
        tokenizer,
        domain_samples,
        eval_domain_samples,
        device,
        param_sample_ratio=param_sample_ratio,
        batch_size=batch_size
    )

    # Dictionary to store gradients for each eval domain
    eval_gradients = {}

    # Calculate gradients for each eval domain
    logging.info(f"Processing {len(eval_domain_samples)} evaluation domains")
    for eval_domain, eval_batches in tqdm(eval_domain_samples.items(), desc="Computing eval gradients"):
        all_eval_inputs = []
        for eval_batch in eval_batches:
            eval_input = tokenizer.apply_chat_template(
                eval_batch,
                tokenize=False,
                add_generation_prompt=False,
            )
            all_eval_inputs.append(eval_input)

        domain_gradients = [torch.zeros_like(fisher_diag, device='cpu') for fisher_diag in fisher_diags]
        batch_count = 0

        for i in range(0, len(all_eval_inputs), batch_size):
            batch_eval_inputs = all_eval_inputs[i:i + batch_size]

            eval_input = tokenizer(
                batch_eval_inputs,
                padding=True,
                max_length=384,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            with torch.enable_grad():
                eval_outputs = model(**eval_input, labels=eval_input["input_ids"])
                eval_loss = eval_outputs.loss

                gradient = torch.autograd.grad(eval_loss, selected_params)

                for grad_idx, grad in enumerate(gradient):
                    domain_gradients[grad_idx] += grad.detach().cpu().flatten()

                batch_count += 1

                del eval_outputs
                del gradient
                torch.cuda.empty_cache()

        logging.info(f"Processing evaluation domain: {eval_domain}, eval_loss: {eval_loss.item():.4f}")

        # Average gradients across batches
        domain_gradients = [grad / batch_count for grad in domain_gradients]
        eval_gradients[eval_domain] = domain_gradients

    # Process each training domain
    logging.info(f"Processing {len(domain_samples)} training domains")
    for domain, batches in tqdm(domain_samples.items(), desc="Computing KL divergences"):
        logging.info(f"Computing KL divergence for domain: {domain}")

        all_domain_inputs = []
        for batch in batches:
            domain_input = tokenizer.apply_chat_template(
                batch["conversations"],
                tokenize=False,
                add_generation_prompt=False,
            )
            all_domain_inputs.append(domain_input)

        domain_gradients = [torch.zeros_like(fisher_diag, device='cpu') for fisher_diag in fisher_diags]
        batch_count = 0

        for i in range(0, len(all_domain_inputs), batch_size):
            batch_domain_inputs = all_domain_inputs[i:i + batch_size]

            domain_input = tokenizer(
                batch_domain_inputs,
                padding=True,
                max_length=384,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            with torch.enable_grad():
                domain_outputs = model(**domain_input, labels=domain_input["input_ids"])
                domain_loss = domain_outputs.loss

                gradient = torch.autograd.grad(domain_loss, selected_params)

                for grad_idx, grad in enumerate(gradient):
                    domain_gradients[grad_idx] += grad.detach().cpu().flatten()

                batch_count += 1

                del domain_outputs
                del gradient
                torch.cuda.empty_cache()

        # Average gradients across batches
        domain_gradients = [grad / batch_count for grad in domain_gradients]

        # Calculate KL divergence with each eval domain using diagonal Fisher
        domain_kls = {}
        for eval_domain, eval_gradient in eval_gradients.items():
            kl_div = 0
            for fisher_diag, domain_grad, eval_grad in zip(fisher_diags, domain_gradients, eval_gradient):
                # grad_diff = eval_grad - domain_grad
                # kl_div += 0.5 * torch.sum(fisher_diag * grad_diff * grad_diff)

                # Fisher Kernel
                kl_div += 0.5 * torch.sum(fisher_diag * domain_grad * eval_grad)

            domain_kls[eval_domain] = float(kl_div)
            logging.info(f"KL divergence for {domain} with {eval_domain}: {domain_kls[eval_domain]:.4f}")

        kl_divergences[domain] = domain_kls
        logging.info(f"Average KL divergence for {domain}: {kl_divergences[domain]}")

    return kl_divergences


def main(args):
    set_seed(42)
    logging.info("Loading data...")

    # 1. 加载训练数据
    dataset_alias = args.dataset_alias
    dic = load_specific(dataname="tulu3", dic={})
    dic2 = load_specific(dataname="relevant_trains", dic={})
    dic.update(dic2)
    train_data_dic = dic

    # 打印数据集信息
    for domain, data in train_data_dic.items():
        logging.info(f"Loaded {len(data)} samples for domain {domain}, ratio in total: {len(data) / sum([len(data) for data in train_data_dic.values()]):.2f}")

    total_samples = min(args.total_samples,
                        sum([len(data) for data in train_data_dic.values()]))
    logging.info(f"Generating training data for {total_samples} samples...")

    # 2. 加载评估数据集
    eval_datasets = load_eval_data.load_eval_datasets()
    # 固定采样用于验证集loss计算
    fixed_eval_samples = {}
    eval_sample_size = 200
    for dataset_name, dataset in eval_datasets.items():
        if len(dataset) >= eval_sample_size:
            fixed_eval_samples[dataset_name] = random.sample(dataset, eval_sample_size)
        else:
            fixed_eval_samples[dataset_name] = random.choices(dataset, k=eval_sample_size)

    # 记录每个评估集的loss
    eval_losses = {dataset_name: [] for dataset_name in eval_datasets.keys()}

    # 3. 对训练集执行层次化聚类
    cluster_results = hierarchical_clustering(dic, initial_clusters=args.num_clusters)
    # # 使用clustered_data替代原始的train_data_dic
    train_data_dic = cluster_results

    # 计算初始验证集loss
    initial_model = AutoModelForCausalLM.from_pretrained(
        args.pretrained,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    initial_tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
    initial_tokenizer.pad_token = initial_tokenizer.eos_token

    for dataset_name, eval_data in fixed_eval_samples.items():
        initial_loss = compute_validation_loss(
            initial_model,
            initial_tokenizer,
            eval_data,
            "cuda",
            batch_size=20
        )
        eval_losses[dataset_name].append(initial_loss)
        logging.info(f"Initial loss for {dataset_name}: {initial_loss:.4f}")

    del initial_model
    torch.cuda.empty_cache()
    # --------------------------------------------------------------------------------

    # 每一轮训练的数据
    round_samples = total_samples // args.round_number
    mixture_dic = {domain: 1.0 / len(train_data_dic) for domain in train_data_dic}
    for round_number in range(args.round_number):
        logging.info(f"Round {round_number + 1} of {args.round_number}")
        # 设置模型路径
        model_path = args.pretrained if round_number == 0 else args.output_model_path
        output_path = args.output_model_path
        # 学习率衰减
        initial_lr = 2e-6
        min_lr = 0
        decay = (args.round_number - round_number) / args.round_number
        lr = initial_lr * decay + min_lr * (1 - decay)
        cur_min_lr = initial_lr *  (args.round_number - round_number -1) / args.round_number
        cur_min_lr = max(0, cur_min_lr)

        if round_number == 0:
            # 初始化使用uniform采样
            mixture_dic = {key: 1 / len(train_data_dic) for key in train_data_dic.keys()}
            mixture_dic = DIDS_mixture_multi_task(
                train_data_dic,
                eval_datasets,
                round_samples,
                args.num_clusters,
                round_number,
                model_path,
                outer_lr=lr,
                beta=0.2,
                task_name=args.task,
                alpha=mixture_dic,
                eval_losses=eval_losses
            )
        else:
            # 使用改进的DIDS算法计算新的mixture比例
            mixture_dic = DIDS_mixture_multi_task(
                train_data_dic,
                eval_datasets,
                round_samples,
                args.num_clusters,
                round_number,
                model_path,
                outer_lr=lr,
                beta=0.2,
                task_name=args.task,
                alpha=mixture_dic,
                eval_losses=eval_losses
            )

        # 生成训练数据
        training_data, data_path, train_data_dic = generate_training_data(
            train_data_dic,
            round_samples,
            args.dataset_alias,
            mixture_dic,
            round_number
        )

        # 训练模型
        logging.info(f"Round {round_number} Training model...")
        train_model(data_path, model_path, output_path, dataset_alias, round_number, lr=lr, min_lr=cur_min_lr)

        # 计算新的验证集loss
        logging.info("Computing validation losses after training...")
        current_model = AutoModelForCausalLM.from_pretrained(
            args.output_model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )

        for dataset_name, eval_data in fixed_eval_samples.items():
            current_loss = compute_validation_loss(
                current_model,
                initial_tokenizer,
                eval_data,
                "cuda",
                batch_size=20
            )
            eval_losses[dataset_name].append(current_loss)
            logging.info(f"Current loss for {dataset_name}: {current_loss:.4f}")

        # 保存loss历史
        os.makedirs(f"./data/eval_losses/{args.task}", exist_ok=True)
        with open(f"./data/eval_losses/{args.task}/round_{round_number}.json", "w", encoding="utf-8") as f:
            json.dump(eval_losses, f, indent=4, ensure_ascii=False)

        del current_model
        torch.cuda.empty_cache()

    logging.info(f"Eval model...")
    utils.run_lm_eval(args.output_model_path, lora_path=None)


if __name__ == "__main__":
    # 指定spawn
    multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="DIDS")
    # parser.add_argument("--pretrained", type=str, default="meta-llama/Llama-3.2-1B")
    # parser.add_argument("--pretrained", type=str, default="meta-llama/Llama-3.1-8B")
    # parser.add_argument("--pretrained", type=str, default="EleutherAI/pythia-6.9b")
    # parser.add_argument("--pretrained", type=str, default="togethercomputer/RedPajama-INCITE-7B-Base")
    # parser.add_argument("--pretrained", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--pretrained", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--total_samples", type=int, default=int(0.5 * 1000), help="Number of samples")
    parser.add_argument("--round_number", type=int, default=25, help="Number of rounds to train")
    parser.add_argument("--output_model_path", type=str, default=f"{DATA_PATH}/models")
    parser.add_argument("--num_clusters", type=int, default=0)

    args = parser.parse_args()

    args.dataset_alias = f"{args.task}_{args.total_samples // 1000}k_{args.round_number}round_{args.num_clusters}clusters"
    args.output_model_path = f"{args.output_model_path}/{args.dataset_alias}_{args.pretrained.replace('/', '-')}"
    main(args)
