import json
import logging
import math
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
import random
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
from scipy.optimize import curve_fit
from tqdm import tqdm
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_lm_eval(model_path, output_dir="./results", lora_path=None, apply_template=True):
    """
    Run lm-eval-harness evaluation using command line

    Args:
        model_path (str): Path to the model checkpoint
        output_dir (str): Directory to save results
    """
    # Create timestamp for unique output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"{model_path.replace('/', '-').replace('.', '')}-{lora_path}")

    # Ensure output directory exists
    os.makedirs(output_file, exist_ok=True)

    # Construct command
    # export HF_DATASETS_TRUST_REMOTE_CODE=1
    os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
    if lora_path is None:
        model_args_str = f"pretrained={model_path},trust_remote_code=True"
    else:
        model_args_str = f"pretrained={model_path},trust_remote_code=True,peft={lora_path}"

    vllm_args = f"pretrained={model_path},tensor_parallel_size={torch.cuda.device_count()},dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1"

    cmd = f"python -m lm_eval --model hf --model_args {model_args_str} --tasks bbh,gsm8k,piqa,pubmedqa,boolq,ifeval,mmlu,truthfulqa,minerva_math --batch_size auto --output_path {output_file} --device cuda --trust_remote_code --log_samples "

    vllm_cmd = f"lm_eval --model vllm --model_args {vllm_args} --tasks bbh,gsm8k,piqa,pubmedqa,boolq,ifeval,mmlu,truthfulqa,minerva_math --batch_size auto --output_path {output_file} --trust_remote_code --log_samples "

    if apply_template:
        cmd += " --apply_chat_template  "
        vllm_cmd += " --apply_chat_template  "
    number_fewshot = 3
    if number_fewshot is not None:
        cmd += f" --num_fewshot {number_fewshot} "
        vllm_cmd += f" --num_fewshot {number_fewshot} "

    if lora_path is not None:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   bufsize=1)
    else:
        process = subprocess.Popen(vllm_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   bufsize=1)
    for line in process.stdout:
        print(line, end='')
        sys.stdout.flush()
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)

def run_lm_eval_single_domain(model_path, output_dir="./results", lora_path=None, apply_template=True, single_domain_dataset_name=None):
    """
    Run lm-eval-harness evaluation using command line

    Args:
        model_path (str): Path to the model checkpoint
        output_dir (str): Directory to save results
    """
    # Create timestamp for unique output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"{model_path.replace('/', '-').replace('.', '')}-{lora_path}")

    # Ensure output directory exists
    os.makedirs(output_file, exist_ok=True)

    # Construct command
    # export HF_DATASETS_TRUST_REMOTE_CODE=1
    os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
    if lora_path is None:
        model_args_str = f"pretrained={model_path},trust_remote_code=True"
    else:
        model_args_str = f"pretrained={model_path},trust_remote_code=True,peft={lora_path}"

    vllm_args = f"pretrained={model_path},tensor_parallel_size={torch.cuda.device_count()},dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1"

    # ["mmlu", "truthfulqa", "bbh", "gsm8k", "mathqa","arc_easy","boolq","ifeval","logiqa","minerva_math","piqa","pubmedqa"]
    if single_domain_dataset_name == "mmlu":
        task_name = "mmlu"
    elif single_domain_dataset_name == "truthfulqa":
        task_name = "truthfulqa"
    elif single_domain_dataset_name == "bbh":
        task_name = "bbh"
    elif single_domain_dataset_name == "gsm8k":
        task_name = "gsm8k"
    elif single_domain_dataset_name == "mathqa":
        task_name = "mathqa"
    elif single_domain_dataset_name == "arc_easy":
        task_name = "arc_easy"
    elif single_domain_dataset_name == "boolq":
        task_name = "boolq"
    elif single_domain_dataset_name == "ifeval":
        task_name = "ifeval"
    elif single_domain_dataset_name == "logiqa":
        task_name = "logiqa"
    elif single_domain_dataset_name == "minerva_math":
        task_name = "minerva_math"
    elif single_domain_dataset_name == "piqa":
        task_name = "piqa"
    elif single_domain_dataset_name == "pubmedqa":
        task_name = "pubmedqa"
    else:
        raise ValueError("single_domain_dataset_name not found")

    cmd = f"python -m lm_eval --model hf --model_args {model_args_str} --tasks {task_name} --batch_size auto --output_path {output_file} --device cuda --trust_remote_code --log_samples "

    vllm_cmd = f"lm_eval --model vllm --model_args {vllm_args} --tasks {task_name} --batch_size auto --output_path {output_file} --trust_remote_code --log_samples "

    if apply_template:
        cmd += " --apply_chat_template  "
        vllm_cmd += " --apply_chat_template  "
    number_fewshot = 3
    if number_fewshot is not None:
        cmd += f" --num_fewshot {number_fewshot} "
        vllm_cmd += f" --num_fewshot {number_fewshot} "

    if lora_path is not None:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   bufsize=1)
    else:
        process = subprocess.Popen(vllm_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   bufsize=1)
    for line in process.stdout:
        print(line, end='')
        sys.stdout.flush()
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)

def compute_gradient_alignments(model, tokenizer, domain_batch, eval_batch, device='cuda'):
    """
    Compute gradient alignments between domain batches and specific batch,
    using all parameters of the model
    Args:
        domain_batch: Dictionary mapping domain name to batch data
        eval_batch: Batch from specific dataset
    Returns:
        alignments: Dictionary mapping domain name to alignment score
    """
    model.eval()
    alignments = {}

    logging.info("Starting gradient alignment computation")

    # Get all trainable parameters
    model_params = [param for param in model.parameters() if param.requires_grad]

    if not model_params:
        raise ValueError("No trainable parameters found in model")

    # Get specific gradient
    logging.info("Processing evaluation batch")
    eval_input = tokenizer.apply_chat_template(
        eval_batch,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt"
    ).to(device)

    with torch.enable_grad():
        eval_outputs = model(eval_input, labels=eval_input)
        eval_loss = eval_outputs.loss
        logging.info(f"Evaluation loss: {eval_loss.item():.4f}")

    eval_gradient = torch.autograd.grad(eval_loss, model_params)
    sequence_length = eval_input.size(1)
    eval_gradient = [g.detach().cpu() / sequence_length for g in eval_gradient]
    del eval_outputs
    torch.cuda.empty_cache()

    logging.info(f"Processing {len(domain_batch)} domains")
    # Compute alignment with each domain
    for domain, batch in tqdm(domain_batch.items(), desc="Computing alignment", total=len(domain_batch)):
        # Limit the length of conversations
        for i in range(len(batch["conversations"])):
            batch["conversations"][i]["content"] = batch["conversations"][i]["content"][:1000]

        logging.info(f"Computing alignment for domain: {domain}")
        domain_input = tokenizer.apply_chat_template(
            batch["conversations"],
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt"
        ).to(device)

        with torch.enable_grad():
            domain_outputs = model(domain_input, labels=domain_input)
            domain_loss = domain_outputs.loss
            logging.info(f"Domain loss for {domain}: {domain_loss.item():.4f}")

        domain_gradient = torch.autograd.grad(domain_loss, model_params)
        sequence_length = domain_input.size(1)
        domain_gradient = [g.detach().cpu() / sequence_length for g in domain_gradient]
        del domain_outputs
        torch.cuda.empty_cache()

        # Compute similarity on CPU
        alignment = sum(F.cosine_similarity(dg.flatten(), sg.flatten(), dim=0)
                        for dg, sg in zip(domain_gradient, eval_gradient))
        alignments[domain] = float(alignment)
        logging.info(f"Alignment score for {domain}: {alignments[domain]:.4f}")

    return alignments


def compute_gradient_alignments_multi_task(model, tokenizer, domain_batch, eval_domain_dataset, device='cuda'):
    """
    Compute gradient alignments between domain batches and multiple evaluation datasets,
    using all layers of the LLaMA model.

    Args:
        model: The LLaMA model
        tokenizer: The tokenizer
        domain_batch: Dictionary mapping domain name to batch data
        eval_domain_dataset: Dictionary mapping eval domain name to batch data
        device: Computing device

    Returns:
        alignments: Dictionary mapping domain name to average alignment score
    """
    model.eval()
    alignments = {}

    logging.info("Starting gradient alignment computation for multiple tasks")

    # Get all trainable parameters
    trainable_params = []
    param_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            param_names.append(name)

    if not trainable_params:
        raise ValueError("No trainable parameters found in model")

    logging.info(f"Found {len(trainable_params)} trainable parameters")

    # Dictionary to store gradients for each eval domain
    eval_gradients = {}

    # Calculate gradients for each eval domain
    logging.info(f"Processing {len(eval_domain_dataset)} evaluation domains")
    for eval_domain, eval_batch in eval_domain_dataset.items():
        logging.info(f"Processing evaluation domain: {eval_domain}")

        eval_input = tokenizer.apply_chat_template(
            eval_batch,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt"
        ).to(device)

        with torch.enable_grad():
            eval_outputs = model(eval_input, labels=eval_input)
            eval_loss = eval_outputs.loss
            logging.info(f"Evaluation loss for {eval_domain}: {eval_loss.item():.4f}")

        gradient = torch.autograd.grad(eval_loss, trainable_params)
        sequence_length = eval_input.size(1)
        eval_gradients[eval_domain] = [g.detach().cpu() / sequence_length for g in gradient]

        del eval_outputs
        torch.cuda.empty_cache()

    # Process each training domain
    logging.info(f"Processing {len(domain_batch)} training domains")
    for domain, batch in tqdm(domain_batch.items(), desc="Computing alignments", total=len(domain_batch)):
        # Limit the length of conversations
        for i in range(len(batch["conversations"])):
            batch["conversations"][i]["content"] = batch["conversations"][i]["content"][:1000]

        logging.info(f"Computing alignment for domain: {domain}")
        domain_input = tokenizer.apply_chat_template(
            batch["conversations"],
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt"
        ).to(device)

        with torch.enable_grad():
            domain_outputs = model(domain_input, labels=domain_input)
            domain_loss = domain_outputs.loss
            logging.info(f"Domain loss for {domain}: {domain_loss.item():.4f}")

        domain_gradient = torch.autograd.grad(domain_loss, trainable_params)
        sequence_length = domain_input.size(1)
        domain_gradient = [g.detach().cpu() / sequence_length for g in domain_gradient]

        del domain_outputs
        torch.cuda.empty_cache()

        # Calculate alignment with each eval domain
        domain_alignments = {}
        # Dictionary to store layer-wise alignments
        layer_alignments = defaultdict(dict)

        for eval_domain, eval_gradient in eval_gradients.items():
            # Calculate overall alignment
            alignment = sum(F.cosine_similarity(dg.flatten(), eg.flatten(), dim=0)
                            for dg, eg in zip(domain_gradient, eval_gradient))
            domain_alignments[eval_domain] = float(alignment)

            # Calculate layer-wise alignments
            for param_name, dg, eg in zip(param_names, domain_gradient, eval_gradient):
                layer_name = param_name.split('.')[1] if 'layers.' in param_name else 'other'
                layer_alignment = float(F.cosine_similarity(dg.flatten(), eg.flatten(), dim=0))
                if layer_name not in layer_alignments[eval_domain]:
                    layer_alignments[eval_domain][layer_name] = []
                layer_alignments[eval_domain][layer_name].append(layer_alignment)

            # Log overall alignment
            logging.info(
                f"Overall alignment score for {domain} with {eval_domain}: {domain_alignments[eval_domain]:.4f}")

            # Log layer-wise alignments
            for layer_name, alignments_list in layer_alignments[eval_domain].items():
                avg_layer_alignment = sum(alignments_list) / len(alignments_list)
                logging.info(f"Layer {layer_name} alignment for {domain} with {eval_domain}: {avg_layer_alignment:.4f}")

        # Calculate average alignment across all eval domains
        alignments[domain] = {
            'overall': sum(domain_alignments.values()) / len(domain_alignments),
            'layer_wise': {}
        }

        # Calculate average layer-wise alignments across all eval domains
        for eval_domain in eval_domain_dataset.keys():
            for layer_name in layer_alignments[eval_domain]:
                if layer_name not in alignments[domain]['layer_wise']:
                    alignments[domain]['layer_wise'][layer_name] = []
                alignments[domain]['layer_wise'][layer_name].extend(layer_alignments[eval_domain][layer_name])

        # Average the layer-wise alignments
        for layer_name in alignments[domain]['layer_wise']:
            alignments[domain]['layer_wise'][layer_name] = (
                    sum(alignments[domain]['layer_wise'][layer_name]) /
                    len(alignments[domain]['layer_wise'][layer_name])
            )

        logging.info(f"Average overall alignment score for {domain}: {alignments[domain]['overall']:.4f}")
        for layer_name, avg_alignment in alignments[domain]['layer_wise'].items():
            logging.info(f"Average layer {layer_name} alignment for {domain}: {avg_alignment:.4f}")

    return alignments


def DGA_mixture_multi_task(train_data_dic, eval_datasets, round_samples, num_clusters, round_number, model_path,
                           outer_lr=0.01, beta=0.1, task_name="mixture_DGA", alpha=None):
    """
    Dynamic Gradient Alignment with EMA

    Args:
        train_data_dic: Dictionary mapping domain name to training data
        eval_datasets: Dictionary of evaluation datasets
        outer_lr: Learning rate η for weight updates
        beta: EMA parameter β
    """
    logging.info("Starting DGA_mixture")
    logging.info(f"Parameters - outer_lr: {outer_lr}, beta: {beta}")

    # Load model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    logging.info(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    logging.info("Model and tokenizer loaded successfully")

    # Initialize weights
    domains = list(train_data_dic.keys())
    logging.info(f"Domains: {domains}")

    alpha_ema = alpha.copy()  # α⁰_EMA
    logging.info(f"Initial weights: {alpha}")

    # Get evaluation dataset
    eval_domain_dataset = {}
    for dataset_name, dataset in eval_datasets.items():
        logging.info(f"Adding evaluation dataset: {dataset_name} with {len(dataset)} samples")
        # 每个dataset里面选择一个数据
        eval_domain_dataset[dataset_name] = random.choice(dataset)

    logging.info(f"Total evaluation samples: {len(eval_domain_dataset)}")

    # Sample batch from each domain and specific set
    domain_batch = {}
    for domain, data in train_data_dic.items():
        if len(data) > 0:
            domain_batch[domain] = random.choice(data)
            logging.info(f"Sampled batch from domain {domain}: {domain_batch[domain]}")
        else:
            logging.warning(f"Empty data for domain {domain}")

    logging.info(f"Sampled evaluation batch: {eval_domain_dataset}")

    # Compute gradient alignments
    alignments = compute_gradient_alignments_multi_task(
        model,
        tokenizer,
        domain_batch,
        eval_domain_dataset,
        device
    )
    logging.info(f"Computed alignments: {alignments}")

    # Update instantaneous weights α^(t+1)
    alpha_hat = {}
    for domain in domains:
        if domain in alignments:
            alpha_hat[domain] = alignments[domain] * np.exp(-outer_lr * alignments[domain])
            logging.info(f"Updated weight for {domain}: {alpha_hat[domain]:.4f}")
        else:
            alpha_hat[domain] = 0
            logging.warning(f"No alignment score for domain {domain}, setting weight to 0")

    # Normalize weights
    alpha_sum = sum(alpha_hat.values())
    alpha = {d: w / alpha_sum for d, w in alpha_hat.items()}
    logging.info(f"Normalized weights: {alpha}")

    # Update EMA weights
    for domain in domains:
        alpha_ema[domain] = (1 - beta) * alpha_ema[domain] + beta * alpha[domain]
        # alpha_ema[domain] = alpha[domain]
    logging.info(f"Updated EMA weights: {alpha_ema}")

    # Save alpha_ema at ./data/mixture_ratio/task_name/round_number.json
    os.makedirs(f"./data/mixture_ratio/{task_name}", exist_ok=True)
    with open(f"./data/mixture_ratio/{task_name}/{round_number}.json", "w") as f:
        json.dump(alpha_ema, f)

    # Clean GPU memory
    torch.cuda.empty_cache()

    # Return EMA weights for sampling
    return alpha_ema


def calculate_potential_factor(loss_history: List[float]) -> float:
    """
    计算domain的潜力因子，基于loss轨迹预测收敛值

    Args:
        loss_history: loss的历史记录

    Returns:
        float: 潜力因子 (0,1] - 越大表示潜力越大
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

        # 计算拟合优度R²
        y_pred = exp_decay(x, *popt)
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

        # 计算潜力分数
        if predicted_loss < current_loss:
            # 基于预测的改进空间
            potential = (current_loss - predicted_loss) / current_loss

            # 根据拟合优度调整潜力值
            potential *= max(0.5, r2)  # 即使拟合不好也保留一定的潜力估计
        else:
            # 如果预测异常，使用历史最小值作为保守估计
            min_loss = min(loss_history)
            potential = max(0, (current_loss - min_loss) / current_loss)

        # 计算预计loss和当前loss的提升量
        improvement = current_loss - predicted_loss

    except Exception as e:
        logging.warning(f"Loss prediction failed: {e}")
        # 发生错误时使用简单的历史数据估计
        min_loss = min(loss_history)
        potential = max(0, (current_loss - min_loss) / current_loss)
        improvement = 0

    return max(0.01, min(1.0, potential)), improvement


def sigmoid(x: float) -> float:
    """Sigmoid函数，用于将数值映射到(0,1)区间"""
    return 1 / (1 + math.exp(-x))


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
        k: int = 20,
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
            normalized_similarities[domain][eval_domain] = kl_diffs[domain] / total_diff
    logging.info(f"Normalized similarities : {normalized_similarities}")

    # 计算每个domain对每个下游任务loss变化的贡献
    domain_contributions = {domain: {eval_domain: 0.0 for eval_domain in eval_domain_samples.keys()}
                            for domain in domains}
    loss_changes = {}  # 存储loss变化情况
    potentials = {}  # 存储每个eval_domain的潜力因子

    # 对每个eval_domain分别计算贡献
    for eval_domain, loss_history in eval_losses.items():
        if len(loss_history) >= 2:
            prev_loss = loss_history[-2]
            curr_loss = loss_history[-1]
            change = prev_loss - curr_loss
            loss_changes[eval_domain] = change

            # 计算潜力因子
            potential, future_improvement = calculate_potential_factor(loss_history)
            potentials[eval_domain] = potential, future_improvement

            # 对该eval_domain计算每个domain的贡献
            for domain in domains:
                # 获取当前domain对该eval_domain的影响力
                similarity = normalized_similarities[domain][eval_domain]

                if change > 0:  # 正向改进
                    # 正向贡献 = 影响力 * (1 + potential因子)
                    contribution = similarity * (1 + potential * 0.5) * change
                else:  # 负向变化
                    # 负向贡献 = -(1-影响力) * (1 - potential因子)
                    contribution = -(1 - similarity) * (1 - potential * 0.3)

                domain_contributions[domain][eval_domain] = contribution

    logging.info(f"potentials: {potentials}")
    logging.info(f"loss_changes: {loss_changes}")
    logging.info(f"domain_contributions: {domain_contributions}")

    # 基于贡献计算新的混合比例

    new_weights = {}
    if len(eval_losses) > 0 and len(potentials) > 0 and len(loss_changes) > 0:
        for domain in domains:
            # 对每个domain，考虑其对所有eval_domain的总体贡献
            domain_total_contribution = 0.0
            total_eval_domains = len(eval_domain_samples)

            for eval_domain in eval_domain_samples.keys():
                domain_total_contribution += domain_contributions[domain][eval_domain] / total_eval_domains

            # 将贡献度保存为权重
            new_weights[domain] = domain_total_contribution
            logging.info(f"Total contribution for {domain}: {domain_total_contribution:.4f}")
    else:
        # 直接根据影响力计算权重
        for domain in domains:
            new_weights[domain] = sum(normalized_similarities[domain].values())

    # 直接对权重进行归一化,使总和为1
    total_weight = sum(new_weights.values())
    alpha_new = {domain: w / total_weight for domain, w in new_weights.items()}
    logging.info(f"New weights: {alpha_new}")

    # 应用EMA更新
    logging.info(f"previous EMA weights: {alpha_ema}")
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
        "raw_weights": new_weights,
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
                     batch_size=10):
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
                grad_diff = eval_grad - domain_grad
                kl_div += 0.5 * torch.sum(fisher_diag * grad_diff * grad_diff)

            domain_kls[eval_domain] = float(kl_div)
            logging.info(f"KL divergence for {domain} with {eval_domain}: {domain_kls[eval_domain]:.4f}")

        kl_divergences[domain] = domain_kls
        logging.info(f"Average KL divergence for {domain}: {kl_divergences[domain]}")

    return kl_divergences
