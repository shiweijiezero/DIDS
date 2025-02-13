import os

import datasets
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import logging
import pickle
from functools import wraps
from tqdm import tqdm

class DatasetCache:
    def __init__(self, cache_dir: str = "./cache/datasets"):
        """
        初始化数据集缓存
        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, dataset_name: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{dataset_name}.pkl"

    def load_cache(self, dataset_name: str) -> List[List[Dict[str, str]]] | None:
        """加载缓存的数据集"""
        cache_path = self.get_cache_path(dataset_name)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logging.warning(f"Failed to load cache for {dataset_name}: {e}")
            return None

    def save_cache(self, dataset_name: str, data: List[List[Dict[str, str]]]):
        """保存数据集到缓存"""
        cache_path = self.get_cache_path(dataset_name)
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logging.warning(f"Failed to save cache for {dataset_name}: {e}")

def cached_dataset(func):
    """数据集加载函数的缓存装饰器"""
    cache = DatasetCache()

    @wraps(func)
    def wrapper(split, *args, **kwargs):
        # 使用函数名和split参数作为缓存键
        cache_key = f"{func.__name__}_{split}"
        print(cache_key)

        # 尝试从缓存加载
        cached_data = cache.load_cache(cache_key)
        if cached_data is not None:
            logging.info(f"Loaded {cache_key} from cache")
            return cached_data

        # 如果没有缓存,执行原始函数
        data = func(split=split, *args, **kwargs)

        # 保存到缓存
        cache.save_cache(cache_key, data)
        return data
    return wrapper

def load_mmlu_dataset_impl(split="validation"):
    """加载MMLU数据集"""
    logging.info("Loading MMLU dataset")
    dataset = datasets.load_dataset("cais/mmlu", "all", split=split)
    eval_data = []

    for item in dataset:
        choices = []
        for choice in ['choices']:
            if choice in item:
                choices = item[choice]
                break

        if not choices:
            logging.warning("No choices found in MMLU item")
            continue

        try:
            question = f"Below is a multiple choice question. Please select the correct answer.\n\nQuestion: {item['question']}\n"
            for i, choice in enumerate(choices):
                question += f"{chr(65 + i)}. {choice}\n"

            answer_idx = item['answer'] if isinstance(item['answer'], int) else 0
            answer = f"{chr(65 + answer_idx)}. {choices[answer_idx]}"

            eval_data.append([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ])
        except Exception as e:
            logging.warning(f"Error processing MMLU item: {e}")
            continue

    return eval_data

def load_truthfulqa_dataset_impl(split="validation"):
    """加载TruthfulQA数据集"""
    logging.info("Loading TruthfulQA dataset")
    dataset = datasets.load_dataset("truthfulqa/truthful_qa", "generation", split=split)
    eval_data = []

    for item in dataset:
        question = f"Please answer the following question truthfully and accurately:\n\n{item['question']}"
        answer = item["best_answer"]
        eval_data.append([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ])
    return eval_data

def load_bbh_dataset_impl(split="test"):
    """加载BBH数据集"""
    logging.info("Loading BBH dataset {}".format(split))
    task_names = ['boolean_expressions', 'causal_judgement', 'date_understanding',
                  'disambiguation_qa', 'dyck_languages', 'formal_fallacies',
                  'geometric_shapes', 'hyperbaton', 'logical_deduction_five_objects',
                  'logical_deduction_seven_objects', 'logical_deduction_three_objects',
                  'movie_recommendation', 'multistep_arithmetic_two', 'navigate',
                  'object_counting', 'penguins_in_a_table', 'reasoning_about_colored_objects',
                  'ruin_names', 'salient_translation_error_detection', 'snarks',
                  'sports_understanding', 'temporal_sequences',
                  'tracking_shuffled_objects_five_objects',
                  'tracking_shuffled_objects_seven_objects',
                  'tracking_shuffled_objects_three_objects', 'web_of_lies', 'word_sorting']

    eval_data = []

    for task in tqdm(task_names, desc="Loading BBH tasks"):
        try:
            dataset = datasets.load_dataset("lukaemon/bbh", task, split=split)

            for item in dataset:
                question = f"Please solve the following {task} problem:\n\n{item['input']}"
                answer = item["target"]
                eval_data.append([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ])

            logging.info(f"Successfully loaded BBH task: {task}")
        except Exception as e:
            logging.warning(f"Failed to load BBH task {task}: {e}")
            continue

    return eval_data

def load_gsm8k_dataset_impl(split="train"):
    """加载GSM8K数据集"""
    logging.info("Loading GSM8K dataset")
    dataset = datasets.load_dataset("openai/gsm8k", "main", split=split)
    eval_data = []

    for item in dataset:
        question = f"Please solve this math word problem. Show your step-by-step solution.\n\n{item['question']}"
        answer = item["answer"]
        eval_data.append([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ])
    return eval_data

def load_mathqa_dataset_impl(split="validation"):
    """加载MathQA数据集"""
    logging.info("Loading MathQA dataset")
    dataset = datasets.load_dataset("math_qa", split=split)
    eval_data = []

    for item in dataset:
        question = f"Please solve the following multiple choice math problem. Show your reasoning and select the correct answer choice.\n\n" \
                   f"Question: {item['Problem']}\n\n" \
                   f"Options:\n{item['options']}"

        answer = item['Rationale']

        eval_data.append([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ])
    return eval_data

# ------------------ 新添加的数据集加载函数 ------------------
def load_arc_easy_dataset_impl(split="validation"):
    """加载ARC-Easy数据集"""
    logging.info("Loading ARC-Easy dataset")
    dataset = datasets.load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)
    eval_data = []

    for item in dataset:
        choices = item['choices']['text']
        prefix_choices = item['choices']['label']
        question = f"Below is a multiple choice science question. Please select the correct answer.\n\nQuestion: {item['question']}\n"
        for i, choice in enumerate(choices):
            question += f"{prefix_choices[i]}. {choice}\n"

        answer_laabel = item['answerKey'] # A, B, C, D
        answer = f"{answer_laabel}. {choices[prefix_choices.index(answer_laabel)]}"

        eval_data.append([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ])
    return eval_data

def load_boolq_dataset_impl(split="validation"):
    """加载BoolQ数据集"""
    logging.info("Loading BoolQ dataset")
    dataset = datasets.load_dataset("google/boolq", split=split)
    eval_data = []

    for item in dataset:
        question = f"Based on the following passage, please answer true or false:\n\nPassage: {item['passage']}\n\nQuestion: {item['question']}"
        answer = "true" if item['answer'] else "false"
        eval_data.append([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ])
    return eval_data

def load_ifeval_dataset_impl(split="train"):
    """加载IF-Eval数据集
    处理格式:
    - key: 提示ID
    - prompt: 任务描述
    - instruction_id_list: 可验证指令列表
    - kwargs: 指令参数列表
    """
    logging.info("Loading IF-Eval dataset")
    dataset = datasets.load_dataset("google/IFEval", split=split)
    eval_data = []

    for item in dataset:
        # 构建完整的提示，包含所有指令要求
        instruction_context = "Instructions:\n"
        for idx, instruction_id in enumerate(item['instruction_id_list']):
            kwargs = item['kwargs'][idx]
            instruction_context += f"- {instruction_id}"
            if kwargs:
                # 添加非空的kwargs参数
                params = {k: v for k, v in kwargs.items() if v is not None}
                if params:
                    instruction_context += f" with parameters: {params}"
            instruction_context += "\n"

        question = f"{instruction_context}\nTask: {item['prompt']}"

        # 目前暂时将answer留空，因为数据集中可能没有标准答案
        # 在实际评估时需要根据instruction_id_list中的要求来验证生成的答案
        answer = "Please provide a response following all the specified instructions."

        eval_data.append([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ])
    return eval_data

def load_logiqa_dataset_impl(split="validation"):
    """
    加载LogiQA数据集

    LogiQA包含以下字段：
    - context: 背景文本
    - query: 问题
    - options: 选项列表
    - correct_option: 正确答案的索引

    Args:
        split (str): 数据集分割，可选 "train"、"validation" 或 "test"

    Returns:
        List[List[Dict[str, str]]]: 格式化后的对话数据
    """
    logging.info(f"Loading LogiQA dataset - {split} split")
    dataset = datasets.load_dataset("lucasmccabe/logiqa", split=split)
    eval_data = []

    for item in dataset:
        # 构建问题文本，包含背景和问题
        question = (
            f"Based on the following context, please answer the question by selecting the most appropriate option.\n\n"
            f"Context: {item['context']}\n\n"
            f"Question: {item['query']}\n\n"
            "Options:\n"
        )

        # 添加选项
        for i, option in enumerate(item['options']):
            question += f"{chr(65 + i)}. {option}\n"

        # 获取正确答案
        correct_idx = item['correct_option']
        answer = f"{chr(65 + correct_idx)}. {item['options'][correct_idx]}"

        # 添加完整的解释
        full_answer = (
            f"{answer}\n\n"
            f"This is the correct answer because it directly addresses the question "
            f"based on the logical reasoning required by the context. "
            f"The context provides information about {item['context'].split('.')[0].lower()}, "
            f"and this option best aligns with the logical implications of the given scenario."
        )

        eval_data.append([
            {"role": "user", "content": question},
            {"role": "assistant", "content": full_answer}
        ])

    logging.info(f"Loaded {len(eval_data)} examples from LogiQA {split} split")
    return eval_data

def load_minerva_math_dataset_impl(split="train"):
    """加载Competition Math数据集

    Args:
        split: 数据集分割，可选 "train" 或 "test"

    Returns:
        包含数学竞赛题目和解答的对话列表
    """
    logging.info("Loading Competition Math dataset")
    dataset = datasets.load_dataset("qwedsacf/competition_math", split=split, trust_remote_code=True)
    eval_data = []

    for item in dataset:
        # 构建问题提示，包含难度和类型信息
        prompt = f"Please solve this {item['level']} {item['type']} problem. Show your solution step by step and provide the final answer in a \\boxed{{}} command.\n\n"
        question = prompt + item['problem']

        # 解答已经包含了步骤和最终答案
        answer = item['solution']

        eval_data.append([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ])
    return eval_data

def load_piqa_dataset_impl(split="validation"):
    """加载PIQA数据集"""
    logging.info("Loading PIQA dataset")
    dataset = datasets.load_dataset("piqa", split=split)
    eval_data = []

    for item in dataset:
        question = f"Choose the most appropriate solution:\n\nGoal: {item['goal']}\n\nOptions:\nA. {item['sol1']}\nB. {item['sol2']}"
        answer = f"{'A' if item['label'] == 0 else 'B'}. {item['sol1'] if item['label'] == 0 else item['sol2']}"
        eval_data.append([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ])
    return eval_data

def load_pubmedqa_dataset_impl(split="train"):
    """加载PubMedQA数据集"""
    logging.info("Loading PubMedQA dataset")
    dataset = datasets.load_dataset("qiaojin/PubMedQA", "pqa_labeled", split=split)
    eval_data = []

    for item in dataset:
        context = "\n".join(item['context']['contexts'])
        question = f"Based on the following medical research abstract, please answer yes/no/maybe:\n\nAbstract: {context}\n\nQuestion: {item['question']}"
        answer = item['final_decision'].upper()
        eval_data.append([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ])
    return eval_data

# 为新添加的数据集加载函数添加缓存装饰器
@cached_dataset
def load_arc_easy_dataset(split="validation"):
    return load_arc_easy_dataset_impl(split)

@cached_dataset
def load_boolq_dataset(split="validation"):
    return load_boolq_dataset_impl(split)

@cached_dataset
def load_ifeval_dataset(split="train"):
    return load_ifeval_dataset_impl(split)

@cached_dataset
def load_logiqa_dataset(split="validation"):
    return load_logiqa_dataset_impl(split)

@cached_dataset
def load_minerva_math_dataset(split="train"):
    return load_minerva_math_dataset_impl(split)

@cached_dataset
def load_piqa_dataset(split="validation"):
    return load_piqa_dataset_impl(split)

@cached_dataset
def load_pubmedqa_dataset(split="train"):
    return load_pubmedqa_dataset_impl(split)

# 为每个数据集加载函数添加缓存
@cached_dataset
def load_mmlu_dataset(split="validation"):
    """加载MMLU数据集"""
    return load_mmlu_dataset_impl(split)

@cached_dataset
def load_truthfulqa_dataset(split="validation"):
    """加载TruthfulQA数据集"""
    return load_truthfulqa_dataset_impl(split)

@cached_dataset
def load_bbh_dataset(split="test"):
    """加载BBH数据集"""
    return load_bbh_dataset_impl(split)

@cached_dataset
def load_gsm8k_dataset(split="train"):
    """加载GSM8K数据集"""
    return load_gsm8k_dataset_impl(split)

@cached_dataset
def load_mathqa_dataset(split="validation"):
    """加载MathQA数据集"""
    return load_mathqa_dataset_impl(split)

def load_eval_datasets(use_cache: bool = True) -> Dict[str, List[Dict[str, Any]]]:
    """
    加载所有评估数据集
    Args:
        use_cache: 是否使用缓存
    Returns:
        Dict[str, List[List[Dict[str, str]]]]: 数据集名称到对话数据的映射
    """
    eval_datasets = {}
    dataset_configs = {
        "mmlu": ("validation", load_mmlu_dataset),
        "truthfulqa": ("validation", load_truthfulqa_dataset),
        "bbh": ("test", load_bbh_dataset),
        "gsm8k": ("train", load_gsm8k_dataset),
        "mathqa": ("validation", load_mathqa_dataset),
        # 添加新的数据集配置
        "arc_easy": ("validation", load_arc_easy_dataset),
        "boolq": ("validation", load_boolq_dataset),
        "ifeval": ("train", load_ifeval_dataset),
        "logiqa": ("validation", load_logiqa_dataset),
        "minerva_math": ("train", load_minerva_math_dataset),
        "piqa": ("validation", load_piqa_dataset),
        "pubmedqa": ("train", load_pubmedqa_dataset)
    }

    for name, (split, loader) in dataset_configs.items():
        try:
            eval_datasets[name] = loader(split=split) if use_cache else loader.__wrapped__(split=split)
            logging.info(f"Loaded {name} dataset with {len(eval_datasets[name])} examples")
        except Exception as e:
            logging.warning(f"Failed to load {name} dataset: {e}")

    return eval_datasets

def load_relevant_trains():
    """
    加载与评估数据集相关的训练数据
    Returns:
        Dict: 数据集名称到训练数据的映射
    """
    dic = {}

    # MMLU相关训练数据
    try:
        dataset = datasets.load_dataset("cais/mmlu", "all", split="auxiliary_train")
        dic["mmlu_train"] = []
        for item in tqdm(dataset, desc="Processing MMLU training data"):
            choices = []
            for choice in ['choices']:
                if choice in item:
                    choices = item[choice]
                    break

            try:
                question = f"Below is a multiple choice question. Please select the correct answer.\n\nQuestion: {item['question']}\n"
                for i, choice in enumerate(choices):
                    question += f"{chr(65 + i)}. {choice}\n"

                answer_idx = item['answer'] if isinstance(item['answer'], int) else 0
                answer = f"{chr(65 + answer_idx)}. {choices[answer_idx]}"

                dic["mmlu_train"].append({
                    "conversations": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                    ]
                })
            except Exception as e:
                logging.warning(f"Error processing MMLU training item: {e}")
                continue
        # dic["mmlu_train"] 随机保留前20k
        dic["mmlu_train"] = np.random.choice(dic["mmlu_train"], 20000, replace=False).tolist()
    except Exception as e:
        logging.warning(f"Failed to load MMLU training data: {e}")

    # TruthfulQA相关训练数据
    try:
        dataset = datasets.load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
        dic["truthfulqa_train"] = []
        for item in tqdm(dataset, desc="Processing TruthfulQA training data"):
            question = f"Please answer the following question truthfully and accurately:\n\n{item['question']}"
            answer = item["best_answer"]
            dic["truthfulqa_train"].append({
                "conversations": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            })
    except Exception as e:
        logging.warning(f"Failed to load TruthfulQA training data: {e}")

    # BBH相关训练数据
    try:
        task_names = ['boolean_expressions', 'causal_judgement', 'date_understanding',
                      'disambiguation_qa', 'dyck_languages', 'formal_fallacies',
                      'geometric_shapes', 'hyperbaton', 'logical_deduction_five_objects',
                      'logical_deduction_seven_objects', 'logical_deduction_three_objects',
                      'movie_recommendation', 'multistep_arithmetic_two', 'navigate',
                      'object_counting', 'penguins_in_a_table', 'reasoning_about_colored_objects',
                      'ruin_names', 'salient_translation_error_detection', 'snarks',
                      'sports_understanding', 'temporal_sequences',
                      'tracking_shuffled_objects_five_objects',
                      'tracking_shuffled_objects_seven_objects',
                      'tracking_shuffled_objects_three_objects', 'web_of_lies', 'word_sorting']

        dic["bbh_train"] = []
        for task in tqdm(task_names, desc="Processing BBH training data"):
            try:
                dataset = datasets.load_dataset("lukaemon/bbh", task, split="test")
                for item in dataset:
                    question = f"Please solve the following {task} problem:\n\n{item['input']}"
                    answer = item["target"]
                    dic["bbh_train"].append({
                        "conversations": [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer}
                        ]
                    })
            except Exception as e:
                logging.warning(f"Failed to load BBH task {task}: {e}")
                continue
    except Exception as e:
        logging.warning(f"Failed to load BBH training data: {e}")

    # GSM8K相关训练数据
    try:
        dataset = datasets.load_dataset("openai/gsm8k", "main", split="train")
        dic["gsm8k_train"] = []
        for item in tqdm(dataset, desc="Processing GSM8K training data"):
            question = f"Please solve this math word problem. Show your step-by-step solution.\n\n{item['question']}"
            answer = item["answer"]
            dic["gsm8k_train"].append({
                "conversations": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            })
    except Exception as e:
        logging.warning(f"Failed to load GSM8K training data: {e}")

    # MathQA相关训练数据
    try:
        dataset = datasets.load_dataset("math_qa", split="train")
        dic["mathqa_train"] = []
        for item in tqdm(dataset, desc="Processing MathQA training data"):
            question = f"Please solve the following multiple choice math problem. Show your reasoning and select the correct answer choice.\n\n" \
                       f"Question: {item['Problem']}\n\n" \
                       f"Options:\n{item['options']}"
            answer = item['Rationale']
            dic["mathqa_train"].append({
                "conversations": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            })
    except Exception as e:
        logging.warning(f"Failed to load MathQA training data: {e}")

    # ARC-Easy相关训练数据
    try:
        dataset = datasets.load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
        dic["arc_easy_train"] = []
        for item in tqdm(dataset, desc="Processing ARC-Easy training data"):
            choices = item['choices']['text']
            prefix_choices = item['choices']['label']
            question = f"Below is a multiple choice science question. Please select the correct answer.\n\nQuestion: {item['question']}\n"
            for i, choice in enumerate(choices):
                question += f"{prefix_choices[i]}. {choice}\n"

            answer_label = item['answerKey']
            answer = f"{answer_label}. {choices[prefix_choices.index(answer_label)]}"

            dic["arc_easy_train"].append({
                "conversations": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            })
    except Exception as e:
        logging.warning(f"Failed to load ARC-Easy training data: {e}")

    # BoolQ相关训练数据
    try:
        dataset = datasets.load_dataset("google/boolq", split="train")
        dic["boolq_train"] = []
        for item in tqdm(dataset, desc="Processing BoolQ training data"):
            question = f"Based on the following passage, please answer true or false:\n\nPassage: {item['passage']}\n\nQuestion: {item['question']}"
            answer = "true" if item['answer'] else "false"
            dic["boolq_train"].append({
                "conversations": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            })
    except Exception as e:
        logging.warning(f"Failed to load BoolQ training data: {e}")

    # IF-Eval相关训练数据
    try:
        dataset = datasets.load_dataset("google/IFEval", split="train")
        dic["ifeval_train"] = []
        for item in tqdm(dataset, desc="Processing IF-Eval training data"):
            instruction_context = "Instructions:\n"
            for idx, instruction_id in enumerate(item['instruction_id_list']):
                kwargs = item['kwargs'][idx]
                instruction_context += f"- {instruction_id}"
                if kwargs:
                    params = {k: v for k, v in kwargs.items() if v is not None}
                    if params:
                        instruction_context += f" with parameters: {params}"
                instruction_context += "\n"

            question = f"{instruction_context}\nTask: {item['prompt']}"
            answer = "Please provide a response following all the specified instructions."

            dic["ifeval_train"].append({
                "conversations": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            })
    except Exception as e:
        logging.warning(f"Failed to load IF-Eval training data: {e}")

    # LogiQA相关训练数据
    try:
        dataset = datasets.load_dataset("lucasmccabe/logiqa", split="train")
        dic["logiqa_train"] = []
        for item in tqdm(dataset, desc="Processing LogiQA training data"):
            question = (
                f"Based on the following context, please answer the question by selecting the most appropriate option.\n\n"
                f"Context: {item['context']}\n\n"
                f"Question: {item['query']}\n\n"
                "Options:\n"
            )

            for i, option in enumerate(item['options']):
                question += f"{chr(65 + i)}. {option}\n"

            correct_idx = item['correct_option']
            answer = f"{chr(65 + correct_idx)}. {item['options'][correct_idx]}"

            dic["logiqa_train"].append({
                "conversations": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            })
    except Exception as e:
        logging.warning(f"Failed to load LogiQA training data: {e}")

    # Minerva Math (Competition Math)相关训练数据
    try:
        dataset = datasets.load_dataset("qwedsacf/competition_math", split="train", trust_remote_code=True)
        dic["minerva_math_train"] = []
        for item in tqdm(dataset, desc="Processing Minerva Math training data"):
            prompt = f"Please solve this {item['level']} {item['type']} problem. Show your solution step by step and provide the final answer in a \\boxed{{}} command.\n\n"
            question = prompt + item['problem']
            answer = item['solution']

            dic["minerva_math_train"].append({
                "conversations": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            })
    except Exception as e:
        logging.warning(f"Failed to load Minerva Math training data: {e}")

    # PIQA相关训练数据
    try:
        dataset = datasets.load_dataset("piqa", split="train")
        dic["piqa_train"] = []
        for item in tqdm(dataset, desc="Processing PIQA training data"):
            question = f"Choose the most appropriate solution:\n\nGoal: {item['goal']}\n\nOptions:\nA. {item['sol1']}\nB. {item['sol2']}"
            answer = f"{'A' if item['label'] == 0 else 'B'}. {item['sol1'] if item['label'] == 0 else item['sol2']}"
            dic["piqa_train"].append({
                "conversations": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            })
    except Exception as e:
        logging.warning(f"Failed to load PIQA training data: {e}")

    # PubMedQA相关训练数据
    try:
        dataset = datasets.load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        dic["pubmedqa_train"] = []
        for item in tqdm(dataset, desc="Processing PubMedQA training data"):
            context = "\n".join(item['context']['contexts'])
            question = f"Based on the following medical research abstract, please answer yes/no/maybe:\n\nAbstract: {context}\n\nQuestion: {item['question']}"
            answer = item['final_decision'].upper()
            dic["pubmedqa_train"].append({
                "conversations": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            })
    except Exception as e:
        logging.warning(f"Failed to load PubMedQA training data: {e}")

    return dic


if __name__ == "__main__":
    # 设置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 测试缓存功能
    print("First load (no cache):")
    eval_datasets = load_eval_datasets()

    # print("\nSecond load (should use cache):")
    # eval_datasets = load_eval_datasets()