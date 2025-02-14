# DIDS for Domain Reweighting during Large Language Model Training

## Environment Setup
```bash
pip install -r requirements.txt
pip install traker[fast]
```

## Model Evaluation
```bash
# Basic evaluation
python eval.py --pretrained EleutherAI/pythia-6.9b

# Evaluation with template
python eval.py --pretrained ./data/models/random_100k_10round_meta-llama-Llama-2-7b-hf/ --apply_template
```

## Baseline Training
The baseline code for doremi and doge is available in the 'baseline-doremi' and 'baseline-doge' folders, respectively.

### Random Training
```bash
# Train and test with 10k samples
python baseline_random.py --pretrained meta-llama/Llama-2-7b-hf --total_samples 10000

# Train and test with full dataset (write total_samples using a large enough number for the dataset)
python baseline_random.py --pretrained meta-llama/Llama-3.1-8B --total_samples 10000000
```

### DGA Training
```bash
# Train and test DGA with semantic embedding clustering (256 clusters, 100k samples)
python mixture_DGA.py --pretrained meta-llama/Llama-3.1-8B --total_samples 100000 --num_clusters 256

# Train and test DGA with gradient clustering
python mixture_DGA_gradient_cluster.py --pretrained meta-llama/Llama-3.1-8B --total_samples 100000 --num_clusters 256

# Train and test multi-task DGA
python mixture_DGA_gradient_cluster_multi_task.py --pretrained meta-llama/Llama-3.1-8B --total_samples 100000 --num_clusters 256
```

## DIDS Training
```bash
# Train and test multi task DIDS with different cluster sizes
python mixture_DIDS.py --pretrained meta-llama/Llama-3.1-8B --total_samples 100000 --num_clusters 0 --task DIDS_32clusters


python mixture_DIDS.py --pretrained meta-llama/Llama-3.1-8B --total_samples 100000 --num_clusters 2 --task DIDS_32clusters

# Train and test single task DIDS
python mixture_DIDS_single_domain.py --pretrained meta-llama/Llama-3.1-8B --total_samples 100000 --num_clusters 32
```

## Dataset Information
The Tulu-3 dataset includes 18 different data sources (https://huggingface.co/datasets/allenai/tulu-3-sft-mixture):
- AI2 Adapt Dev collections (OASST1, FLAN v2, WildChat, Math, Code)
- Persona-based datasets
- Special task datasets (SciFi, Table GPT, AYA)
- Safety and security datasets

```bash
[
'ai2-adapt-dev/oasst1_converted', 
'ai2-adapt-dev/flan_v2_converted',
'ai2-adapt-dev/tulu_hard_coded_repeated_10', 
'ai2-adapt-dev/no_robots_converted', 
'ai2-adapt-dev/tulu_v3.9_wildchat_100k', 'ai2-adapt-dev/personahub_math_v5_regen_149960',
'allenai/tulu-3-sft-personas-math-grade', 
'ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k', 'ai2-adapt-dev/numinamath_tir_math_decontaminated',
'ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k',
'ai2-adapt-dev/personahub_code_v2_34999', 
'ai2-adapt-dev/evol_codealpaca_heval_decontaminated', 
'ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980', 
'ai2-adapt-dev/coconot_converted', 
'ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k', 
'ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k',
'ai2-adapt-dev/tulu_v3.9_sciriff_10k', 
'ai2-adapt-dev/tulu_v3.9_table_gpt_5k', 
'ai2-adapt-dev/tulu_v3.9_aya_100k'
]
```

## Troubleshooting
For large-scale data processing, you may need to:
```bash
pip install pyarrow -U
pip install langdetect  # Required for ifeval
```

Note: When working with large datasets, consider using lower versions of numpy and pandas to avoid errors. Process data in JSONL format instead of JSON for better performance.
