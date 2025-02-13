import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import math
import re

TASKS_AND_METRICS = {
    'bbh': ('exact_match,get-answer', 'BBH Exact Match'),
    'gsm8k': ('exact_match,strict-match', 'GSM8K Exact Match'),
    'piqa': ('acc,none', 'PIQA Accuracy'),
    'pubmedqa': ('acc,none', 'PubMedQA Accuracy'),
    'hellaswag': ('acc,none', 'HellaSwag Accuracy'),
    'boolq': ('acc,none', 'BoolQ Accuracy'),
    'arc_easy': ('acc,none', 'ARC Easy Accuracy'),
    'logiqa': ('acc,none', 'LogiQA Accuracy'),
    'ifeval': ('prompt_level_strict_acc,none', 'IFEval Accuracy'),
    'mmlu': ('acc,none', 'MMLU Accuracy'),
    'truthfulqa_mc1': ('acc,none', 'TruthfulQA Accuracy'),
    'mathqa': ('acc,none', 'MathQA Accuracy'),
    'minerva_math': ('exact_match,none', 'Minerva Math Exact Match'),
    'drop': ('em,none', 'DROP Exact Match')
}

# Updated color scheme
MODEL_COLORS = {
    ('llama2', '2e6'): '#FF8C00',  # Dark Orange
    ('llama2', '1e5'): '#FFD700',  # Gold
    ('llama3', '2e6'): '#32CD32',  # Lime Green
    ('llama3', '1e5'): '#98FB98',  # Pale Green
    ('pythia', '2e6'): '#4169E1',  # Royal Blue
    ('pythia', '1e5'): '#87CEEB',  # Sky Blue
    ('tulu2', '2e6'): '#8B008B',   # Dark Magenta
    ('tulu2', '1e5'): '#DA70D6',   # Orchid
    ('tulu3', '2e6'): '#CD853F',   # Peru
    ('tulu3', '1e5'): '#DEB887',   # Burlywood
    ('raw', 'none'): '#808080',     # Gray
    ('other', '2e6'): '#BA55D3',   # Medium Orchid
    ('other', '1e5'): '#DDA0DD'    # Plum
}

def get_raw_value(model_name, results_data, task_name, metric_name):
    """Get raw model value for comparison."""
    if 'llama-2-7b' in model_name.lower():
        raw_model = 'llama-2-7B-raw-NAtemplate'
    elif 'llama-3.1-8b' in model_name.lower():
        raw_model = 'llama-3.1-8B-raw-NAtemplate'
    elif 'pythia-6.9b' in model_name.lower():
        raw_model = 'pythia-6.9B-raw-NAtemplate'
    else:
        return None

    if raw_model in results_data and task_name in results_data[raw_model]['results']:
        return results_data[raw_model]['results'][task_name].get(metric_name) * 100
    return None

def get_model_type_and_lr(model_name):
    """Extract model type and learning rate from model name."""
    model_name = model_name.lower()

    # Check for raw models first
    if 'raw' in model_name:
        return 'raw', 'none'

    # Check for specific model types
    if 'llama-3.1' in model_name:
        model_type = 'llama3'
    elif 'llama-2' in model_name:
        model_type = 'llama2'
    elif 'pythia' in model_name:
        model_type = 'pythia'
    elif 'tulu-3' in model_name:
        model_type = 'tulu3'
    elif 'tulu-2' in model_name:
        model_type = 'tulu2'
    else:
        model_type = 'other'

    # Extract learning rate
    lr_match = re.search(r'lr(2e6|1e5)', model_name)
    lr = lr_match.group(1) if lr_match else '2e6'  # default to 2e6 if not found

    return model_type, lr

def get_bar_color(model_name):
    """Get the appropriate color for a model based on its type and learning rate."""
    model_type, lr = get_model_type_and_lr(model_name)
    return MODEL_COLORS.get((model_type, lr), MODEL_COLORS[('other', '2e6')])

def load_results(directory):
    """Load all result files from the specified directory."""
    results_data = {}
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            model_name = filename.replace('.json', '')
            with open(os.path.join(directory, filename), 'r') as f:
                results_data[model_name] = json.load(f)
    return results_data

def create_performance_subplot(ax, results_data, task_name, metric_name, plot_title, figsize=None):
    """Create a subplot for a specific task's performance comparison."""
    performances = []
    models = []
    colors = []

    # Collect data for plotting
    for model_name, data in results_data.items():
        if task_name in data['results']:
            metric_value = data['results'][task_name].get(metric_name)
            if metric_value is not None:
                performances.append(metric_value * 100)
                models.append(model_name)
                colors.append(get_bar_color(model_name))

    if not performances:
        print(f"No data found for task: {task_name}")
        return

    # Sort data by performance
    sorted_data = sorted(zip(models, performances, colors), key=lambda x: x[1], reverse=True)
    models, performances, colors = zip(*sorted_data)

    # Create bar plot
    bars = ax.bar(range(len(models)), performances, width=0.8, color=colors)

    # Configure plot appearance
    if figsize:
        title_size, label_size, tick_size, value_size = 14, 12, 10, 8
    else:
        title_size, label_size, tick_size, value_size = 10, 8, 6, 6

    ax.set_title(plot_title, pad=10, fontsize=title_size)
    ax.set_xlabel('Models', fontsize=label_size)
    ax.set_ylabel('Performance (%)', fontsize=label_size)

    # Configure x-axis
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=tick_size)

    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        raw_value = get_raw_value(models[i], results_data, task_name, metric_name)

        if raw_value is not None:
            diff = height - raw_value
            label = f'{height:.1f}%\n(Δ{diff:+.1f}%)'
        else:
            label = f'{height:.1f}%'

        ax.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom', fontsize=value_size)

    # Add legend with updated model types
    legend_elements = []
    for model_type in ['llama2', 'llama3', 'pythia', 'tulu2', 'tulu3', 'raw', 'other']:
        if model_type == 'raw':
            color = MODEL_COLORS[(model_type, 'none')]
            label = f'{model_type.upper()}'
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=label))
        else:
            for lr in ['2e6', '1e5']:
                if (model_type, lr) in MODEL_COLORS:
                    color = MODEL_COLORS[(model_type, lr)]
                    label = f'{model_type.upper()} (lr={lr})'
                    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=label))

    ax.legend(handles=legend_elements, loc='upper right', fontsize=value_size)

def create_single_task_plot(results_data, task_name, metric_name, plot_title):
    """Create and save an individual plot for a specific task."""
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    create_performance_subplot(ax, results_data, task_name, metric_name, plot_title, figsize=(12, 6))
    plt.tight_layout()

    # Save plot
    os.makedirs('visual/individual', exist_ok=True)
    filename = f'visual/individual/{task_name}_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved individual plot: {filename}")

def main():
    """Main execution function."""
    # Load results
    results_data = load_results('results')

    # Create individual plots
    print("Creating individual plots...")
    for task_name, (metric_name, plot_title) in TASKS_AND_METRICS.items():
        create_single_task_plot(results_data, task_name, metric_name, plot_title)

    # Create combined plot
    print("Creating combined plot...")
    n_tasks = len(TASKS_AND_METRICS)
    n_cols = 3
    n_rows = math.ceil(n_tasks / n_cols)

    plt.figure(figsize=(20, 5*n_rows))

    for i, (task_name, (metric_name, plot_title)) in enumerate(TASKS_AND_METRICS.items()):
        ax = plt.subplot(n_rows, n_cols, i+1)
        create_performance_subplot(ax, results_data, task_name, metric_name, plot_title)

    plt.tight_layout()

    # Save combined plot
    os.makedirs('visual', exist_ok=True)
    plt.savefig('visual/combined_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved combined plot: visual/combined_comparison.png")

if __name__ == "__main__":
    main()