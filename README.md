# Data-efficient LLM Fine-tuning for Code Generation

[![Paper](https://img.shields.io/badge/Paper-arXiv?logo=arxiv&logoColor=%23B31B1B&label=arXiv&labelColor=%23f5f5dc&color=%23B31B1B)]()

## Contents

- [Overview](#overview)
- [Method](#Method)
- [Results](#results)
- [Usage](#usage)
- [Citation](#citation)

## Overview

In this work, we propose a data selection strategy in order to improve the effectiveness and efficiency of training for code-based LLMs. By prioritizing data complexity and ensuring that the sampled subset aligns with the distribution of the original dataset, our sampling strategy effectively selects high-quality data. Additionally, we optimize the tokenization process through a "dynamic pack" technique, which minimizes padding tokens and reduces computational resource consumption.

Experimental results show that when training on 40\% of the OSS-Instruct dataset, the DeepSeek-Coder-Base-6.7B model achieves an average performance of 66.9\%, surpassing the 66.1\% performance with the full dataset. Moreover, training time is reduced from 47 minutes to 34 minutes, and the peak GPU memory usage decreases from 61.47 GB to 42.72 GB during a single epoch. Similar improvements are observed with the CodeLlama-Python-7B model on the Evol-Instruct dataset.

## Method

### Data Selection Strategy

<div style="text-align: center;">
<img alt="Overview of our data selection strategy" src="assets/data_selection.png">
</div>

The overview of our proposed data selection strategy, including three steps.

* Step 1: Partitioning the synthetic dataset into multiple clusters.
* Step 2: Computing the Instruction Following Difficulty score by comparing the model's perplexity with and without instructions. 
* Step 3: Sampling the top m\% instances from each re-ranked cluster to form a high-complexity sub-dataset that preserves data consistency.

Finally, the selected data is used for fine-tuning open-source code LLMs.

### Data Tokenization Strategy

<div style="text-align: center;">
<img alt="Illustration of different padding strategies" src="assets/dynamic_pack.png">
</div>

Illustration of different padding strategies, where the blank squares represent padding tokens. 
* Top: Traditional padding strategy aligns samples to the model's maximum input length, resulting in high computational resource consumption. 
* Middle: Dynamic padding strategy reduces the number of padding tokens by aligning samples to the length of the longest sample in each batch. 
* Bottom: Our proposed "dynamic pack" strategy sorts samples by length and concatenates multiple samples within a batch, further optimizing the utilization of the model's maximum input length and reducing padding tokens.

## Results

### Main Results

By prioritizing high-complexity subsets while ensuring that the distribution aligns with the original dataset, our data selection strategy demonstrates its effectiveness across different models and datasets. Selecting high-quality data not only improves performance but also significantly improves training efficiency.

<div style="text-align: center;">
<img alt="Perfomance Comparison" src="assets/comparison_perf.png">
</div>

### Sampling Rates

Impact of sampling rates (ranging from 10\% to 60\%) on model performance. The results demonstrate that model performance peaks at sampling rates between 30\% and 40\%, after which it begins to decline. "Average Full Data" denotes the model's average performance on the HumanEval and MBPP benchmarks when trained on the full data. "Average" reflects the model's average performance on these two benchmarks at different sampling rates.

<div style="text-align: center;">
<img alt="Sampling Rate Comparison" src="assets/comparison_sr.png">
</div>

### Training Efficiency

We evaluate the effectiveness of our "dynamic pack" technique in optimizing training efficiency across different models and datasets. The results include the training time and the peak GPU memory usage during one training epoch. To monitor the GPU memory consumption, we utilize the `torch.cuda.max_memory_allocated` function. DS denotes DeepSeek-Coder, and CL denotes CodeLlama.

<div style="text-align: center;">
<img alt="Efficiency Comparison" src="assets/comparison_eff.png"> 
</div>

## Usage

### Installation

#### 1. Install required packages

- Python 3.11.9
- PyTorch 2.3.0
- CUDA 12.1

```bash
conda create -n finetune python=3.11 -y
conda activate finetune
pip install -r requirements.txt
```

#### 2. Install additional packages for training

Before installing the following packages, please review the installation instructions and requirements on their respective repositories to ensure compatibility with your environment.

- [*DeepSpeed*](https://github.com/microsoft/DeepSpeed?tab=readme-ov-file#pypi)
- [*Flash-Attention*](https://github.com/Dao-AILab/flash-attention#installation-and-features)

```bash
pip install deepspeed==0.14.2
pip install flash-attn==2.5.9.post1 --no-build-isolation
```

### Datasets

- [EVOL-Instruct](https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1): An open-source implementation of Evol-Instruct as described in the [WizardCoder Paper](https://arxiv.org/abs/2306.08568).
- [OSS-Instruct](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K): This dataset is generated by gpt-3.5-turbo-1106. More details can be found in the [Magicoder Paper](https://arxiv.org/abs/2312.02120).

### Data Selection

Our data selection strategy is rooted in the necessity of considering both the complexity and consistency of data to enhance model training efficiency and effectiveness.

#### 1. Calculate IFD scores

To efficiently handle large datasets, we support processing data in chunks. Each chunk is defined by the ***index*** and ***nums*** parameters, which determine the range of data to be processed.

**Note:** 
- For EVOL-Instruct dataset, use `instruction` for ***question_column_name*** and `output` for ***answer_column_name***.
- For OSS-Instruct dataset, use `problem` for ***question_column_name*** and `solution` for ***answer_column_name***.

```bash
# Calculate IFD scores for a single chunk with indices from 0 to 10,000
python data_selection/calculate_ifd.py \
--data_path ${data_path} \
--question_column ${question_column_name} \
--answer_column ${answer_column_name} \
--save_dir ${save_dir} \
--model_path ${model_path} \
--gpu ${gpu_id} \
--index 0 \
--nums 10000
```

After processing all chunks, merge the results using the following command:

```bash
# Merge all processed chunks
python data_selection/calculate_ifd.py \
--save_dir ${save_dir} \
--num_split ${num} \
--merge
```

#### 2. Get embedding using [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) model

```bash
python data_selection/get_embedding.py \
--data_path ${data_path} \
--save_dir ${save_dir} \
--batch_size 512 \
--model_path ${embedding_model_path} \
--gpu ${gpu_id}
```

#### 3. Sample data

```bash
python data_selection/select_data.py \
--data_path ${data_path} \
--npy_path ${npy_path} \
--save_path ${save_path} \
--ratio 0.4
```

### Training

**Parameters:**
- ***model_path***: Path to the pretrained models.
- ***data_path***: Path to the json training data.
- ***model_type***: The type of model being used. Supported values are `codellama` and `deepseek`.
- ***pad_mode***: The padding mode to use. Supported values are `dynamic_pad` and `dynamic_pack`.
- ***output_dir***: Path to save training files.

#### 1. Fully Sharded Data Parallel (FSDP)

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

accelerate launch --config_file "configs/fsdp_config.yaml"  train.py \
--model_name_or_path ${model_path} \
--use_fast_tokenizer True \
--use_flash_attn True \
--data_path  ${data_path} \
--model_type ${model_type} \
--max_seq_len 2048 \
--pad_mode ${pad_mode} \
--output_dir ${output_dir} \
--overwrite_output_dir True \
--gradient_checkpointing True \
--num_train_epochs 3 \
--per_device_train_batch_size 16 \
--gradient_accumulation_steps 4 \
--learning_rate 5e-5 \
--lr_scheduler_type "cosine" \
--warmup_steps 15 \
--save_strategy "no" \
--bf16 True \
--tf32 True \
--logging_strategy "steps" \
--logging_steps 50
```

#### 2. Parameter Efficient FineTuning (PEFT)

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# LoRA
torchrun --nproc_per_node 8 train.py \
--model_name_or_path ${model_path} \
--use_fast_tokenizer True \
--use_flash_attn True \
--use_peft_lora True \
--data_path  ${data_path} \
--model_type ${model_type} \
--max_seq_len 2048 \
--pad_mode ${pad_mode} \
--output_dir ${output_dir} \
--overwrite_output_dir True \
--gradient_checkpointing True \
--num_train_epochs 3 \
--per_device_train_batch_size 16 \
--gradient_accumulation_steps 4 \
--learning_rate 5e-5 \
--lr_scheduler_type "cosine" \
--warmup_steps 15 \
--save_strategy "no" \
--bf16 True \
--tf32 True \
--logging_strategy "steps" \
--logging_steps 50

# QLoRA
torchrun --nproc_per_node 8 train.py \
--model_name_or_path ${model_path} \
--use_fast_tokenizer True \
--use_flash_attn True \
--use_peft_lora True \
--use_4bit_quantization \
--data_path  ${data_path} \
--model_type ${model_type} \
--max_seq_len 2048 \
--pad_mode ${pad_mode} \
--output_dir ${output_dir} \
--overwrite_output_dir True \
--gradient_checkpointing True \
--num_train_epochs 3 \
--per_device_train_batch_size 16 \
--gradient_accumulation_steps 4 \
--learning_rate 5e-5 \
--lr_scheduler_type "cosine" \
--warmup_steps 15 \
--save_strategy "no" \
--bf16 True \
--tf32 True \
--logging_strategy "steps" \
--logging_steps 50
```

### Evaluation

This section details the evaluation of the model's performance on the HumanEval(+) and MBPP(+) metrics.

#### 1. Generate samples

In this step, we use the model to generate code samples using greedy decoding. This involves feeding the model with input prompts and collecting the output generated by the model.

```bash
# Base Model
python benchmark/generate.py \
--dataset [humaneval|mbpp] \
--model_path ${model_path} \
--samples_dir ${samples_dir} \
--gpu ${gpu_id} \
--max_new_tokens 1024

# Instruction Model
python benchmark/generate.py \
--dataset [humaneval|mbpp] \
--model_path ${model_path} \
--samples_dir ${samples_dir} \
--gpu ${gpu_id} \
--max_new_tokens 1024 \
--instruct
```

#### 2. Sanitize samples

After generating the code samples, the next step is to sanitize them. This involves cleaning up the generated code to ensure it is in a valid and executable format. Sanitization helps to remove any potential errors or inconsistencies that might affect the evaluation results.

```bash
python benchmark/sanitize.py --sample_file ${sample_file}
```

#### 3. Evaluate performance

The final step is to evaluate the model's performance based on the sanitized samples.

```bash
python benchmark/evaluate.py --dataset [humaneval|mbpp] --samples ${sample_file}
```

This command will compute the metrics and provide a detailed report on the model's performance.

## Citation

If you find this work helpful, please cite our paper as follows:

```bibtex

```
