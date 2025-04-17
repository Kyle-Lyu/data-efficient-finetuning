import argparse
import os
import math
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

def jload(data_path, mode="r"):
    try:
        with open(data_path, mode) as f:
            data = json.load(f)
    except:
        data = []
        with open(data_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
    return data

def jsave(save_path, data, mode="w", indent=4, ensure_ascii=True):
    with open(save_path, mode) as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--question_column", type=str, default=None)
    parser.add_argument("--answer_column", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--nums", type=int, default=10000)
    parser.add_argument("--num_split", type=int, default=None)
    parser.add_argument("--merge", action="store_true")

    args = parser.parse_args()
    return args
    

def calculate_ifd(tokenizer, model, question: str, answer: str, max_length: int):
    prompt = "Question:\n{question}\n\nAnswer:\n".format(question=question)
    end_index = len(tokenizer.encode(prompt, truncation=True, max_length=max_length))

    # perplexity of directly generating answer
    try:
        answer_ids = tokenizer.encode(answer, return_tensors="pt", truncation=True, max_length=max_length - end_index + 1).to(model.device)
        with torch.no_grad():
            outputs = model(answer_ids, labels=answer_ids.contiguous())
            loss = outputs.loss
        perplexity_answer = torch.exp(loss).to('cpu').item()
    except Exception as e:
        perplexity_answer = 0

    # perplexity of generating answer given question
    try:
        full_prompt = prompt + answer
        input_ids = tokenizer.encode(full_prompt, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
        labels = input_ids.clone()
        labels[0, :end_index] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
        perplexity_answer_condition = torch.exp(loss).to('cpu').item()
    except Exception as e:
        perplexity_answer_condition = 0

    # calculate IFD score
    try:
        ifd_score = perplexity_answer_condition / perplexity_answer
    except ZeroDivisionError:
        ifd_score = 0

    if math.isnan(ifd_score):
        ifd_score = 0

    return ifd_score


def merge_data(save_dir, num_split):
    data = []
    for i in range(num_split):
        file_path = os.path.join(save_dir, f"processed_data_{i}.json")
        data.extend(jload(file_path))

    save_file = os.path.join(save_dir, f"processed_data_all.json")
    jsave(save_file, data, ensure_ascii=False)

    print(f"length of all processed data is {len(data)}")
    print(f"all processed data saved in {save_file}")


def main():
    args = get_args()
    print(args)

    if args.merge:
        merge_data(args.save_dir, args.num_split)
        exit()

    # get model max sequence length
    config = AutoConfig.from_pretrained(args.model_path)
    if args.max_length is None:
        args.max_length = getattr(config, "max_position_embeddings", None)
    if args.max_length is None:
        raise ValueError("Please specify the maximum sample length, for example --max_length 2048")

    # load tokenizer and model
    device = f"cuda:{args.gpu}" if args.gpu else "auto"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=device,
        torch_dtype=torch.float16,
    )
    model.eval()

    # load data
    data = jload(args.data_path)
    index = args.index
    nums = args.nums
    data = data[nums * index : nums * (index + 1)]
    print(f"Index {index}: processing {len(data)} examples from index {index * nums}")

    processed_data = []
    for item in tqdm(data, desc=f"Index {index}"):
        question = item[args.question_column]
        answer = item[args.answer_column]

        if question and answer:
            # calculate IFD score
            ifd_score = calculate_ifd(tokenizer, model, question, answer, args.max_length)
            
            new_item = {
                "question": question,
                "answer": answer,
                "IFD": ifd_score,
            }
            processed_data.append(new_item)
    
    print(f"Index {index}: length of processed data is {len(processed_data)}")

    # save processed data
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    save_file = os.path.join(args.save_dir, f"processed_data_{args.index}.json")
    jsave(save_file, processed_data, ensure_ascii=False)

    print(f"Index {index}: processed data saved in {save_file}")
        

if __name__ == "__main__":
    main()
