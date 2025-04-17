import argparse
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


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
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--gpu", type=str, default=None)

    args = parser.parse_args()
    return args


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_embedding(tokenizer, model, texts):
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model(**encoded_input)
    
    embeddings = mean_pooling(outputs, encoded_input['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.to('cpu').numpy()


def main():
    args = get_args()
    print(args)
    batch_size = args.batch_size

    # load tokenizer and model
    device = f"cuda:{args.gpu}" if args.gpu else "auto"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path, device_map=device,)

    # load data
    data = jload(args.data_path)
    print(f"lenght of data is {len(data)}")

    # process data
    all_embeddings = []
    with tqdm(total=len(data)) as pbar:
        for start_idx in range(0, len(data), batch_size):
            end_idx = min(start_idx + batch_size, len(data))
            batch_data = data[start_idx:end_idx]
            questions = [item["question"] for item in batch_data]

            batch_embeddings = get_embedding(tokenizer, model, questions)
            all_embeddings.extend(batch_embeddings)

            pbar.update((end_idx - start_idx))
    
    all_embeddings = np.vstack(all_embeddings)

    # save embeddings
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    npy_file = os.path.join(args.save_dir, "processed_embeddings.npy")
    np.save(npy_file, all_embeddings)
    print(f"embeddings saved in {npy_file}")


if __name__ == "__main__":
    main()
