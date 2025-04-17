import argparse
import json
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

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
    parser.add_argument("--npy_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--n_clusters", type=int, default=10)
    parser.add_argument("--ratio", type=float, default=0.4)

    args = parser.parse_args()
    return args


def format_data(data):
    formated_data = []
    conv_id = 1
    for item in tqdm(data, desc="Formatting data"):
        conv = []
        conv.append({
            "role": "user",
            "content": item["question"]
        })
        conv.append({
            "role": "assistant",
            "content": item["answer"]
        })

        formated_data.append({
            "conversation_id": conv_id,
            "conversation": conv,
            "IFD": item["IFD"],
        })
        conv_id += 1
        del conv
    return formated_data


def main():
    args = get_args()
    print(args)

    # load data
    data = jload(args.data_path)
    embeddings = np.load(args.npy_path)
    assert len(data) == len(embeddings), "The lengths of JSON data and embeddings do not match."

    # use KMeans algorithm to cluster data
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    clusters = [[] for _ in range(args.n_clusters)]
    for index, label in enumerate(labels):
        clusters[label].append(data[index])

    # select data
    selected_data = []
    ratio = args.ratio
    for cluster in clusters:
        cluster = [x for x in cluster if x["IFD"] < 1.0]
        cluster.sort(key=lambda x: x["IFD"], reverse=True)
        n = int(len(cluster) * ratio)
        selected_data.extend(cluster[:n])

    selected_data_count = len(selected_data)
    selected_data_percentage = (selected_data_count / len(data)) * 100
    print(f"Number of raw data: {len(data)}")
    print(f"Number of selected data: {selected_data_count}")
    print(f"Percentage of selected data: {selected_data_percentage:.2f}%")

    # format data
    formatted_data = format_data(selected_data)
    jsave(args.save_path, formatted_data, ensure_ascii=False)
    print(f"selected data saved in {args.save_path}")


if __name__ == '__main__':
    main()
