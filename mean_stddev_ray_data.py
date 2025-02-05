import inspect
import json
from pathlib import Path
import shutil
from typing import Dict

import numpy as np
import ray

# TODO: will be public in the next release:
# https://github.com/ray-project/ray/commit/7369d797ce560f6d7254e4a55fd47eaff995df4e
from ray.data._internal.aggregate import Mean
from torch.utils.data import Dataset
from torchvision import datasets
from tqdm import tqdm

ray.init()

DATASET_NAMES = [
    "CelebA", "CIFAR10", "CIFAR100", "Country211", "DTD",
    "EMNIST", "EuroSAT", "FakeData", "FashionMNIST", "FER2013", "FGVCAircraft",
    "Flickr8k", "Flickr30k", "Flowers102", "Food101", "GTSRB", "INaturalist", "ImageNet",
    "Imagenette", "KMNIST", "LFWPeople", "LSUN", "MNIST", "Omniglot", "OxfordIIITPet",
    "Places365", "PCAM", "QMNIST", "RenderedSST2", "SEMEION", "SBU", "StanfordCars",
    "STL10", "SUN397", "SVHN", "USPS", "CocoDetection", "Cityscapes", "Kitti",
    "SBDataset", "VOCSegmentation", "VOCDetection", "WIDERFace"
]
DATA_ROOT = "~/tmp_data"
OUTPUT_PATH = "dataset_stats.json"


def load_existing_results(json_path) -> dict:
    """Load previously computed results from JSON, if any."""
    json_path = Path(json_path)
    if json_path.exists():
        return json.loads(json_path.read_text())
    return {}


def save_result(json_path, results):
    """Save results to JSON, appending to existing results."""
    existing = load_existing_results(json_path)
    existing[results["dataset"]] = {
        "means": results["means"],
        "stddevs": results["stddevs"],
        "means_normalized": results["means_normalized"],
        "stddevs_normalized": results["stddevs_normalized"],
    }
    with open(json_path, "w") as f:
        json.dump(existing, f, indent=2)


def compute_mean_stddev_with_ray(dataset: Dataset):
    """
    Use Ray Data to compute per-channel mean and stddev for an iterable of (image, label).
    The iterable is expected to yield raw samples (e.g. from a Torch dataset).

    :param data_iter: A PyTorch dataset
    :return: (mean, stddev) for each channel. These are NumPy arrays of shape (num_channels,).
    """
    # Convert PyTorch dataset to Ray Dataset
    ds = ray.data.from_items(dataset)

    def extract_and_process_image(row: Dict) -> Dict[str, np.ndarray]:
        """Extract and convert image to numpy array."""
        return {"image": np.array(row["item"][0])}

    ds = ds.map(extract_and_process_image)

    def compute_channel_stats(row: Dict) -> Dict[str, float]:
        """Compute per-channel statistics for a single image."""
        img = row["image"]
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        elif img.ndim == 3:
            # Move channels to first dimension if needed
            channel_dim = np.argmin(img.shape)
            if channel_dim != 0:
                img = np.moveaxis(img, channel_dim, 0)

        num_channels = img.shape[0]
        if num_channels not in [1, 2, 3]:
            raise ValueError(f"Expected 1-3 channels, got {num_channels}")

        mean = img.mean(axis=(1, 2))
        stddev = img.std(axis=(1, 2))

        return {f"mean_{i}": mean[i] for i in range(num_channels)} | {
            f"stddev_{i}": stddev[i] for i in range(num_channels)
        }

    ds = ds.map(compute_channel_stats)

    # Count channels from first sample
    num_channels = len([k for k in ds.take(1)[0].keys() if k.startswith("mean_")])

    # Aggregate statistics across dataset
    results = ds.aggregate(
        *[Mean(f"mean_{i}", alias_name=f"mean_{i}") for i in range(num_channels)],
        *[Mean(f"stddev_{i}", alias_name=f"stddev_{i}") for i in range(num_channels)],
    )

    print(f"raw results: {results}")

    # Process results
    means = [results[f"mean_{i}"] for i in range(num_channels)]
    stddevs = [results[f"stddev_{i}"] for i in range(num_channels)]

    return {
        "means": means,
        "stddevs": stddevs,
        "means_normalized": [m / 255.0 for m in means],
        "stddevs_normalized": [s / 255.0 for s in stddevs],
    }


def compute_all_dataset_stats() -> None:
    """Compute mean and standard deviation for all supported torchvision datasets."""
    existing_results = load_existing_results(OUTPUT_PATH)

    for name in tqdm(DATASET_NAMES, desc="Processing datasets"):
        if name in existing_results:
            print(f"Skipping {name} - already processed: {existing_results[name]}")
            continue

        try:
            dataset_cls = getattr(datasets, name)
            print(f"\nComputing statistics for {name}...")

            # The torvision Datasets API is a mess.
            # Here's an attempt to handle different dataset initialization patterns
            dataset_params = inspect.signature(dataset_cls).parameters
            kwargs = {}
            if "root" in dataset_params:
                kwargs["root"] = DATA_ROOT
            if "train" in dataset_params:
                kwargs["train"] = True
            if "split" in dataset_params:
                kwargs["split"] = "train"
            if "download" in dataset_params:
                kwargs["download"] = True
            
                
            dataset = dataset_cls(**kwargs)

            results = compute_mean_stddev_with_ray(dataset)
            results["dataset"] = name
            save_result(OUTPUT_PATH, results)
            print(f"Results for {name}: {results}")

        except Exception as e:
            print(f"Error processing {name}: {e}")
            continue
        finally:
            # Delete the dataset from disk to avoid filling up the disk
            shutil.rmtree(DATA_ROOT, ignore_errors=True)


if __name__ == "__main__":
    compute_all_dataset_stats()

    # After everything finishes, you can see final results:
    final_results = load_existing_results(OUTPUT_PATH)

    # print the results as a single copy-pastable markdown table
    print("\n\nFinal results:")
    print("| Dataset | Mean (unnormalized) | StdDev (unnormalized) | Mean (normalized) | StdDev (normalized) |")
    print("|---------|---------------------|----------------------|------------------|---------------------|")
    for dataset, results in final_results.items():
        means = str([round(x, 2) for x in results['means']]).replace('[', '(').replace(']', ')')
        stddevs = str([round(x, 2) for x in results['stddevs']]).replace('[', '(').replace(']', ')')
        means_norm = str([round(x, 5) for x in results['means_normalized']]).replace('[', '(').replace(']', ')')
        stddevs_norm = str([round(x, 5) for x in results['stddevs_normalized']]).replace('[', '(').replace(']', ')')
        print(f"| {dataset} | `{means}` | `{stddevs}` | `{means_norm}` | `{stddevs_norm}` |")
