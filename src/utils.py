import os
import random
from datetime import timedelta, datetime
import json

import numpy as np
import torch
import torchvision
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

import pandas as pd

from torchvision.transforms import Compose, ToTensor, Normalize, Lambda

from src import ff_mnist, ff_model
import wandb


def parse_args(opt):
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

    print(OmegaConf.to_yaml(opt))
    return opt

def get_input_layer_size(opt):
    if opt.input.dataset == "mnist":
        return 784
    elif opt.input.dataset == "fashionmnist":
        return 784
    elif opt.input.dataset == "cifar10":
        return 3082  # 10 (one-hot label) + 3072 (32*32*3 image)
    elif opt.input.dataset == "stl10":
        return 3082  # 10 (one-hot label) + 3072 (32*32*3 downsampled image, same as CIFAR-10)
    else:
        raise ValueError("Unknown dataset.")

def get_model_and_optimizer(opt):
    model = ff_model.FF_model(opt)
    if "cuda" in opt.device:
        model = model.cuda()
    print(model, "\n")

    # Create optimizer with different hyper-parameters for the main model
    # and the downstream classification model.
    main_model_params = [
        p
        for p in model.parameters()
        if all(p is not x for x in model.classification_loss.parameters())
    ]
    optimizer = torch.optim.SGD(
        [
            {
                "params": main_model_params,
                "lr": opt.training.learning_rate,
                "weight_decay": opt.training.weight_decay,
                "momentum": opt.training.momentum,
            },
            {
                "params": model.classification_loss.parameters(),
                "lr": opt.training.downstream_learning_rate,
                "weight_decay": opt.training.downstream_weight_decay,
                "momentum": opt.training.momentum,
            },
        ]
    )
    return model, optimizer
# 784, 2000, 2000, 2000 # main params
# 6000, 10 # classification_loss params

def get_data(opt, partition):
    # dataset = ff_mnist.FF_MNIST(opt, partition)
    if opt.input.dataset == "mnist":
        dataset = ff_mnist.FF_MNIST(opt, partition, num_classes=10)
    elif opt.input.dataset == "fashionmnist":
        dataset = ff_mnist.FF_FashionMNIST(opt, partition, num_classes=10)
    elif opt.input.dataset == "cifar10":
        dataset = ff_mnist.FF_CIFAR10(opt, partition, num_classes=10)
    elif opt.input.dataset == "stl10":
        dataset = ff_mnist.FF_STL10(opt, partition, num_classes=10)
    else:
        raise ValueError("Unknown dataset.")

    # Improve reproducibility in dataloader.
    g = torch.Generator()
    g.manual_seed(opt.seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.input.batch_size,
        drop_last=True,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=1,
        persistent_workers=True,
    )


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_senti_partition(opt, partition):
    # load reviews data
    # print(os.path.join(get_original_cwd(), opt.input.training_path))
    train_df = pd.read_csv(os.path.join(get_original_cwd(), opt.input.training_path), names=["filename", "split", "labels", "features"])
    test_df = pd.read_csv(os.path.join(get_original_cwd(), opt.input.test_path), names=["filename", "split", "labels", "features"])
    # train_df = pd.read_csv('reviews_train.csv',names=["filename", "split", "labels", "features"])
    # test_df = pd.read_csv('reviews_test.csv',names=["filename", "split", "labels", "features"])
    train_df = train_df.drop(columns=["filename", "split"])
    test_df = test_df.drop(columns=["filename", "split"])
    train_df['labels'] = train_df['labels'].replace({'pos': 1, 'neg': 0})
    test_df['labels'] = test_df['labels'].replace({'pos': 1, 'neg': 0})

    train_labels = torch.tensor(train_df['labels'].values, dtype=torch.long)
    test_labels = torch.tensor(test_df['labels'].values, dtype=torch.long)

    train_data = train_df['features'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ', dtype=np.float32))
    train_data = torch.stack([torch.tensor(x) for x in train_data])

    test_data = test_df['features'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ', dtype=np.float32))
    test_data = torch.stack([torch.tensor(x) for x in test_data])

    final_train_data = torch.hstack((train_data,torch.unsqueeze(train_labels, 1)))
    final_test_data = torch.hstack((test_data,torch.unsqueeze(test_labels, 1)))

    if partition in ["train"]:
        return train_data, train_labels
    else:
        return test_data, test_labels

def get_CIFAR10_partition(opt, partition):
    transform = Compose(
        [
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    if partition in ["train"]:
        cifar = torchvision.datasets.CIFAR10(
            os.path.join(get_original_cwd(), opt.input.path),
            train=True,
            download=True,
            transform=transform,
        )
    elif partition in ["val", "test"]:
        cifar = torchvision.datasets.CIFAR10(
            os.path.join(get_original_cwd(), opt.input.path),
            train=False,
            download=True,
            transform=transform,
        )
    else:
        raise NotImplementedError

    return cifar

def get_MNIST_partition(opt, partition):
    transform = Compose(
        [
            ToTensor(),
        ]
    )
    if partition in ["train"]:
        mnist = torchvision.datasets.MNIST(
            os.path.join(get_original_cwd(), opt.input.path),
            train=True,
            download=True,
            transform=transform,
        )
    elif partition in ["val", "test"]:
        mnist = torchvision.datasets.MNIST(
            os.path.join(get_original_cwd(), opt.input.path),
            train=False,
            download=True,
            transform=transform,
        )
    else:
        raise NotImplementedError

    return mnist


def get_FashionMNIST_partition(opt, partition):
    transform = Compose(
        [
            ToTensor(),
        ]
    )
    if partition in ["train"]:
        fashion_mnist = torchvision.datasets.FashionMNIST(
            os.path.join(get_original_cwd(), opt.input.path),
            train=True,
            download=True,
            transform=transform,
        )
    elif partition in ["val", "test"]:
        fashion_mnist = torchvision.datasets.FashionMNIST(
            os.path.join(get_original_cwd(), opt.input.path),
            train=False,
            download=True,
            transform=transform,
        )
    else:
        raise NotImplementedError

    return fashion_mnist


def get_STL10_partition(opt, partition):
    from torchvision.transforms import Resize
    # Downsample from 96x96 to 32x32 to match CIFAR-10 dimensionality
    # This makes the label signal proportionally as strong as CIFAR-10
    transform = Compose(
        [
            ToTensor(),
            Resize((32, 32)),  # Downsample to 32x32 (3072 pixels, same as CIFAR-10)
            Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
        ]
    )
    if partition in ["train"]:
        stl10 = torchvision.datasets.STL10(
            os.path.join(get_original_cwd(), opt.input.path),
            split='train',
            download=True,
            transform=transform,
        )
    elif partition in ["val", "test"]:
        stl10 = torchvision.datasets.STL10(
            os.path.join(get_original_cwd(), opt.input.path),
            split='test',
            download=True,
            transform=transform,
        )
    else:
        raise NotImplementedError

    return stl10


def dict_to_cuda(dict):
    for key, value in dict.items():
        dict[key] = value.cuda(non_blocking=True)
    return dict


def preprocess_inputs(opt, inputs, labels):
    if "cuda" in opt.device:
        inputs = dict_to_cuda(inputs)
        labels = dict_to_cuda(labels)
    return inputs, labels

# cools down after the first half of the epochs
def get_linear_cooldown_lr(opt, epoch, lr):
    if epoch > (opt.training.epochs // 2):
        return lr * 2 * (1 + opt.training.epochs - epoch) / opt.training.epochs
    else:
        return lr


def update_learning_rate(optimizer, opt, epoch):
    optimizer.param_groups[0]["lr"] = get_linear_cooldown_lr(
        opt, epoch, opt.training.learning_rate
    )
    optimizer.param_groups[1]["lr"] = get_linear_cooldown_lr(
        opt, epoch, opt.training.downstream_learning_rate
    )
    return optimizer


def get_accuracy(opt, output, target):
    """Computes the accuracy."""
    with torch.no_grad():
        prediction = torch.argmax(output, dim=1)
        return (prediction == target).sum() / opt.input.batch_size


def print_results(partition, iteration_time, scalar_outputs, epoch=None):
    if epoch is not None:
        print(f"Epoch {epoch} \t", end="")

    print(
        f"{partition} \t \t"
        f"Time: {timedelta(seconds=iteration_time)} \t",
        end="",
    )
    if scalar_outputs is not None:
        for key, value in scalar_outputs.items():
            print(f"{key}: {value:.4f} \t", end="")
    print()
    partition_scalar_outputs = {}
    if scalar_outputs is not None:
        for key, value in scalar_outputs.items():
            partition_scalar_outputs[f"{partition}_{key}"] = value
    wandb.log(partition_scalar_outputs, step=epoch)

# create save_model function
def save_model(model):
    torch.save(model.state_dict(), f"{wandb.run.name}-model.pt")
    # log model to wandb
    wandb.save(f"{wandb.run.name}-model.pt")


def log_results(result_dict, scalar_outputs, num_steps):
    for key, value in scalar_outputs.items():
        if isinstance(value, float):
            result_dict[key] += value / num_steps
        else:
            result_dict[key] += value.item() / num_steps
    return result_dict


def save_experiment_results(opt, train_results, val_results, test_results, training_time, carbon_data=None):
    """
    Save experiment results in a structured JSON format.
    
    Args:
        opt: Configuration object
        train_results: Dictionary of training metrics
        val_results: Dictionary of validation metrics
        test_results: Dictionary of test metrics
        training_time: Total training time in seconds
        carbon_data: Dictionary containing carbon emission data
    """
    # Create results directory if it doesn't exist
    results_dir = os.path.join(get_original_cwd(), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine goodness function from config
    goodness_function = getattr(opt.model, 'goodness_function', 'sum_of_squares')
    
    # Create filename: dataset_goodnessfunction_timestamp.json
    filename = f"{opt.input.dataset}_{goodness_function}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)
    
    # Convert result values to native Python types
    def convert_to_native(obj):
        if isinstance(obj, torch.Tensor):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return obj
    
    # Prepare results structure
    results = {
        "experiment_info": {
            "dataset": opt.input.dataset,
            "goodness_function": goodness_function,
            "timestamp": timestamp,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "hyperparameters": {
            "seed": opt.seed,
            "device": opt.device,
            "batch_size": opt.input.batch_size,
            "epochs": opt.training.epochs,
            "learning_rate": opt.training.learning_rate,
            "weight_decay": opt.training.weight_decay,
            "momentum": opt.training.momentum,
            "downstream_learning_rate": opt.training.downstream_learning_rate,
            "downstream_weight_decay": opt.training.downstream_weight_decay,
            "hidden_dim": opt.model.hidden_dim,
            "num_layers": opt.model.num_layers,
            "peer_normalization": opt.model.peer_normalization
        },
        "training_time": {
            "total_seconds": training_time,
            "formatted": str(timedelta(seconds=training_time))
        },
        "metrics": {
            "train": {k: convert_to_native(v) for k, v in train_results.items()},
            "validation": {k: convert_to_native(v) for k, v in val_results.items()} if val_results else {},
            "test": {k: convert_to_native(v) for k, v in test_results.items()} if test_results else {}
        }
    }
    
    # Add carbon data if available
    if carbon_data:
        results["carbon_footprint"] = carbon_data
    
    # Save to JSON file
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {filepath}")
    print(f"{'='*60}\n")
    
    return filepath


def get_carbon_summary(emissions_file):
    """
    Extract carbon emission summary from codecarbon emissions file.
    
    Args:
        emissions_file: Path to the emissions CSV file
        
    Returns:
        Dictionary containing carbon emission metrics
    """
    if not os.path.exists(emissions_file):
        return None
    
    try:
        df = pd.read_csv(emissions_file)
        if len(df) == 0:
            return None
        
        # Get the last row (most recent emission data)
        last_row = df.iloc[-1]
        
        carbon_data = {
            "emissions_kg_co2": float(last_row.get('emissions', 0)),
            "emissions_g_co2": float(last_row.get('emissions', 0)) * 1000,
            "duration_seconds": float(last_row.get('duration', 0)),
            "energy_consumed_kwh": float(last_row.get('energy_consumed', 0)),
            "country": str(last_row.get('country_name', 'Unknown')),
            "region": str(last_row.get('region', 'Unknown')),
            "cpu_energy_kwh": float(last_row.get('cpu_energy', 0)),
            "gpu_energy_kwh": float(last_row.get('gpu_energy', 0)),
            "ram_energy_kwh": float(last_row.get('ram_energy', 0))
        }
        
        return carbon_data
    except Exception as e:
        print(f"Warning: Could not read carbon emissions data: {e}")
        return None


def save_epoch_metrics(opt, epoch_metrics, results_filename):
    """
    Save per-epoch training and validation metrics to a JSON file.
    
    Args:
        opt: Configuration object
        epoch_metrics: List of dictionaries containing per-epoch metrics
        results_filename: Path to the main results file (used to extract naming info)
    """
    # Create epoch_metrics subdirectory
    results_dir = os.path.join(get_original_cwd(), "results")
    epoch_metrics_dir = os.path.join(results_dir, "epoch_metrics")
    os.makedirs(epoch_metrics_dir, exist_ok=True)
    
    # Extract base filename from results_filename (without path and extension)
    base_filename = os.path.splitext(os.path.basename(results_filename))[0]
    
    # Create epoch metrics filename
    epoch_filename = f"{base_filename}_epochs.json"
    epoch_filepath = os.path.join(epoch_metrics_dir, epoch_filename)
    
    # Convert tensor values to native Python types
    def convert_to_native(obj):
        if isinstance(obj, torch.Tensor):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj
    
    # Convert all epoch metrics
    converted_metrics = convert_to_native(epoch_metrics)
    
    # Prepare epoch data structure
    epoch_data = {
        "experiment_info": {
            "dataset": opt.input.dataset,
            "goodness_function": getattr(opt.model, 'goodness_function', 'sum_of_squares'),
            "total_epochs": len(epoch_metrics)
        },
        "epochs": converted_metrics
    }
    
    # Save to JSON file
    with open(epoch_filepath, 'w') as f:
        json.dump(epoch_data, f, indent=4)
    
    print(f"Per-epoch metrics saved to: {epoch_filepath}")
