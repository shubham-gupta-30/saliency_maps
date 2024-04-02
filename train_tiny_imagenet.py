import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms


import torchvision
import torch 
import torch.nn as nn

import matplotlib.pyplot as plt
import wandb

import torch.optim as optim 
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from captum.attr import IntegratedGradients

def prepare_model(num_classes=2, device="cpu", model_name="resnet101", unfreeze=False):
    if model_name == "resnet101":
        model = models.resnet101()
    elif model_name == "resnet18":
        model = models.resnet18()
    elif model_name == "resnet34":
        model = models.resnet34()
    else:
        model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model.to(device)

def collate_fn(data):
    images = torch.stack([dt[0] for dt in data])
    labels = torch.tensor([dt[1] for dt in data])
    return {
        "images": images.float(),
        "labels": labels,
    }
    
def get_ig_sal_maps(model, inputs, labels):
    interpretation_method = IntegratedGradients(model)
    attr_raw = interpretation_method.attribute(inputs, target=labels, n_steps=5)
    return attr_raw

def get_grad_times_input_sal_map(model, inputs, labels):
    inputs.requires_grad_(True)
    logits = model(inputs)
    loss = F.cross_entropy(logits, labels)
    # loss.backward(retain_graph=True)
    grads = torch.autograd.grad(loss, inputs, create_graph=True)[0]
    # sal_maps = inputs * inputs.grad
    sal_maps = inputs * grads
    return sal_maps
    

def get_sal_map(model, inputs, labels):
    with torch.enable_grad():
        if saliency_method == "inp_grad":
            sal_maps = get_grad_times_input_sal_map(model, inputs, labels)
        else:
            sal_maps = get_ig_sal_maps(model, inputs, labels)
    return sal_maps

def get_tv_loss(sal_maps):
    if len(sal_maps.shape) == 3:
        sal_maps = sal_maps.unsqueeze(1)
    diff_i = torch.abs(sal_maps[:, :, :-1, :] - sal_maps[:, :, 1:, :])
    diff_j = torch.abs(sal_maps[:, :, :, :-1] - sal_maps[:, :, :, 1:])
    
    # Pad diff_j to match the height of diff_i
    diff_j_padded = F.pad(diff_j, (0, 1, 0, 0), "constant", 0)  # Pad bottom
    # Similarly, pad diff_i to match the width of diff_j if needed
    diff_i_padded = F.pad(diff_i, (0, 0, 0, 1), "constant", 0)  # Pad right
    
    # print(diff_j_padded.shape, diff_i_padded.shape)
    
    # Sum of differences
    tvs = diff_i_padded + diff_j_padded
    tvs = tvs.mean(dim=(1, 2, 3))
    return tvs.mean()


def plot(model, data, device):
    model.eval()
    with torch.no_grad():
        inputs = data["images"].to(device)
        labels = data["labels"].to(device)
        sal_maps = get_sal_map(model, inputs, labels)
        sal_maps_normed = normalize_sal_map(sal_maps)
        # masked_out = inputs * (1-sal_maps_normed)

        B = inputs.shape[0]
        nrows = B
        ncols = 2
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, nrows * 4))
        
        inputs = (inputs - inputs.amin(dim=(1, 2, 3), keepdims=True))/(inputs.amax(dim=(1, 2, 3), keepdims=True) - inputs.amin(dim=(1, 2, 3), keepdims=True))
        # masked_out = (masked_out - masked_out.amin(dim=(1, 2, 3), keepdims=True))/(masked_out.amax(dim=(1, 2, 3), keepdims=True) - masked_out.amin(dim=(1, 2, 3), keepdims=True))

        inputs = inputs.cpu().numpy().transpose((0, 2, 3, 1))
        sal_maps_normed = sal_maps_normed.mean(1, keepdims=True).cpu().numpy().transpose((0, 2, 3, 1))
        # masked_out = masked_out.cpu().numpy().transpose((0, 2, 3, 1))

        for i in range(B):
            axs[i, 0].imshow(inputs[i].squeeze(), aspect='equal')
            axs[i, 0].axis('off')
            
            axs[i, 1].imshow(sal_maps_normed[i].squeeze(), cmap="Greys", aspect='equal')
            axs[i, 1].axis('off')

        plt.tight_layout()
        plt.savefig("saliency_maps.png")
    model.train()
    return fig
    
    
    


def compute_faithfulness(predictions, predictions_masked):
    # get the prediction indices
    pred_cl = predictions.argmax(dim=1, keepdim=True)

    # get the corresponding output probabilities
    predictions_selected = torch.gather(predictions, dim=1, index=pred_cl)
    predictions_masked_selected = torch.gather(
        predictions_masked, dim=1, index=pred_cl
    )

    faithfulness = (
        predictions_selected - predictions_masked_selected
    ).squeeze()

    return faithfulness

def normalize_sal_map(M):
    M = M/M.abs().amax(dim=(-1, -2), keepdim=True)
    M = 1 + M
    M = M / 2
    return M

def get_faithfulness_metric(model, inputs, sal_maps):
    predictions = F.softmax(model(inputs.float()), dim=-1)
    sal_maps = normalize_sal_map(sal_maps)
    masked_out = inputs * (1 - sal_maps)
    predictions_masked = F.softmax(model(masked_out.float()), dim=-1)
    faithfulness = compute_faithfulness(predictions, predictions_masked)
    return faithfulness
    

def model_forward(model, inputs, labels):
    outputs = model(inputs)
    sal_maps = get_sal_map(model, inputs, labels)
    sal_maps_normed = normalize_sal_map(sal_maps)
    
    batch_num_correct = get_num_correct(out=outputs, labels=labels)
    batch_ce_loss = F.cross_entropy(outputs, labels)
    batch_len = inputs.shape[0]
    batch_tv_loss = get_tv_loss(sal_maps_normed)
    batch_faithfulness = get_faithfulness_metric(model, inputs, sal_maps)
    
    return batch_num_correct, batch_ce_loss, batch_tv_loss, batch_len, batch_faithfulness
    

def get_num_correct(out, labels):
    _, predicted = torch.max(out, -1)
    num_correct = (predicted == labels).sum().item()
    return num_correct



def test(model, test_loader):
    model.eval()
    total_correct = 0
    total = 0
    total_ce_loss = 0
    total_tv_loss = 0
    faithfulness_per_sample = []
    with torch.no_grad():
        for _, batch in enumerate(tqdm(test_loader)):
            inputs, labels = batch['images'].to(device), batch['labels'].to(device)
            batch_num_correct, batch_ce_loss, batch_tv_loss, batch_len, batch_faithfulness = model_forward(model, inputs, labels)
            total += batch_len
            total_correct += batch_num_correct
            total_ce_loss += batch_ce_loss.item() * batch_len
            total_tv_loss += batch_tv_loss.item() * batch_len
            faithfulness_per_sample += batch_faithfulness.tolist()
            
        test_faithfulness_mean = np.mean(faithfulness_per_sample)
        test_faithfulness_std = np.std(faithfulness_per_sample)
        ce_loss = total_ce_loss / total
        tv_loss = total_tv_loss / total
        loss = ce_loss + lambda_tv * tv_loss + lambda_faithfulness * test_faithfulness_mean
        accuracy = total_correct / total
    model.train()
    
    test_metrics = {"test": {
                        "acc": accuracy,
                        "loss": loss,
                        "ce_loss": ce_loss,
                        "tv_loss": tv_loss,
                        "faithfulness": {
                            "mean": test_faithfulness_mean,
                            "std": test_faithfulness_std
                        }}}
    
    print(f"test_faithfulness_mean: {test_faithfulness_mean}, ce_loss: {ce_loss}, tv_loss: {tv_loss}, loss: {loss}, accuracy: {accuracy}")
    return test_metrics
    

def train(model, train_loader, test_loader, optimizer, num_epochs=10):
    for epoch in range(num_epochs):    
        model.train()
        total_loss = 0
        total = 0
        total_ce_loss = 0
        total_tv_loss = 0
        total_correct = 0
        faithfulness_per_sample = []
        for i, batch in enumerate(tqdm(train_loader)):
            
            if (epoch * len(train_loader) + i  ) % args.test_every == 0:
                train_faithfulness_mean = np.mean(faithfulness_per_sample) if len(faithfulness_per_sample) > 0 else 0
                train_faithfulness_std = np.std(faithfulness_per_sample) if len(faithfulness_per_sample) > 0 else 0
                ce_loss = total_ce_loss / max(total, 1)
                tv_loss = total_tv_loss / max(total, 1)
                loss = total_loss / max(total, 1)
                accuracy = total_correct / max(total, 1)
                
                metrics = {"epoch": epoch,
                           "train": {
                                "acc": accuracy,
                                "loss": loss,
                                "ce_loss": ce_loss,
                                "tv_loss": tv_loss,
                                "faithfulness": {
                                    "mean": train_faithfulness_mean,
                                    "std": train_faithfulness_std
                                }}}
                
                metrics.update(test(model, test_loader))
                print(f'Epoch {epoch+1}, train_faithfulness_mean: {train_faithfulness_mean}, ce_loss: {ce_loss}, tv_loss: {tv_loss}, loss: {loss}, accuracy: {accuracy}')
                fig = plot(model, sample_data, device)
                
                metrics.update({"samples": wandb.Image(fig)})
                
                wandb.log(metrics)
                total_loss = 0
                total = 0
                total_ce_loss = 0
                total_tv_loss = 0
                total_correct = 0
                faithfulness_per_sample = []
            
            optimizer.zero_grad()
            inputs, labels = batch['images'].to(device), batch['labels'].to(device)
            batch_num_correct, batch_ce_loss, batch_tv_loss, batch_len, batch_faithfulness = model_forward(model, inputs, labels)
            loss = batch_ce_loss + lambda_tv * batch_tv_loss + lambda_faithfulness * batch_faithfulness.mean()
            loss.backward()
            total += batch_len
            total_correct += batch_num_correct
            optimizer.step()
            total_loss += loss.item() * batch_len
            total_ce_loss += batch_ce_loss.item() * batch_len
            total_tv_loss += batch_tv_loss.item() * batch_len
            faithfulness_per_sample += batch_faithfulness.tolist()
            print(f'Epoch {epoch+1}"{i}, train_faithfulness_mean: {np.mean(faithfulness_per_sample)}, ce_loss: {total_ce_loss / total}, tv_loss: {total_tv_loss / total}, loss: {loss}, accuracy: {total_correct / total}')
        
                
        


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on MNIST")

    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training and testing")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs for training")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--data_dir", type=str, default="/home-local2/shgup1.extra.nobkp/mnist", help="Directory for storing MNIST data")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--lambda_tv", type=float, default=0, help="Lambda value for TV")
    parser.add_argument("--lambda_faithfulness", type=float, default=0, help="Lambda value for faithfulness")
    parser.add_argument("--saliency_method", type=str, default="inp_grad")
    parser.add_argument("--test_every", type=int, default=50)
    parser.add_argument("--test_iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device_id", type=int, default=0)

    args = parser.parse_args()
    
    num_workers = {"train": 2, "val": 0, "test": 0}
    data_dir = "/home-local2/shgup1.extra.nobkp/tiny-imagenet/tiny-imagenet-200/"

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]
        ),
    }
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ["train", "test"]
    }
    
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    
    sample_indices = torch.randperm(len(image_datasets["test"]))[:args.num_samples]
    samples_subset = data.Subset(image_datasets["test"], sample_indices)
    
    num_test_samples = args.test_iters * args.batch_size
    test_indices = torch.randperm(len(image_datasets["test"]))[:num_test_samples]
    test_subset = data.Subset(image_datasets["test"], test_indices)

    train_loader = data.DataLoader(image_datasets["train"], batch_size=args.batch_size, shuffle = True, num_workers=num_workers["train"], collate_fn=collate_fn, drop_last=True)
    test_loader = data.DataLoader(test_subset, batch_size=args.batch_size, shuffle = False, num_workers=num_workers["test"], collate_fn=collate_fn)
    sample_loader = data.DataLoader(samples_subset, batch_size=args.num_samples, shuffle = False, num_workers=num_workers["test"], collate_fn=collate_fn)
    sample_data = next(iter(sample_loader))


    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available else "cpu")
    print(device)
    model = prepare_model(num_classes=200, device=device, model_name="resnet18")  
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    lambda_tv = args.lambda_tv
    lambda_faithfulness = args.lambda_faithfulness
    saliency_method = args.saliency_method
    exp_name = f"tiny_imgnt_tv_{lambda_tv}_ff_{lambda_faithfulness}_sal_{args.saliency_method}"

    wandb.init(
        project="saliency_map",
        name=f"exp_{exp_name}",
        settings=wandb.Settings(start_method="fork"),
        config=args)

    # Train the model
    train(model, train_loader, test_loader, optimizer, num_epochs=args.num_epochs)
            