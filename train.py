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



def collate_fn(data):
    images = torch.stack([torchvision.transforms.ToTensor()(dt[0]) for dt in data])
    images = images + torch.randn_like(images) * 0.1
    images = images + images.amin(dim=(-1, -2), keepdim=True)
    images = images / images.amax(dim=(-1, -2), keepdim=True)
    # print(images.shape)
    eps = 1e-6
    images = (images - images.mean(dim=(-1,-2), keepdim=True))/(images.std(dim=(-1,-2), keepdim=True) + eps)
    labels = torch.tensor([dt[1] for dt in data])
    
    return {
        "images": images.float(),
        "labels": labels,
    }
    
class ConvModel(nn.Module):
    def __init__(self, kernel_1=7, kernel_2=3, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=kernel_1),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=kernel_2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier_head = nn.Linear(20, num_classes)
        
    def forward(self, x):
        x = self.encoder(x).squeeze()
        x = self.classifier_head(x)
        return x
    
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
        masked_out = inputs * (1-sal_maps_normed)

        B = inputs.shape[0]
        fig, axs = plt.subplots(nrows=B, ncols=3, figsize=(12, B * 3))
        
        inputs = inputs.cpu()
        sal_maps_normed = sal_maps_normed.cpu()
        masked_out = masked_out.cpu()

        for i in range(B):
            cax = axs[i, 0].imshow(inputs[i].squeeze(), cmap='gray')
            axs[i, 0].set_title('Original Image')
            axs[i, 0].axis('off')
            fig.colorbar(cax, ax=axs[i, 0])
            
            cax = axs[i, 1].imshow(sal_maps_normed[i].squeeze(), cmap='gray')
            axs[i, 1].set_title('Normalized Saliency Map')
            axs[i, 1].axis('off')
            fig.colorbar(cax, ax=axs[i, 1])
            
            cax = axs[i, 2].imshow(masked_out[i].squeeze(), cmap='gray')
            axs[i, 2].set_title('Masked Out Image')
            axs[i, 2].axis('off')
            fig.colorbar(cax, ax=axs[i, 2])

        plt.tight_layout()
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
        for batch in test_loader:
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
        for batch in train_loader:
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
        
        train_faithfulness_mean = np.mean(faithfulness_per_sample)
        train_faithfulness_std = np.std(faithfulness_per_sample)
        ce_loss = total_ce_loss / total
        tv_loss = total_tv_loss / total
        loss = total_loss / total
        accuracy = total_correct / total
        
        metrics = {"train": {
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
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()
        
    train_mnist_data = torchvision.datasets.MNIST(root = args.data_dir, download = True)
    print(f"Train Dataset length: {len(train_mnist_data)}")

    test_mnist_data = torchvision.datasets.MNIST(root = args.data_dir, download = True, train=False)
    print(f"Test Dataset length: {len(test_mnist_data)}")


    train_loader = torch.utils.data.DataLoader(train_mnist_data,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            collate_fn=collate_fn,
                                            num_workers=1)

    test_loader = torch.utils.data.DataLoader(test_mnist_data,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            collate_fn=collate_fn,
                                            num_workers=1)


    sample_loader = torch.utils.data.DataLoader(test_mnist_data,
                                                batch_size=args.num_samples,
                                                shuffle=False,
                                                collate_fn=collate_fn,
                                                num_workers=1)
    
    
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    sample_data = next(iter(sample_loader))


    device = torch.device("cuda:1" if torch.cuda.is_available else "cpu")
    print(device)
    model = ConvModel()  
    model = model.to(device) 
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    lambda_tv = args.lambda_tv
    lambda_faithfulness = args.lambda_faithfulness
    saliency_method = args.saliency_method
    exp_name = f"mnist_tv_{lambda_tv}_ff_{lambda_faithfulness}_sal_{args.saliency_method}"

    wandb.init(
        project="saliency_map",
        name=f"exp_{exp_name}",
        settings=wandb.Settings(start_method="fork"),
        config=args)

    # Train the model
    train(model, train_loader, test_loader, optimizer, num_epochs=args.num_epochs)
            