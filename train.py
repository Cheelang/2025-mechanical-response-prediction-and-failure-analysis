import os
import time
import datetime
import torch
import pandas as pd
from torch.optim import AdamW
from my_dataset import get_Dataset
import transforms as T
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from src.swin.vision_transformer import SwinUnet
from src.models.unet import UNetRegressor
from src.models.fcn import FCNRegressor
class train_transform:
    def __init__(self, size):
        self.transforms = T.Compose([
            T.Resize(size=size),
            T.ToTensor(),
        ])
    def __call__(self, img):
        return self.transforms(img)
class eval_transform:
    def __init__(self, size):
        self.transforms = T.Compose([
            T.Resize(size),
            T.ToTensor(),
        ])
    def __call__(self, img):
        return self.transforms(img)
def create_model(args):
    if args.model == "swin":
        model = SwinUnet(in_chans=3, img_size=args.train_size, num_classes=1)
    elif args.model == "unet":
        model = UNetRegressor(in_chans=3, base_ch=32, depth=4, num_classes=1)
    elif args.model == "fcn":
        model = FCNRegressor(in_chans=3, base_ch=32, depth=3, num_classes=1)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    return model
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_dataset = get_Dataset(args.data_path, train=True, transforms=train_transform(size=args.train_size))
    val_dataset = get_Dataset(args.data_path, train=False, transforms=eval_transform(size=args.train_size))
    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=num_workers, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=num_workers, pin_memory=True)
    model = create_model(args)
    model.to(device)
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
    best_val_loss = float('inf')
    start_time = time.time()
    df = pd.DataFrame(columns=['train_loss', 'val_loss'])
    images, targets = next(iter(train_loader))
    images, targets = images.to(device), targets.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(images)
    print("==== Debug: Initial check ====")
    print("Targets:", targets[:10].detach().cpu().numpy())
    print("Outputs:", outputs[:10].detach().cpu().numpy())
    print("Targets mean/std/min/max:", 
          targets.mean().item(), targets.std().item(),
          targets.min().item(), targets.max().item())
    print("Outputs mean/std/min/max:", 
          outputs.mean().item(), outputs.std().item(),
          outputs.min().item(), outputs.max().item())
    print("Initial loss (MSE):", torch.nn.functional.mse_loss(outputs, targets).item())
    print("================================")
    for epoch in range(args.start_epoch, args.epochs):
        train_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, lr_scheduler=lr_scheduler, print_freq=args.print_freq)
        val_loss = evaluate(model, optimizer, val_loader, device)
        df.loc[epoch] = [train_loss, val_loss]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"save_weights/{args.model}/data4midstress_model.pth")
        if not os.path.exists('log/'):
            os.makedirs('log/')
        df.to_csv(f'log/{args.model}/log_data4midstress_{args.model}.csv', index=False)
    total_time = time.time() - start_time
    print("Training time: {}".format(str(datetime.timedelta(seconds=int(total_time)))))
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Training for scalar regression task")
    parser.add_argument("--model", default="swin")
    parser.add_argument("--data-path", default="./data2_mid")
    parser.add_argument("--device", default="cuda:1", help="Training device")
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--train_size", default=224, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float, dest='weight_decay')
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--print-freq', default=10, type=int, help='Log print frequency')
    return parser.parse_args()
if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(f"./save_weights/{args.model}"):
        os.makedirs(f"./save_weights/{args.model}")
    if not os.path.exists(f"./log/{args.model}"):
        os.makedirs(f"./log/{args.model}")
    main(args)