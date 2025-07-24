import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# ──────────────────────────────────────────────
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from torch.utils.tensorboard import SummaryWriter
# ──────────────────────────────────────────────

from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

dir_img = Path('../USA_segmentation/resize/RGB_images')
dir_nrg = Path('../USA_segmentation/resize/NRG_images')
dir_mask = Path('../USA_segmentation/resize/masks')
dir_checkpoint = Path('./checkpoints/')


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    # try:
    #     dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    # except (AssertionError, RuntimeError, IndexError):
    #     dataset = BasicDataset(dir_img, dir_mask, img_scale)
    dataset = BasicDataset(dir_img, dir_nrg, dir_mask, img_scale)
    pos_weight = torch.tensor(5, device=device)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_set.dataset.augment = True

    # 3. Create data loaders
    # loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    loader_args = dict(batch_size=batch_size, num_workers=2, pin_memory=False, prefetch_factor=1, persistent_workers=False)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # ──────────────────────────────────────────────
    # TensorBoard：初始化 SummaryWriter
    writer = SummaryWriter(comment=f'_lr{learning_rate}_bs{batch_size}')
    # 记录超参
    writer.add_hparams(
        {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'val_percent': val_percent,
            'img_scale': img_scale,
            'amp': amp
        },
        {}
    )
    # ──────────────────────────────────────────────

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    # grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    grad_scaler = torch.amp.GradScaler('cuda', enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        # loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        # loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                        # # 1) BCE with pos_weight
                        logits = masks_pred.squeeze(1)  # (B, H, W)
                        loss_bce = criterion(logits, true_masks.float())

                        # 2) 加权 Dice loss
                        probs = torch.sigmoid(logits)  # (B, H, W)
                        # 传入一个 “前景权重” 给 dice_loss
                        # 注意：dice_loss 新版签名 dice_loss(input, target, multiclass, class_weight)
                        loss_dice = dice_loss(
                            probs,
                            true_masks.float(),
                            multiclass=False,
                            class_weight=pos_weight  # 也可以用一个标量 tensor([99]) 来只给前景加权
                        )

                        loss = loss_bce + loss_dice

                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                # ──────────────────────────────────────────────
                # TensorBoard：记录训练损失
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/epoch', epoch, global_step)
                # ──────────────────────────────────────────────

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                # division_step = (n_train // (5 * batch_size))
                # if division_step > 0:
                #     if global_step % division_step == 0:
        # 记录权重 & 梯度直方图
        for tag, value in model.named_parameters():
            tag = tag.replace('/', '.')
            if not (torch.isinf(value) | torch.isnan(value)).any():
                writer.add_histogram('Weights/' + tag, value.data.cpu(), global_step)
            if value.grad is not None and not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                writer.add_histogram('Gradients/' + tag, value.grad.data.cpu(), global_step)

        val_score = evaluate(model, val_loader, device, amp)
        scheduler.step(val_score)

        logging.info('Validation Dice score: {}'.format(val_score))

        # ──────────────────────────────────────────────
        # TensorBoard：记录验证指标、学习率、可视化
        # writer.add_scalar('val/dice', val_score, global_step)
        # writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
        # writer.add_images('val/example_images', images[:4].cpu(), global_step)
        # writer.add_images('val/true_masks', true_masks[:4].unsqueeze(1).float().cpu(), global_step)
        # writer.add_images('val/pred_masks',
        #                   masks_pred.argmax(dim=1)[:4].unsqueeze(1).float().cpu(),
        #                   global_step)
        writer.add_scalar('val/dice', val_score, global_step)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

        # 把 6 通道拆成 RGB (0,1,2) 和 NRG (3,4,5) 两组三通道分别可视化
        rgb_imgs = images[:, 0:3, :, :].cpu()
        nrg_imgs = images[:, 3:6, :, :].cpu()
        writer.add_images('val/rgb_images', rgb_imgs[:4], global_step)
        writer.add_images('val/nrg_images', nrg_imgs[:4], global_step)

        # 真实 mask（二分类）和预测 mask（sigmoid threshold）
        true_masks_viz = true_masks[:4].unsqueeze(1).float().cpu()
        writer.add_images('val/true_masks', true_masks_viz, global_step)

        if model.n_classes == 1:
            probs = torch.sigmoid(masks_pred).cpu()
            bin_pred = (probs > 0.5).float()
            writer.add_images('val/pred_masks', bin_pred[:4], global_step)
        else:
         # 老的多分类可视化
            pred_labels = masks_pred.argmax(dim=1).unsqueeze(1).float().cpu()
            writer.add_images('val/pred_masks', pred_labels[:4], global_step)

        # ──────────────────────────────────────────────

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / f'checkpoint_epoch{epoch}.pth'))
            logging.info(f'Checkpoint {epoch} saved!')

    # ──────────────────────────────────────────────
    # 训练结束，关闭 SummaryWriter
    writer.close()
    # ──────────────────────────────────────────────


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=20.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = UNet(n_channels=7, n_classes=args.classes, bilinear=args.bilinear)
    # model = model.to(memory_format=torch.channels_last)
    # model = model.to(device)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
