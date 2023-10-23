import os
import torch
import monai
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def train(
    model,
    model_name,
    train_loader,
    val_loader,
    loss_function,
    optimizer,
    scheduler,
    device,
    log_dir,
    metric,
    epochs
):
    post_pred = monai.transforms.Compose([monai.transforms.AsDiscrete(argmax=True, to_onehot=2)])
    post_label = monai.transforms.Compose([monai.transforms.AsDiscrete(to_onehot=2)])
    writer = SummaryWriter(log_dir)
    current_step = 0
    best_metric = 0
    for epoch in tqdm(
        range(epochs),
        desc=" Overall Training Progress",
        position=0,
    ):
        current_step = _train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=loss_function,
            device=device,
            writer=writer,
            epoch=epoch,
            STEP=current_step,
            post_pred=post_pred, 
            metric=metric,
            post_label=post_label
        )

        new_metric = _val_one_epoch(
            model=model,
            device=device,
            val_loader=val_loader,
            post_pred=post_pred,
            metric=metric,
            writer=writer,
            epoch=epoch,
            loss_function=loss_function,
            post_label=post_label
        )

        if new_metric > best_metric:
            best_metric = new_metric
            logging.info(
                f"Epoch {epoch:3d}\t\tNew Highest Metric: {best_metric:.4f}\t\tSaving Model"
            )
            _save_model(model, model_name)


def _train_one_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    loss_function,
    device,
    writer,
    epoch,
    STEP,
    post_pred,
    metric,
    post_label    
):
    """
    Implement a training loop for one epoch
    """
    epoch_loss = 0
    i = 0
    model.train()
    for batch_data in tqdm(
        train_loader, desc=" Epoch Train Progress", position=1, leave=False
    ):
        STEP += 1
        i += 1
        
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        outputs = [post_pred(i) for i in monai.data.decollate_batch(outputs)]
        labels = [post_label(i) for i in monai.data.decollate_batch(labels)]
        
        metric(outputs, labels)
        epoch_loss += loss.item()

        ## TIMM per batch update
        scheduler.step_update(num_updates=STEP)
        writer.add_scalar("Train/lr", optimizer.param_groups[0]["lr"], STEP)
    
    ## TIMM per epoch update
    scheduler.step(epoch+1)
    
    ## Calculate mean dice across epoch
    epoch_dice = metric.aggregate().item()
    metric.reset()

    writer.add_scalar("Train/Epoch-Loss", epoch_loss, epoch)
    writer.add_scalar("Train/Epoch-Dice", epoch_dice, epoch)
    return STEP


def _val_one_epoch(
    model,
    device,
    val_loader,
    post_pred,
    metric,
    writer,
    epoch,
    loss_function,
    post_label
):
    val_loss = 0
    model.eval()
    with torch.no_grad():

        # For each batch in validation set
        for val_data in tqdm(
            val_loader, desc=" Epoch Val Progress", position=1, leave=False
        ):
            val_images, val_labels = (
                val_data["image"].to(device),
                val_data["label"].to(device),
            )
            with torch.cuda.amp.autocast():
                val_preds = model(val_images)
                
            val_loss += loss_function(val_preds, val_labels).item()
            # Process results from validation
            val_outputs = [post_pred(i) for i in monai.data.decollate_batch(val_preds)]
            val_labels = [post_label(i) for i in monai.data.decollate_batch(val_labels)]

            metric(val_outputs, val_labels)
        val_dice_list = metric.get_buffer().cpu().numpy()
        dice_vals = [item for sublist in val_dice_list for item in sublist]
        val_batch_dice = metric.aggregate().item()
        metric.reset()

    # Write values to TensorBoard
    writer.add_scalar("Validation/Dice", val_batch_dice, epoch)
    writer.add_scalar("Validation/Loss", val_loss, epoch)
    figure = plot_img_seg(
        np.squeeze(val_images[0].cpu()), 
        np.squeeze(val_labels[0][1,:,:].cpu()), 
        np.squeeze(val_outputs[0][1,:,:].cpu())
    )
    var_figure = plot_dice_dist(dice_vals)
    writer.add_figure(tag="Validation/Model_Output", figure=figure, global_step=epoch)
    writer.add_figure(tag="Validation/Dice_Distribution", figure=var_figure, global_step=epoch)
    # Return metric
    return val_batch_dice


def _save_model(model, model_name):
    torch.save(
        model.state_dict(),
        os.path.join("models", model_name + ".pth")
    )


def test_model(model, loader, device):
    post_pred = monai.transforms.Compose([monai.transforms.AsDiscrete(argmax=True, to_onehot=2)])
    post_label = monai.transforms.Compose([monai.transforms.AsDiscrete(to_onehot=2)])
    model.eval()
    with torch.no_grad():
        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y_true = torch.tensor([], dtype=torch.float32, device=device)
        path_list = []
        
        # For each batch in validation set
        for data in tqdm(loader, desc=" Testing Progress", position=1, leave=True):
            images, labels, paths = (
                data["image"].to(device),
                data["label"].to(device),
                data["image_path"]
            )
            y_pred = torch.cat([y_pred, model(images)], dim=0)
            y_true = torch.cat([y_true, labels], dim=0)
            path_list.extend(paths)

        # Process results from validation
        y_true_proc = [post_label(i) for i in monai.data.decollate_batch(y_true, detach=False)]
        y_pred_proc = [post_pred(i) for i in monai.data.decollate_batch(y_pred)]
        
        results_list = []
        results_length = len(y_true_proc)
        for i in range(results_length):
            result = {}
            result["true"] = y_true_proc[i].cpu()
            result["pred"] = y_pred_proc[i].cpu()
            result["pred_raw"] = y_pred[i].cpu()
            result["path"] = path_list[i]
            results_list.append(result)
        
        df = pd.DataFrame(results_list)
        df["dice"] = df.apply(lambda x: calc_dice(x["pred"], x["true"]), axis=1)

    return df

def plot_img_seg(image, label, seg):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 15))
    fig.suptitle("Ground Truth and Model Output")
    
    ax1.set_title("Ground Truth")
    ax1.imshow(image, cmap="bone")
    ax1.imshow(label, alpha=0.5)
    ax1.axis("off")
    
    ax2.set_title("Model Output")
    ax2.imshow(image, cmap="bone")
    ax2.imshow(seg, alpha=0.5)
    ax2.axis("off")
    
    return fig

def calc_dice(pred, true):
    dice_metric = monai.metrics.DiceMetric(include_background=False)
    dice = dice_metric(pred[None,:,:,:], true[None,:,:,:])
    return dice.item()

def plot_dice_dist(dice_score_list):
    fig, _ = plt.subplots()
    sns.set_theme()
    sns.set_style("white")
    sns.stripplot(dice_score_list, orient="h")
    plt.title("Dice Score Distribution")
    plt.xlabel("Dice Score")
    plt.xlim((0,1)) 
    
    return fig  