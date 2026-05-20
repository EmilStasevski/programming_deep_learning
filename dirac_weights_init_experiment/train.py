import torch


def train_one_epoch(model, dataloader, criterion, optimizer, device, loss_history):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # --- THE MAGIC FIX: Clip gradients to prevent explosion ---
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            loss_history.append(running_loss / 100)
            running_loss = 0.0