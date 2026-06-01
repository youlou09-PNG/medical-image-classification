for epoch in range(EPOCHS):

    # ==========================
    # TRAINING
    # ==========================
    model.train()

    running_train_loss = 0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

        _, preds = torch.max(outputs, 1)

        train_total += labels.size(0)
        train_correct += (
            preds == labels
        ).sum().item()

    train_loss = (
        running_train_loss
        / len(train_loader)
    )

    train_acc = (
        train_correct
        / train_total
    )

    # ==========================
    # VALIDATION
    # ==========================
    model.eval()

    running_val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_val_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (
                preds == labels
            ).sum().item()

    val_loss = (
        running_val_loss
        / len(val_loader)
    )

    val_acc = (
        val_correct
        / val_total
    )

    # ==========================
    # STORE HISTORY
    # ==========================
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    # ==========================
    # SAVE BEST MODEL
    # ==========================
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            PROJECT_ROOT / "models" / "best_baseline_model.pth"
        )

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Train Acc: {train_acc:.4f} "
        f"Val Acc: {val_acc:.4f}"
    )