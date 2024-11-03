# Train diverse ensemble
diverse_ensemble = DiverseEnsembleModel(params['model_params']).to(device)
diverse_optimizer = optim.AdamW(diverse_ensemble.parameters(), lr=1e-3, weight_decay=1e-4)
diverse_scheduler = CosineAnnealingWarmRestarts(
    diverse_optimizer, T_0=10, T_mult=2, eta_min=1e-6)

# Set up TensorBoard
writer = TorchSummaryWriter(log_dir=os.path.join(CONFIG['model_dir'], CONFIG['model_names']['tensorboard']))

logging.info("Training diverse ensemble model...")
best_diverse_accuracy = 0
diverse_best_state = None

for epoch in range(train_params['num_epochs']):
    diverse_ensemble.train()
    epoch_loss = 0
    for batch_idx, (data, spectral_features, target) in enumerate(train_loader):
        data, spectral_features, target = data.to(device), spectral_features.to(device), target.to(device)
        diverse_optimizer.zero_grad()
        output = diverse_ensemble(data, spectral_features)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(diverse_ensemble.parameters(), max_norm=1.0)
        
        diverse_optimizer.step()
        epoch_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    # Calculate average training loss for the epoch
    avg_train_loss = epoch_loss / len(train_loader)
    
    # Validation
    diverse_ensemble.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, spectral_features, target in val_loader:
            data, spectral_features, target = data.to(device), spectral_features.to(device), target.to(device)
            output = diverse_ensemble(data, spectral_features)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader)
    accuracy = correct / len(val_loader.dataset)

    # Log to TensorBoard
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    writer.add_scalar('Loss/validation', val_loss, epoch)
    writer.add_scalar('Accuracy/validation', accuracy, epoch)
    writer.add_scalar('Learning Rate', diverse_optimizer.param_groups[0]['lr'], epoch)

    diverse_scheduler.step()

    # Check for improvement and save the best model
    if accuracy > best_diverse_accuracy:
        best_diverse_accuracy = accuracy
        diverse_best_state = diverse_ensemble.state_dict()
        save_model(diverse_ensemble, CONFIG['model_names']['diverse'])
        logging.info(f"New best diverse ensemble model saved. Accuracy: {accuracy:.4f}")

# Close TensorBoard writer
writer.close()

# Evaluate the best diverse ensemble model
if diverse_best_state is not None:
    diverse_ensemble.load_state_dict(diverse_best_state)
    test_loss, test_accuracy, test_predictions = evaluate_model(
        diverse_ensemble, (X_test, X_test_spectral, y_test), criterion, device)
    logging.info(f"Diverse Ensemble Model - Final Test Accuracy: {test_accuracy:.4f}")

    # Generate and save confusion matrix for diverse ensemble
    cm = confusion_matrix(y_test.cpu().numpy(), test_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Diverse Ensemble')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(CONFIG['model_dir'], CONFIG['model_names']['diverse_confusion_matrix']))
    plt.close()

# Knowledge distillation
single_model = ImprovedSleepdetector(**params['model_params']).to(device)
logging.info("Performing knowledge distillation...")
distilled_model = distill_knowledge(ensemble_model, single_model, train_loader, 
                                  (X_val, X_val_spectral, y_val), device)
save_model(distilled_model, CONFIG['model_names']['distilled'])

# Final evaluation of all models
_, ensemble_accuracy, _ = evaluate_model(ensemble_model, (X_test, X_test_spectral, y_test), criterion, device)
_, diverse_accuracy, _ = evaluate_model(diverse_ensemble, (X_test, X_test_spectral, y_test), criterion, device)
_, distilled_accuracy, _ = evaluate_model(distilled_model, (X_test, X_test_spectral, y_test), criterion, device)

logging.info(f"""Training completed.
Ensemble Model - Final Test Accuracy: {ensemble_accuracy:.4f}
Diverse Ensemble Model - Final Test Accuracy: {diverse_accuracy:.4f}
Distilled Model - Final Test Accuracy: {distilled_accuracy:.4f}""")