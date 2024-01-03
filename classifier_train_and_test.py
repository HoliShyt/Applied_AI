from config import DEVICE, LEARNING_RATE, OUT_DIR
from datasets import initialize_dataloaders
from models import instantiate_model

from typing import Sequence, Any

import torch
import torch.utils.data as tdata
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


# Train model on training dataset for one epoch
def train_model_single_epoch(model: torch.nn.Module,
                             train_data_loader: tdata.DataLoader,
                             loss_criterion: torch.nn.Module,
                             optimizer: torch.optim.Optimizer):
    print('Training model')

    running_loss = 0
    n_training_batches = len(train_data_loader)

    # Instantiate progress bar
    with tqdm(train_data_loader, total=n_training_batches) as prog_bar:
        # Train on each batch
        for i, data in enumerate(prog_bar):
            images, labels = data

            optimizer.zero_grad()
            # Flatten 2d images into column vectors, feed to network, do backprop
            images = torch.flatten(images, start_dim=1).to(DEVICE)
            output = model.forward(images)

            loss = loss_criterion(output, labels.to(DEVICE))
            loss.backward()

            optimizer.step()

            # Get loss value, update progress bar
            loss_value = loss.item()
            running_loss += loss_value

            prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

    # Training loss is running loss divided by number of batches
    training_loss = running_loss/n_training_batches
    print(f"Training loss: {training_loss}")
    return training_loss


# Validate model on validation dataset for one epoch
def validate_model_single_epoch(model: torch.nn.Module,
                                valid_data_loader: tdata.DataLoader,
                                loss_criterion: torch.nn.Module):
    print('Validating model')

    running_loss = 0
    n_valid_batches = len(valid_data_loader)

    # Instantiate progress bar
    with tqdm(valid_data_loader, total=n_valid_batches) as prog_bar:
        # Validate on each batcn
        for i, data in enumerate(prog_bar):
            images, labels = data

            # Flatten 2d image to vector, send to model, get loss
            images = torch.flatten(images, start_dim=1).to(DEVICE)

            with torch.no_grad():
                output = model.forward(images)

                loss = loss_criterion(output, labels.to(DEVICE))
                loss_value = loss.item()

            running_loss += loss_value

            prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

    # Validation loss is running loss divided by number of validation batches
    validation_loss = running_loss / n_valid_batches
    print(f"Validation loss: {validation_loss}")
    return validation_loss


# Train and validate model on all epochs
def train_and_validate_model_all_epochs(n_epochs: int,
                                        model: torch.nn.Module,
                                        train_data_loader: tdata.DataLoader,
                                        valid_data_loader: tdata.DataLoader,
                                        loss_criterion: torch.nn.Module,
                                        optimizer: torch.optim.Optimizer):

    # List of training/validation losses for all epochs for given model
    training_loss_list = [float('nan')] * n_epochs
    valid_loss_list = [float('nan')] * n_epochs

    for i in range(n_epochs):
        # Train and validate on all epochs and return completed lists
        print(f"\n= EPOCH {i+1} =")
        training_loss = train_model_single_epoch(model, train_data_loader, loss_criterion, optimizer)
        valid_loss = validate_model_single_epoch(model, valid_data_loader, loss_criterion)

        training_loss_list[i] = training_loss
        valid_loss_list[i] = valid_loss

    return training_loss_list, valid_loss_list


# Test model on testing dataset
def test_model(model: torch.nn.Module,
               test_data_loader: tdata.DataLoader):
    print('Testing model')

    # Loss criterion is negative log likelihood because we only have mlp with log_softmax as output layer,
    # but this line should be changed if we had more models
    loss_criterion = torch.nn.NLLLoss()

    running_loss = 0
    n_classification_errors = 0
    n_test_samples = 0
    n_test_batches = len(test_data_loader)

    # Initialize progress bar
    with tqdm(test_data_loader, total=n_test_batches) as prog_bar:
        # Test on all test batches
        for i, data in enumerate(prog_bar):
            images, labels = data

            # Flatten 2d images to vectors, feed to model, get loss and number of classification errors
            with torch.no_grad():
                images = torch.flatten(images, start_dim=1).to(DEVICE)

                output = model.forward(images)
                predicted_labels = torch.argmax(output, dim=-1)
                n_classification_errors += torch.sum(labels.to(DEVICE) != predicted_labels).item()

                loss = loss_criterion(output, labels.to(DEVICE))
                loss_value = loss.item()

            running_loss += loss_value
            n_test_samples += images.shape[0]

            prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

    # Testing loss is running loss over number of batches
    testing_loss = running_loss / n_test_batches
    # Estimated risk is the number of classification errors divided by the total number of sample points
    estimated_risk = n_classification_errors / n_test_samples
    return testing_loss, estimated_risk


# Find the best model in terms of validation loss for multiple sets of initialization parameters, in model_param_sets
def find_best_model(model_type: str,
                    n_epochs: int,
                    train_data_loader: tdata.DataLoader,
                    valid_data_loader: tdata.DataLoader,
                    model_param_sets: Sequence[Any]):

    # Remember the best validation loss, model + index, and the list of training/validation losses for the best model
    best_valid_loss = float('inf')
    best_model = None
    best_model_idx = -1
    best_train_list = []
    best_valid_list = []

    # For each set to test, instantiate corresponding model
    for i, param_set in enumerate(model_param_sets):
        print(f"---\n\nModel [{i+1}]:")
        model = instantiate_model(model_type, *param_set).to(DEVICE)
        print(model)

        # Loss criterion is negative log likelihood because we only have mlp with log_softmax as output layer,
        # but this line should be changed if we had more models
        loss_criterion = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Train and validate model on all epochs
        training_loss_list, valid_loss_list = train_and_validate_model_all_epochs(n_epochs, model,
                                                                                  train_data_loader,
                                                                                  valid_data_loader,
                                                                                  loss_criterion,
                                                                                  optimizer)

        # Remember model and its properties if its final validation loss is better than the best
        if valid_loss_list[-1] < best_valid_loss:
            best_valid_loss = valid_loss_list[-1]
            best_model = model
            best_model_idx = i+1
            best_train_list = training_loss_list
            best_valid_list = valid_loss_list

    print(f"BEST MODEL FOUND: Model [{best_model_idx}]")
    print(best_model)
    return best_model, best_train_list, best_valid_list


# Full procedure : train multiple models, find the best one, save it and test it
def full_classifier_procedure(dataset: str, trainval_split: float,
                              classifier_type: str, n_epochs: int,
                              parameter_sets: Sequence[Any]):
    # Initialize dataloaders and find the best model of given types with the sets of paramters
    trainloader, validloader, testloader = initialize_dataloaders(dataset, trainval_split)
    best_model, best_train_list, best_valid_list = find_best_model(classifier_type, n_epochs,
                                                                   trainloader, validloader,
                                                                   parameter_sets)

    # Save weights for best model
    model_folder = f"{OUT_DIR}/{classifier_type}_{dataset}"
    torch.save({'model_state_dict': best_model.state_dict()}, f'{model_folder}/best_model.pth')
    # Plot training loss and validation loss over epochs for the best model
    figure_1, train_ax = plt.subplots()
    figure_2, valid_ax = plt.subplots()
    epoch_axis = list(range(1, n_epochs+1))
    train_ax.plot(epoch_axis, best_train_list, color='tab:blue')
    train_ax.set_xlabel('epoch')
    train_ax.set_ylabel('train loss')
    valid_ax.plot(epoch_axis, best_valid_list, color='tab:red')
    valid_ax.set_xlabel('epoch')
    valid_ax.set_ylabel('validation loss')
    figure_1.savefig(f"{model_folder}/train_loss.png")
    figure_2.savefig(f"{model_folder}/valid_loss.png")
    plt.close('all')

    # Test model on testing data, these are our final values
    testing_loss, estimated_risk = test_model(best_model, testloader)
    print(f"FINAL TESTING LOSS: {testing_loss}")
    print(f"ESTIMATED CLASSIFIER RISK: {estimated_risk}")
