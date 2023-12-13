import numpy as np
import torch
import torch.nn as nn
from tqdm import trange, tqdm
from typing import Tuple
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from .dataset import construct_datasets
from .classifier import Classifier


def train_classifier(dataloader: DataLoader, classifier: Classifier, optimizer: torch.optim,
                     loss_function: nn.Module, verbose: bool, random_seed: int=None) -> float:
    """ Trains the classifier for one epoch.

    Parameters
    ----------
    dataloader: Data loader from HateSpeechDataset class
    classifier: Instance of Classifier class
    optimizer: Optimizer for the classifier
    loss_function: Loss function for the classifier
    verbose: Whether to print training details
    random_seed: Random seed for reproducibility

    Returns
    -------
    average_loss: Average loss of the classifier
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
    # Training loop
    if verbose:
        dataloader = tqdm(dataloader)
    train_loss = 0
    for labels, sentences in dataloader:
        # Forward pass
        outputs = classifier(sentences)
        labels = labels.to(outputs.device)
        # Loss and backpropagation
        loss = loss_function(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Update the total loss
        current_loss = loss.item()
        train_loss += current_loss
    average_loss = train_loss / len(dataloader)
    return average_loss


def evaluate_classifier(dataloader: DataLoader, classifier: Classifier, 
                        verbose: bool) -> Tuple[float, float]:
    """ Evaluates the classifier.

    Parameters
    ----------
    dataloader: Data loader from HateSpeechDataset class
    classifier: Instance of Classifier class
    verbose: Whether to print training details

    Returns
    -------
    accuracy: Accuracy of the classifier
    f1: F1-score of the classifier (harmonic mean of precision and recall)
    """
    classifier.eval()
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        if verbose:
            dataloader = tqdm(dataloader)
        for labels, sentences in dataloader:
            outputs = classifier(sentences)
            predictions = torch.argmax(outputs, dim=1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    f1s = f1_score(true_labels, predicted_labels, average=None)
    if verbose:
        print(f'F1 Scores: {[round(f1, 4) for f1 in f1s]}')
        print('Micro:', f1_score(true_labels, predicted_labels, average='micro'))
        print('Macro:', f1_score(true_labels, predicted_labels, average='macro'))
    return accuracy, f1


def run_experiments(data_dir: str, model_name: str, layers_to_unfreeze: int, pooling: str, 
                    classifier_hidden_size: int, n_experiments: int, n_epochs: int, patience: int, 
                    batch_size: int, verbose: bool=False) -> Tuple[Classifier, float, int]:
    """ Run experiments training and evaluating the classifier.

    Parameters
    ----------
    data_dir: location of the data
    model_name: Pretrained model to load
    layers_to_unfreeze: Number of pretrained layers to unfreeze
    pooling: 'CLS' (BERT only) or 'MLP'
    classifier_hidden_size: Hidden size of the classifier
    n_experiments: Number of experiments to run
    n_epochs: Number of epochs to train the classifier
    patience: Number of epochs for early stopping (using F1-score)
    batch_size: Maximum number of examples in a batch
    verbose: Whether to print training details, default False

    Returns
    -------
    classifier: Trained classifier from the final experiment
    best_test_f1: F1-score on test set using classifier with the best dev F1-score
    trainable_params: Number of trainable parameters in the classifier
    """
    # Loss function
    loss_function = nn.CrossEntropyLoss()

    # Load the tokenizer and construct datasets and data loaders
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if 'gpt2' in model_name:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = 1024
    datasets, dataloaders = construct_datasets(
        prefix=data_dir,
        batch_size=batch_size,
        tokenizer=tokenizer,
        device=device)

    # Train the classifier and evaluate its performance
    dev_f1s = []
    test_f1s = []

    # Run the experiment n times
    for i in range(n_experiments):
        # Initialize the classifier
        torch.manual_seed(i)
        classifier = Classifier(model_name, layers_to_unfreeze, pooling,
                                classifier_hidden_size, datasets['train'].n_classes)
        classifier.to(classifier.model.device)

        # Initialize the optimizer
        optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-4)

        # Train the classifer and print the accuracy on the dev set
        best_dev_f1 = 0.0
        no_improvement_count = 0
        if verbose:
            print(f'Training classifier {i + 1}')
        for epoch in range(n_epochs):
            train_loss = train_classifier(dataloaders['train'], classifier, optimizer, loss_function, verbose, i)
            dev_accuracy, dev_f1 = evaluate_classifier(dataloaders['dev'], classifier, verbose)
            if verbose:
                print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Dev Accuracy: {dev_accuracy:.4f}, Dev F1: {dev_f1:.4f}')
            # Early stopping
            if dev_f1 >= best_dev_f1:
                best_dev_f1 = dev_f1
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            if no_improvement_count >= patience:
                if verbose:
                    print(f'Early stopping: No improvement in F1-score for {patience} consecutive epochs.')
                break

        # Evaluate the classifier on the test set
        test_accuracy, test_f1  = evaluate_classifier(dataloaders['test'], classifier, verbose)
        if verbose:
            print(f'Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}\n')

        # Save the dev and test accuracies
        dev_f1s.append(dev_f1)
        test_f1s.append(test_f1)

    best_test_f1 = test_f1s[np.argmax(dev_f1s)]
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    if verbose:
        print(f'Trainable parameters: {trainable_params}')
        print('Dev Set F1')
        print(f'    Mean: {np.average(dev_f1s):.4f}')
        print(f'    Std: {np.std(dev_f1s):.4f}')
        print(f'Best Model F1: {best_test_f1:.4f}')
    return classifier, best_test_f1, trainable_params


def run_experiments_5cv(model_name: str, layers_to_unfreeze: int, pooling: str,
                        classifier_hidden_size: int, n_experiments: int,
                        n_epochs: int, patience: int, batch_size: int):
    """ Run experiments with 5 fold cross validation.

    Parameters
    ----------
    model_name: Pretrained model to load
    layers_to_unfreeze: Number of pretrained layers to unfreeze
    pooling: 'CLS' (BERT only) or 'MLP'
    classifier_hidden_size: Hidden size of the classifier
    n_experiments: Number of experiments to run
    n_epochs: Number of epochs to train the classifier
    patience: Number of epochs for early stopping (using F1-score)
    batch_size: Maximum number of examples in a batch

    Returns
    -------
    f1s: F1-scores of the classifier across 5 folds
    """
    print(f'Model: {model_name}')
    print(f'    Layers to unfreeze: {layers_to_unfreeze}')
    print(f'    Pooling: {pooling}')
    f1s = []
    for i in trange(1, 6):
        data_dir = f'data/fold{i}/'
        _, f1, trainable_params = run_experiments(data_dir, model_name, layers_to_unfreeze,
                                                  pooling, classifier_hidden_size, n_experiments,
                                                  n_epochs, patience, batch_size)
        f1s.append(f1)
    print(f'    Total trainable parameters: {trainable_params}')
    print('Results')
    print(f'    F1 Scores: {[round(f1, 4) for f1 in f1s]}')
    print(f'    Mean: {np.mean(f1s):.4f}')
    print(f'    Std: {np.std(f1s):.4f}\n')
    return f1s