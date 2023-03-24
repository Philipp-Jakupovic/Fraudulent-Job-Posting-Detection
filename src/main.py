import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import JobPostingClassifier
from dataset import JobPostingDataset, tokenizer, max_len
from preprocessing import load_data, preprocess_data, split_data
from train import train_epoch
from validation import evaluate
from test import test_model
from sklearn.model_selection import KFold

def main():
    """
    Main function to load and preprocess data, create Dataset and DataLoader objects, initialize model, loss function, and optimizer,
    train and validate the model using K-fold cross-validation, and test the model. It also calculates precision, recall, and F1-score.
    
    This function does not accept any input arguments.
    
    Returns:
        None
    """
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess data
    filepath = "~/fraud_detection/data/fake_job_postings.csv"
    df = load_data(filepath)
    df = preprocess_data(df)
    train_val_df, test_df = split_data(df, test_size=0.2)
    
    # Initialize KFold cross-validator
    n_splits = 5
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Train and validate the model using K-fold cross-validation
    n_epochs = 3
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_df)):
        print(f"Fold {fold + 1}/{n_splits}")

        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]

        train_dataset = JobPostingDataset(train_df['text'], train_df['fraudulent'], tokenizer, max_len)
        val_dataset = JobPostingDataset(val_df['text'], val_df['fraudulent'], tokenizer, max_len)

        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        # Initialize model, loss function, and optimizer
        model = JobPostingClassifier(2).to(device)
        loss_fn = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=2e-5)

        for epoch in range(n_epochs):
            print(f"Epoch {epoch + 1}/{n_epochs}")
            train_loss, train_acc = train_epoch(model, train_dataloader, loss_fn, optimizer, device)
            print(f"Training loss: {train_loss}, Training accuracy: {train_acc}")

            val_loss, val_acc = evaluate(model, val_dataloader, loss_fn, device)
            print(f"Validation loss: {val_loss}, Validation accuracy: {val_acc}")

    # Test the model
    test_dataset = JobPostingDataset(test_df['text'], test_df['fraudulent'], tokenizer, max_len)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    test_acc, (precision, recall, f1_score) = test_model(model, test_dataloader, device)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")

if __name__ == '__main__':
    main()
