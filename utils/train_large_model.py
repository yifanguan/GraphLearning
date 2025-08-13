import torch
from utils.over_smoothing_measure import dirichlet_energy


def train(model, data, train_idx, optimizer, criterion, energy_lambda, energy_threshold):
    model.train()
    out, embedding, norms_per_layer = model(data)
    energy_loss = dirichlet_energy(embedding, data.edge_index)
    loss = criterion(out[train_idx], data.y[train_idx])
    train_learning_loss = loss
    if energy_loss > energy_threshold:
        loss += energy_loss * energy_lambda
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), energy_loss.item(), train_learning_loss.item(), norms_per_layer


@torch.no_grad()
def evaluate(model, criterion, data, train_split_idx, validation_split_idx, test_split_idx, energy_lambda):
    model.eval()
    out, embedding, _ = model(data)
    energy_loss = dirichlet_energy(embedding, data.edge_index)
    valid_learning_loss = criterion(out[validation_split_idx], data.y[validation_split_idx])
    valid_loss = valid_learning_loss + energy_loss * energy_lambda
    test_learning_loss = criterion(out[test_split_idx], data.y[test_split_idx])
    test_loss = test_learning_loss + energy_loss * energy_lambda

    y_true_train = data.y[train_split_idx]
    y_pred_train = out[train_split_idx]
    y_pred_train = y_pred_train.argmax(dim=-1)
    y_true_validation = data.y[validation_split_idx]
    y_pred_validation = out[validation_split_idx]
    y_pred_validation = y_pred_validation.argmax(dim=-1)
    y_true_test = data.y[test_split_idx]
    y_pred_test = out[test_split_idx]
    y_pred_test = y_pred_test.argmax(dim=-1)

    correct_train = y_true_train == y_pred_train
    correct_valid = y_true_validation == y_pred_validation
    correct_tes = y_true_test == y_pred_test

    train_acc = correct_train.sum().item() / len(correct_train)
    valid_acc = correct_valid.sum().item() / len(correct_valid)
    test_acc = correct_tes.sum().item() / len(correct_tes)

    return train_acc, valid_acc, test_acc, valid_loss.item(), test_loss.item(), energy_loss.item(), \
            valid_learning_loss.item(), test_learning_loss.item()
