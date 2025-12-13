from torch_geometric.transforms import Compose, NormalizeFeatures, RandomNodeSplit
import math
from .mup import MuGNN, make_mup_optimizer

def train_val_test_mask_helper(dataset_name, dataset):
    '''
    UPDATE dataset masks and idx for experiment.
    This is the single place for changing them for simplicity.
    Customized for each dataset.
    '''
    if dataset_name == 'pubmed' or dataset_name == 'cora' or dataset_name == 'citeseer' or dataset_name == 'ogbn-arxiv' or dataset_name == 'ogbn-products':
        transform = RandomNodeSplit(num_train_per_class=0.6, num_val=0.2, num_test=0.2, split='train_rest')
        dataset.graph = transform(dataset.graph)
        dataset.train_idx = dataset.graph.train_mask
        dataset.train_mask = dataset.graph.train_mask
        dataset.val_idx = dataset.graph.val_mask
        dataset.val_mask = dataset.graph.val_mask
        dataset.test_idx = dataset.graph.test_mask
        dataset.test_mask = dataset.graph.test_mask

    return dataset

def train_loop(widths, depths, lrs, input_dim, output_dim, K, device,
                          num_epochs, log_every, train_func, evaluate_func):
    rows = []
    for width in widths:
        for depth in depths:
            for log2lr in lrs:
                lr = 2**log2lr
                key = f"width={width} depth={depth} lr={lr:g}"
                print(f"\n=== Training {key} ===")

                model = MuGNN(
                    input_dim=input_dim, # dataset.num_node_features
                    hidden_dim=width,
                    output_dim=output_dim, # dataset.num_classes
                    num_fc_layers=depth, # depth = num_fc_layers + 1; actually, so we minus one here
                    K=K).to(device)

                optimizer = make_mup_optimizer(model, base_lr=lr, opt="sgd", weight_decay=0.0, momentum=0)

                # Best trackers (value + epoch)
                best = {
                    "train_loss": (math.inf, -1),
                    "val_loss":   (math.inf, -1),
                    "test_loss":  (math.inf, -1),
                    "train_acc":  (0.0, -1),
                    "val_acc":    (0.0, -1),
                    "test_acc":   (0.0, -1),
                }

                # loss trackers
                # we evaluate val and test every log_every epoch, so record these loss point for val and test loss plot
                # we evaluate train every epoch (for best train loss, we get its value from evaluate step, the train loss plot is
                # diverged from this observation for more data points purpose) change this part if needed.
                loss_dict = {
                    "train_loss": [],
                    "val_loss": [],
                    "test_loss": []
                }

                for epoch in range(1, num_epochs + 1):
                    train_loss = train_func(model=model, optimizer=optimizer)
                    loss_dict['train_loss'].append((train_loss, epoch))

                    if epoch == 1 or epoch % log_every == 0 or epoch == num_epochs:
                        m = evaluate_func(model=model) # result dict
                        # mini-batch original code: left here for example visualization
                        # m = evaluate(model, {'train' : train_loader, 'val' : val_loader, 'test' : test_loader}, device) # result dict

                        # update bests
                        for k in ["train_loss", "val_loss", "test_loss"]:
                            if m[k] < best[k][0]:
                                best[k] = (m[k], epoch)
                        for k in ["train_acc", "val_acc", "test_acc"]:
                            if m[k] > best[k][0]:
                                best[k] = (m[k], epoch)

                        # record loss
                        loss_dict['val_loss'].append((m['val_loss'], epoch))
                        loss_dict['test_loss'].append((m['test_loss'], epoch))

                        print(
                            f"Epoch {epoch:03d} | "
                            f"Train: loss {m['train_loss']:.4f}, acc {m['train_acc']:.4f}, best {best['train_acc'][0]:.4f} (ep {best['train_acc'][1]}) | "
                            f"Val: loss {m['val_loss']:.4f}, acc {m['val_acc']:.4f}, best {best['val_acc'][0]:.4f} (ep {best['val_acc'][1]}) | "
                            f"Test: loss {m['test_loss']:.4f}, acc {m['test_acc']:.4f}, best {best['test_acc'][0]:.4f} (ep {best['test_acc'][1]})"
                        )

                # last epoch metrics
                last_m = evaluate_func(model=model)

                # store a row per run
                rows.append({
                    "width": width,
                    "depth": depth,
                    "lr": lr,
                    "best_train_loss": best["train_loss"][0],
                    "best_train_loss_epoch": best["train_loss"][1],
                    "best_val_loss": best["val_loss"][0],
                    "best_val_loss_epoch": best["val_loss"][1],
                    "best_test_loss": best["test_loss"][0],
                    "best_test_loss_epoch": best["test_loss"][1],
                    "best_train_acc": best["train_acc"][0],
                    "best_train_acc_epoch": best["train_acc"][1],
                    "best_val_acc": best["val_acc"][0],
                    "best_val_acc_epoch": best["val_acc"][1],
                    "best_test_acc": best["test_acc"][0],
                    "best_test_acc_epoch": best["test_acc"][1],
                    "train_loss": loss_dict['train_loss'],
                    "val_loss": loss_dict['val_loss'],
                    "test_loss": loss_dict['test_loss'],
                    "last_train_loss": last_m["train_loss"],
                    "last_val_loss": last_m["val_loss"],
                    "last_test_loss": last_m["test_loss"],
                    "last_train_acc": last_m["train_acc"],
                    "last_val_acc": last_m["val_acc"],
                    "last_test_acc": last_m["test_acc"],
                })
    return rows
