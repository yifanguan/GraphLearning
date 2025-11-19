from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import Compose, NormalizeFeatures, RandomNodeSplit

transform = Compose([
    NormalizeFeatures(),
    # RandomNodeSplit(num_train_per_class=0.6, num_val=0.2, num_test=0.2, split='train_rest')
    # RandomNodeSplit(split="random", num_train_per_class=20, num_val=500, num_test=1000)
])

# dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=transform)
dataset = Planetoid(root='/tmp/PubMed', name='PubMed', transform=transform)
# dataset = Amazon(root='/tmp/Amazon', name='Computers', transform=transform)
# dataset = Amazon(root='/tmp/Amazon', name='Photo', transform=transform)
# dataset = Coauthor(root='/tmp/Coauthor', name='Physics', transform=transform)
# dataset = Coauthor(root='/tmp/Coauthor', name='CS', transform=transform)
# dataset = WikiCS(root='/tmp/WikiCS', transform=transform)
data = dataset[0]
print(data)
# Device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")
# data = data.to(device)
print(f"# Train: {data.train_mask.sum().item()}")
print(f"# Val: {data.val_mask.sum().item()}")
print(f"# Test: {data.test_mask.sum().item()}")
