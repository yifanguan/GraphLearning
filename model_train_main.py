from main import run


dataset_name = 'texas'
num_mp_layers = 6
num_fl_layers = 2 # number of mlp layer
mp_hidden_dim = 50
fl_hidden_dim = 128
epsilon = 5**0.5/2
optimizer_lr = 0.01
# weight_decay=5e-4
loss_func = 'CrossEntropyLoss'
total_epoch = 100
###############################
best_val, best_test = run(dataset_name, num_mp_layers, num_fl_layers, mp_hidden_dim,
                          fl_hidden_dim, epsilon, optimizer_lr, loss_func, total_epoch, index=0, freeze=False)

