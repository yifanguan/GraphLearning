from utils.distinct_features_and_rank import generate_expressive_power_plot, generate_expressive_power_plot_with_training
import torch
import gc

# mnist 0 digit has 517313 distinct features
# texas has 129 distinct features, 4 wl iterations, 183 nodes
# cornell5 has 18577 distinct features, 3 wl iterations, 18660 nodes
# amazon-photo has 7460 distinct features, 3 wl iterations, 7650 nodes
# amazon-computers has 13349 distinct features, 4 wl iterations, 13752 nodes
# coauthor-cs has 17891 distinct features, 5 wl iterations, 18333 nodes
# coauthor-physics has 33661 distinct features, 5 wl iterations, 34493 nodes
# wikics has 10862 distinct features, 4 wl iterations, 11701 nodes
# roman-empire 22661 distinct features, 8 wl iterations, 22662 nodes
# amazon-ratings 19071 distinct features, 5 wl iterations, 24492
# minesweeper 1275 distinct features, 50 wl iterations, 10000 nodes
# tolokers 11595 distinct features, 4 wl iterations
# questions 27899 distinct features, 6 wl iterations, 48921 nodes
# squirrel 2197 distinct features, 4 wl iterations, 2223 nodes
# chameleon 808 distinct features, 3 wl iterations
# ogbn-arxiv 162564 distinct features, 7 wl iterations
# ogbn-products 2306652 distinct features, 7 wl iterations
# ogbn-proteins 128261 distinct features, 3 wl iterations

# generate_expressive_power_plot(dataset_name='Cora', mp_depth=8, tolerance=1e-5, dim_list=[50])
# # generate_expressive_power_plot_with_training(dataset_name='Cora', mp_depth=6, tolerance=1e-5, dim_list=[50, 100, 500, 1000, 2000, 4000, 8000],
# #                                              num_fl_layers=2, fl_hidden_dim=128, epsilon=5**0.5/2, optimizer_lr=0.01, total_epoch=300)


# generate_expressive_power_plot(dataset_name='texas', mp_depth=7, tolerance=1e-5, dim_list=[50])

# # generate_expressive_power_plot(dataset_name='texas', mp_depth=10, tolerance=1e-7, dim_list=[50,100,150,200,250,300,400,500,1000])

# # dim_list=[100, 500, 1000, 2000, 5000, 10000, 20000, 40000, 80000]
# generate_expressive_power_plot(dataset_name='cornell5', mp_depth=8, tolerance=1e-5, dim_list=[100, 500, 1000, 2000, 5000, 10000])
# generate_expressive_power_plot(dataset_name='cornell5', mp_depth=8, tolerance=1e-5, dim_list=[100])


# generate_expressive_power_plot(dataset_name='citeseer', mp_depth=7, tolerance=1e-5, dim_list=[50])
# generate_expressive_power_plot(dataset_name='citeseer', mp_depth=7, tolerance=1e-5, dim_list=[50, 100, 500, 1000, 2000, 4000, 8000])


# generate_expressive_power_plot(dataset_name='pubmed', mp_depth=8, tolerance=1e-5, dim_list=[50, 100, 500, 1000, 2000, 4000, 8000])
# generate_expressive_power_plot(dataset_name='pubmed', mp_depth=8, tolerance=1e-5, dim_list=[50])


# generate_expressive_power_plot(dataset_name='mnist', mp_depth=6, tolerance=1e-5, dim_list=[100, 500, 1000])


# generate_expressive_power_plot_transfer_learning(dataset_name='texas', mp_depth=6, tolerance=1e-5, dim_list=[50])

# generate_expressive_power_plot_with_training(dataset_name='texas', mp_depth=7, tolerance=1e-5, dim_list=[50,100,150,200,250,300,400,500,1000],
#                                              num_fl_layers=2, fl_hidden_dim=128, epsilon=5**0.5/2, optimizer_lr=0.01, total_epoch=100)


# generate_expressive_power_plot(dataset_name='amazon-photo', mp_depth=5, tolerance=1e-5, dim_list=[50, 100, 500, 1000, 2000, 4000, 8000, 16000])
# generate_expressive_power_plot(dataset_name='amazon-photo', mp_depth=5, tolerance=1e-5, dim_list=[50])



# generate_expressive_power_plot(dataset_name='amazon-computers', mp_depth=6, tolerance=1e-5, dim_list=[50])


# generate_expressive_power_plot(dataset_name='coauthor-cs', mp_depth=6, tolerance=1e-5, dim_list=[50])


# generate_expressive_power_plot(dataset_name='coauthor-physics', mp_depth=6, tolerance=1e-5, dim_list=[50])


# generate_expressive_power_plot(dataset_name='wikics', mp_depth=6, tolerance=1e-5, dim_list=[50])


generate_expressive_power_plot(dataset_name='ogbn-arxiv', mp_depth=8, tolerance=1e-5, dim_list=[50])



# generate_expressive_power_plot(dataset_name='amazon-photo', mp_depth=6, tolerance=1e-5, dim_list=[50, 100, 500, 1000, 2000, 4000, 8000, 16000])


# generate_expressive_power_plot(dataset_name='minesweeper', mp_depth=50, tolerance=1e-5, dim_list=[50, 100, 250, 500, 1000, 2000, 4000])


# generate_expressive_power_plot(dataset_name='amazon-photo', mp_depth=5, tolerance=1e-5, dim_list=[50, 100, 500, 1000, 2000, 4000, 8000])

# generate_expressive_power_plot_with_training(dataset_name='amazon-photo', mp_depth=5, tolerance=1e-5, dim_list=[50, 100, 500, 1000, 2000, 4000, 8000],
#                                              num_fl_layers=2, fl_hidden_dim=128, epsilon=5**0.5/2, optimizer_lr=0.01, total_epoch=1000)


# TO RUN:
# generate_expressive_power_plot_with_training(dataset_name='cornell5', mp_depth=7, tolerance=1e-5, dim_list=[100, 500, 1000, 2000, 5000, 10000],
#                                              num_fl_layers=2, fl_hidden_dim=128, epsilon=5**0.5/2, optimizer_lr=0.01, total_epoch=300)





######################################******Using new model******######################################
# generate_expressive_power_plot(dataset_name='Cora', mp_depth=3, tolerance=1e-5, skip_conneciton=False, dim_list=[50, 100, 500, 1000, 2000, 4000, 8000])
# generate_expressive_power_plot_with_training(dataset_name='Cora', mp_depth=3, tolerance=1e-5, skip_connection=False, dropout=0, dim_list=[50, 100, 500, 1000, 2000, 4000, 8000],
#                                              num_fl_layers=5, fl_hidden_dim=128, epsilon=5**0.5/2, optimizer_lr=0.001, total_epoch=500)
# torch.cuda.empty_cache()       # release cached blocks
# gc.collect()                   # force Python to collect garbage
# torch.cuda.ipc_collect()       # clean up CUDA inter-process handles (optional)1
# dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# for dropout in dropout_rates:
#     generate_expressive_power_plot_with_training(dataset_name='Cora', mp_depth=3, tolerance=1e-5, skip_connection=False, dropout=dropout, dim_list=[50, 100, 500, 1000, 2000, 4000, 8000],
#                                                 num_fl_layers=5, fl_hidden_dim=128, epsilon=5**0.5/2, optimizer_lr=0.001, total_epoch=500)
    # ---- Clean up GPU memory ----
    # torch.cuda.empty_cache()       # release cached blocks
    # gc.collect()                   # force Python to collect garbage
    # torch.cuda.ipc_collect()       # clean up CUDA inter-process handles (optional)



# generate_expressive_power_plot(dataset_name='citeseer', mp_depth=6, tolerance=1e-5, skip_conneciton=False, dim_list=[50, 100, 500, 1000, 2000, 4000, 8000])
# generate_expressive_power_plot_with_training(dataset_name='citeseer', mp_depth=2, tolerance=1e-5, skip_connection=False, dropout=0, dim_list=[50, 100, 500, 1000, 2000, 4000, 8000],
#                                              num_fl_layers=5, fl_hidden_dim=128, epsilon=5**0.5/2, optimizer_lr=0.001, total_epoch=500)
# # ---- Clean up GPU memory ----
# torch.cuda.empty_cache()       # release cached blocks
# gc.collect()                   # force Python to collect garbage
# torch.cuda.ipc_collect()       # clean up CUDA inter-process handles (optional)1

# dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# # dropout_rates = [0.7, 0.8, 0.9]
# for dropout in dropout_rates:
#     generate_expressive_power_plot_with_training(dataset_name='citeseer', mp_depth=2, tolerance=1e-5, skip_connection=False, dropout=dropout, dim_list=[50, 100, 500, 1000, 2000, 4000, 8000],
#                                                  num_fl_layers=5, fl_hidden_dim=128, epsilon=5**0.5/2, optimizer_lr=0.001, total_epoch=1000)
#     # ---- Clean up GPU memory ----
#     torch.cuda.empty_cache()       # release cached blocks
#     gc.collect()                   # force Python to collect garbage
#     torch.cuda.ipc_collect()       # clean up CUDA inter-process handles (optional)
