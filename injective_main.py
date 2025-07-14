from utils.distinct_features_and_rank import generate_expressive_power_plot, generate_expressive_power_plot_with_training

# mnist 0 digit has 517313 distinct features
# texas has 129 distinct features, 4 wl iterations
# cornell5 has 18577 distinct features, 3 wl iterations
# amazon-photo has 7460 distinct features, 3 wl iterations
# amazon-computers has 13349 distinct features, 4 wl iterations
# coauthor-cs has 17891 distinct features, 5 wl iterations
# coauthor-physics has 33661 distinct features, 5 wl iterations
# wikics has 10862 distinct features, 4 wl iterations
# roman-empire 22661 distinct features, 8 wl iterations
# amazon-ratings 19071 distinct features, 5 wl iterations
# minesweeper 1275 distinct features, 50 wl iterations
# tolokers 11595 distinct features, 4 wl iterations
# questions 27899 distinct features, 6 wl iterations
# squirrel 2197 distinct features, 4 wl iterations
# chameleon 808 distinct features, 3 wl iterations
# ogbn-arxiv 162564 distinct features, 7 wl iterations
# ogbn-products 162564 distinct features, 7 wl iterations
# ogbn-proteins 128261 distinct features, 3 wl iterations


# generate_expressive_power_plot(dataset_name='Cora', mp_depth=6, tolerance=1e-5, dim_list=[5, 10, 50, 150, 300, 500, 1000, 2000, 4000, 8000])

# generate_expressive_power_plot(dataset_name='texas', mp_depth=7, tolerance=1e-5, dim_list=[50,100,150,200,250,300,400,500,1000])

# generate_expressive_power_plot(dataset_name='texas', mp_depth=10, tolerance=1e-7, dim_list=[50,100,150,200,250,300,400,500,1000])

# dim_list=[100, 500, 1000, 2000, 5000, 10000, 20000, 40000, 80000]
# generate_expressive_power_plot(dataset_name='cornell5', mp_depth=8, tolerance=1e-5, dim_list=[100, 500, 1000, 2000, 5000, 10000])

# generate_expressive_power_plot(dataset_name='mnist', mp_depth=6, tolerance=1e-5, dim_list=[100, 500, 1000])


# generate_expressive_power_plot_transfer_learning(dataset_name='texas', mp_depth=6, tolerance=1e-5, dim_list=[50])

# generate_expressive_power_plot_with_training(dataset_name='texas', mp_depth=7, tolerance=1e-5, dim_list=[50,100,150,200,250,300,400,500,1000],
#                                              num_fl_layers=2, fl_hidden_dim=128, epsilon=5**0.5/2, optimizer_lr=0.01, total_epoch=100)




# generate_expressive_power_plot(dataset_name='amazon-photo', mp_depth=6, tolerance=1e-5, dim_list=[50, 100, 500, 1000, 2000, 4000, 8000, 16000])

# generate_expressive_power_plot(dataset_name='amazon-photo', mp_depth=6, tolerance=1e-5, dim_list=[50, 100, 500, 1000, 2000, 4000, 8000, 16000])





# TO RUN:
# generate_expressive_power_plot_with_training(dataset_name='cornell5', mp_depth=7, tolerance=1e-5, dim_list=[100, 500, 1000, 2000, 5000, 10000],
#                                              num_fl_layers=2, fl_hidden_dim=128, epsilon=5**0.5/2, optimizer_lr=0.01, total_epoch=300)
