from utils.distinct_features_and_rank import generate_expressive_power_plot, generate_expressive_power_plot_with_training

# generate_expressive_power_plot(dataset_name='Cora', mp_depth=6, tolerance=1e-5, dim_list=[5, 10, 50, 150, 300, 500, 1000, 2000, 4000, 8000])

# generate_expressive_power_plot(dataset_name='texas', mp_depth=7, tolerance=1e-5, dim_list=[50,100,150,200,250,300,400,500,1000])

# generate_expressive_power_plot(dataset_name='texas', mp_depth=10, tolerance=1e-7, dim_list=[50,100,150,200,250,300,400,500,1000])

# dim_list=[100, 500, 1000, 2000, 5000, 10000, 20000, 40000, 80000]
# generate_expressive_power_plot(dataset_name='cornell5', mp_depth=8, tolerance=1e-5, dim_list=[100, 500, 1000, 2000, 5000, 10000])

# generate_expressive_power_plot(dataset_name='mnist', mp_depth=6, tolerance=1e-5, dim_list=[100, 500, 1000])


# generate_expressive_power_plot_transfer_learning(dataset_name='texas', mp_depth=6, tolerance=1e-5, dim_list=[50])

generate_expressive_power_plot_with_training(dataset_name='texas', mp_depth=7, tolerance=1e-5, dim_list=[50,100,150,200,250,300,400,500,1000],
                                             num_fl_layers=2, fl_hidden_dim=128, epsilon=5**0.5/2, optimizer_lr=0.01, total_epoch=100)
