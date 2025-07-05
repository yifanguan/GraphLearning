import matplotlib.pyplot as plt

def add_hyperparameter_text(params):
    '''
    Utility function used to return a clean ax for ploting data,
    and add fixed hyperparameters setting as text above title.

    params: a dict containing hyperparameters' values
    '''
    fig, ax = plt.subplots(figsize=(10, 5))

    # Adjust top margin to make room above title
    fig.subplots_adjust(top=0.8)

    # Add hyperparameter text ABOVE the title
    # param_text = "Hyperparameters:\n" + "   ".join(f"{k}: {v}" for k, v in params.items())
    param_text = "Hyperparameters:      "
    items = list(params.items())
    for i in range(0, len(items), 4):
        line = "   ".join(f"{k}: {v}" for k, v in items[i:i+4])
        param_text += line + "\n"

    fig.text(0.5, 0.92, param_text,
             fontsize=10,
             ha='center',
             va='bottom')

    return fig, ax
