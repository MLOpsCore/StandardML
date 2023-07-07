import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import ModelCheckpoint, \
    ReduceLROnPlateau, CSVLogger, EarlyStopping


def generate_callbacks(model_path, model_name):
    return [
        ModelCheckpoint(f"{model_path}/{model_name}/{model_name}.h5"),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
        CSVLogger(f"{model_path}/{model_name}/data.csv"),
        EarlyStopping(monitor='val_loss', patience=10,
                      restore_best_weights=False)
    ]


def plot_history(history, working_path, name):
    index, cols, rows = 1, 2, 3

    fig = plt.figure(figsize=(cols * 5, rows * 5))
    fig.suptitle(name, fontsize=16)

    for key, value in history.items():
        if "val" in key:
            continue

        val_lbl = f"val_{key}"

        ax = fig.add_subplot(rows, cols, index)
        ax.plot(value, label=key)

        if key != 'lr':
            ax.plot(history.get(val_lbl), label=val_lbl)

        ax.legend(), ax.set_title(key)
        index += 1

    plt.tight_layout()
    plt.savefig(f"{working_path}/{name}/training_process.png", dpi=300)
    plt.show()
