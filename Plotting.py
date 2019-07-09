import matplotlib.pyplot as plt
import pandas as pd
import model


# TODO: comment out the code here, important for future useability
def plot_data(filename, y_label, x_label, Headers=[], data_len=None, marker=".", linestyle="None", markersize=0.5,
              color=['red'], legend=False, legend_labels=[], save_fig=False, title="", logy=False):
    if data_len:
        df = pd.read_csv(filename, nrows=data_len)
    else:
        df = pd.read_csv(filename)
    for item in Headers:
        if logy == True:
            plt.semilogy(df[item], marker=marker, color=color[Headers.index(item)], linestyle=linestyle,
                         markersize=markersize)
        else:
            plt.plot(df[item], marker=marker, color=color[Headers.index(item)], linestyle=linestyle,
                         markersize=markersize)
        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        if legend == True:
            plt.legend(legend_labels, loc='upper right')
        if save_fig == True:
            plt.savefig("/Users/Robin/Desktop/ML Project/Code/Plots/" + title + ".png", dpi=300)


def run_plots():
    plt.show()


def plot_all_SE_nodes(filename, nodes, data_len, x_label, y_label, save_fig=False, title=""):
    for item in nodes:
        plot_data(filename, data_len=data_len, Headers=[item + "_SE"], x_label=x_label, y_label=y_label,
                  save_fig=save_fig, title=item + title, logy=True)
        run_plots()


if __name__ == "__main__":
    #for i in range (1,9):
"""
        # plot_data("/Users/Robin/Desktop/ML Project/Code/Test_Results/model_simulation_data_model6_test_results.csv", "DateTime", "39835202_29985195_SE")
        plot_all_SE_nodes("/Users/Robin/Desktop/ML Project/Code/Test_Results/model_simulation_data_model{}_test_results.csv".format(i),
                          model.MEASUREMENT_NODES_2, data_len=67000, y_label="Squared Error", x_label="Sample Number", save_fig=True, title="model {}".format(i))
        plot_data("/Users/Robin/Desktop/ML Project/Code/Training_Results/simulation_data_model{}_tests.csv".format(i), "Loss", "Epoch",
                  Headers=["val_loss", "loss"], markersize=3, linestyle="solid", color=["red", "blue"], legend=True,
                  legend_labels=["val_loss", "loss"], title="Training and Validation Plot Model {}".format(i), save_fig=True)
        run_plots()
        plot_data("/Users/Robin/Desktop/ML Project/Code/Test_Results/model_simulation_data_model{}_test_results.csv".format(i), "MSE",
                  "Sample Number", Headers=["MSE"], title="MSE model {}".format(i), save_fig=True, logy=True)
        run_plots()"""
       plot_data("Training_Results/simulation_data_model{}_tests.csv".format(1), "Loss", "Epoch",
                  Headers=["val_loss", "loss"], markersize=3, linestyle="solid", color=["red", "blue"], legend=True,
                  legend_labels=["val_loss", "loss"], title="Training and Validation Plot Model {}".format(i), save_fig=True)
       run_plots()
