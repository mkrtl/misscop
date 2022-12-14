import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plot_settings import tex_fonts

save_fig = True

rho = 0.1
beta_0, beta_1 = 0, 2

rho_string = str(rho).replace(".", "")
results_df = pd.read_csv("C:/Users/Q508552/Desktop/Paper/SimulationResults/rho{}shift{}reg{}N1000.csv".format(
    rho_string, beta_0, beta_1))

plt.style.use('seaborn-paper')
plt.rcParams.update(tex_fonts)


def rename_df(df):
    return df.rename({"rho_naive": r"$\rho_{SCOPE}$",
                      "rho_known_marginals": r"$\rho^0$",
                      "rho_two_step": r"$\rho_{EM}$",
                      "test_stat_naive": r"$d^{KS}_{SCOPE}$",
                      "test_stat_two_step": r"$d^{KS}_{EM}$",
                      "test_statistic_known_marginals": r"$d^{KS}_0$"}, axis=1)


def make_ks_box_plot(df, ax=None, size_dots=2, title=None, orient="h"):
    sns.boxplot(data=df[[r"$d^{KS}_{SCOPE}$", r"$d^{KS}_0$", r"$d^{KS}_{EM}$"]], ax=ax, orient=orient)
    sns.swarmplot(data=df[[r"$d^{KS}_{SCOPE}$", r"$d^{KS}_0$", r"$d^{KS}_{EM}$"]],
                  size=size_dots, color="0.25", ax=ax, orient=orient)
    if ax is None:
        plt.title(title)
    else:
        ax.set_title(title)


def make_rho_box_plot(df, curr_rho, ax=None, size_dots=2, title=None, orient="h"):
    sns.boxplot(data=df[[r"$\rho_{SCOPE}$", r"$\rho^0$", r"$\rho_{EM}$"]], ax=ax, orient=orient)
    sns.swarmplot(data=df[[r"$\rho_{SCOPE}$", r"$\rho^0$", r"$\rho_{EM}$"]],
                  size=size_dots,
                  color="0.25",
                  ax=ax, orient=orient)
    if ax is None:
        plt.axhline(y=curr_rho, color="black", linestyle="--")
        plt.title(title)
    else:
        if orient == "h":
            ax.axvline(x=curr_rho, color="black", linestyle="--")
        else:
            ax.axhline(y=curr_rho, color="black", linestyle="--")
        ax.set_title(title)
        ax.axvline
"""
plt.figure(figsize=(20, 5))
plt.title(r"KS test statistics for $\rho = {}$, $\beta_0 = {}$, $\beta_1 = {}$".format(rho, beta_0, beta_1),
          fontsize=16)
plt.ylim(0, 0.4)
results_df = rename_df(results_df)
make_ks_box_plot(results_df)

if save_fig:
    plt.savefig(
        "C:/Users/Q508552/Documents/LaTex/Paper/GaussianCopulaUnderMAR/Plots/"
        "boxplot_kstest_rho{}shift{}reg{}N1000.pdf".format(rho_string, beta_0, beta_1))
plt.show()

plt.figure(figsize=(20, 5))
make_rho_box_plot(results_df, rho, orient="v")
plt.title(r"Estimators for $\rho$ for $\rho = {}$, $\beta_0 = {}$, $\beta_1 = {}$".format(rho, beta_0, beta_1),
          fontsize=16)

if save_fig:
    plt.savefig(
        "C:/Users/Q508552/Documents/LaTex/Paper/GaussianCopulaUnderMAR/"
        "Plots/boxplot_rho_rho{}shift{}reg{}N1000.pdf".format(rho_string, beta_0, beta_1))
plt.show()
"""

rhos = [0.1, 0.1, 0.5, 0.5]
path_pattern = "C:/Users/Q508552/Desktop/Paper/SimulationResults/rho{}shift{}reg{}N1000.csv"
generate_paths = lambda current_rho, alpha_0, alpha_1: path_pattern.format(
    str(current_rho), alpha_0, alpha_1).replace(".", "", 1)

rho_alphas = [(0.1, -2, 1),
              (0.1, 0, 2),
              (0.5, -2, 1),
              (0.5, 0, 2)]


f, axes = plt.subplots(nrows=1, ncols=len(rho_alphas), sharey=True)
for k, param in enumerate(rho_alphas):
    path = generate_paths(*param)
    df = pd.read_csv(path)
    df = rename_df(df)
    title = r"$\rho={},\beta=({}, {})$".format(*param)
    make_rho_box_plot(df, rhos[k], ax=axes[k], size_dots=0.8, title=title, orient="v")
    plt.suptitle(r"Estimators for $\rho$ for Different Settings and Methods")

if save_fig:
    plt.savefig(
        "C:/Users/Q508552/Documents/LaTex/Paper/GaussianCopulaUnderMAR/"
        "Plots/boxplot_rho_in_one.pdf")

plt.show()

f, axes = plt.subplots(nrows=1, ncols=len(rho_alphas), sharey=True)
for k, param in enumerate(rho_alphas):
    path = generate_paths(*param)
    df = pd.read_csv(path)
    df = rename_df(df)
    title = r"$\rho={},\beta=({}, {})$".format(*param)
    make_ks_box_plot(df, ax=axes[k], size_dots=0.5, title=title, orient="v")
    plt.suptitle(r"Kolmogorov-Smirnov Statistic $d^{KS}$ for Different Settings and Methods")

if save_fig:
    plt.savefig(
        "C:/Users/Q508552/Documents/LaTex/Paper/GaussianCopulaUnderMAR/"
        "Plots/boxplot_kstest_in_one.pdf")

plt.show()
