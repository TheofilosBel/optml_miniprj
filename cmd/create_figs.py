import argparse
import pandas as pd
import os.path as op
import os
import matplotlib.pyplot as plt
import seaborn as sns

def save_figure(base_path, df, y_col, x_col, hue_col, ylim, fig_name, fig_size=(4,3.5), xlabel="Number of mini-batches", ylabel="FID", fontsize_labels='10', fontsize_legend='9'):

    plt.figure(figsize=fig_size)
    ax0 = sns.lineplot(data=df, y=y_col, x=x_col, hue=hue_col)
    ax0.legend_.set_title(None)
    ax0.legend(loc='upper right')
    ax0.set(ylim=ylim)
    ax0.set_xlabel(xlabel, fontsize=fontsize_labels)
    ax0.set_ylabel(ylabel, fontsize=fontsize_labels)
    ax0.ticklabel_format(style='sci', scilimits=(0,0), axis='x')
    plt.setp(ax0.get_legend().get_texts(), fontsize=fontsize_legend)
    plt.savefig(op.join(base_path, fig_name + ".png"), bbox_inches='tight', dpi=100)
    plt.savefig(op.join(base_path, fig_name + ".eps"), bbox_inches='tight', format='eps')

def create_SGD_ESGD_figures(base_path, metrics_df, optimizer_name, min_threshold_list, figure_name, is_ylim):
    smoothing = 6
    for min_threshold in min_threshold_list:

        cols_filter = (metrics_df['optimizer']==optimizer_name) & \
                        (metrics_df['bsz']==128) & \
                        (metrics_df['iter']<150000) & \
                        (metrics_df['loss']=="St")

        sgd_optimizer_df = metrics_df[cols_filter]
        sgd_grouped_df = sgd_optimizer_df.groupby('name')['FID'].min().reset_index()
        sgd_min_name = sgd_grouped_df[sgd_grouped_df['FID']<min_threshold]['name'].values
        sgd_optimizer_df = sgd_optimizer_df[sgd_optimizer_df['name'].isin(sgd_min_name)]

        sgd_rolling_metrics_df = sgd_optimizer_df \
                                    .groupby('name')[['FID', 'IS_mean']] \
                                    .rolling(smoothing) \
                                    .mean() \
                                    .reset_index() \
                                    .drop(columns="name") \
                                    .rename(columns={"level_1":"index", "FID":"roll_FID", "IS_mean":"roll_IS_mean"}) \
                                    .dropna()

        sgd_optimizer_df = pd.merge(sgd_optimizer_df.reset_index(), sgd_rolling_metrics_df, on='index')
        opt_name_abbr = 'SGD' if optimizer_name == 'SGD' else 'E-SGD'
        sgd_optimizer_df['parameters'] = opt_name_abbr+ \
                                        ', ηG '+sgd_optimizer_df['lrG'].map(lambda x: "{:.0e}".format(x)).astype(str)+ \
                                        ', ηD '+sgd_optimizer_df['lrD'].map(lambda x: "{:.0e}".format(x)).astype(str)

        sgd_optimizer_df = sgd_optimizer_df.sort_values(by=['optimizer', 'lrG', 'lrD'], ascending=True)

        # save FID, IS figures with only top-3
        if min_threshold<400:
            #save FID figure
            save_figure(base_path, df=sgd_optimizer_df, y_col='roll_FID', x_col='iter', hue_col='parameters',
                        ylim=(320, 520), fig_name=f'{figure_name}_FID', fig_size=(4,3.5),
                        xlabel="Number of mini-batches", ylabel="FID",
                        fontsize_labels='10', fontsize_legend='9')
            #save IS figure
            save_figure(base_path, df=sgd_optimizer_df, y_col='roll_IS_mean', x_col='iter', hue_col='parameters',
                    ylim=is_ylim, fig_name=f'{figure_name}_IS', fig_size=(4,3.5),
                    xlabel="Number of mini-batches", ylabel="Inception Score",
                    fontsize_labels='10', fontsize_legend='9')

        if min_threshold>=400:
        # save FID figure with all parameters
            save_figure(base_path, df=sgd_optimizer_df, y_col='roll_FID', x_col='iter', hue_col='parameters',
                        ylim=(320, 520), fig_name=f'ALL_{figure_name}_FID', fig_size=(10,6),
                        xlabel="Number of mini-batches", ylabel="FID",
                        fontsize_labels='14', fontsize_legend='10')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss_csv", required=True, type=str, help="The location of the loss csv generated from the run-logs")
    parser.add_argument("--fid_csv", required=True, type=str, help="The location of the fid csv generated from the run-logs")
    parser.add_argument("--figs_output", required=True, type=str, help="The path to place the generated graphs")
    args = parser.parse_args()

    # Create the output dir
    if not op.isdir(args.figs_output):
        print("[INFO] Argument 'figs_output' doesn't exist, we will create it")
        os.mkdir(args.figs_output)

    # Load the csv's
    loss_df = pd.read_csv(args.loss_csv)
    fid_df = pd.read_csv(args.fid_csv)

    # Create the figures
    #create SGD figures
    create_SGD_ESGD_figures(args.figs_output, fid_df, optimizer_name='SGD', min_threshold_list=[310, 400], figure_name='SGD', is_ylim=(1, 1.14))

    #create Extra SGD figures
    create_SGD_ESGD_figures(args.figs_output, fid_df, optimizer_name='ExtraSGD', min_threshold_list=[315, 400], figure_name='Extra_SGD', is_ylim=(1, 1.23))