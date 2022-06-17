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

#Adam figures
def create_Adam_figures(base_path, metrics_df, min_threshold_list=[11.1, 13.7]):
    smoothing = 4

    for min_threshold in min_threshold_list:
        cols_filter = (metrics_df['optimizer']=='Adam') & \
                        (metrics_df['bsz']==128) & \
                        (metrics_df['iter']<150000) & \
                        (metrics_df['loss']=="St")

        adam_optimizer_df = metrics_df[cols_filter]
        adam_grouped_df = adam_optimizer_df.groupby('name')['FID'].min().reset_index()
        adam_min_name = adam_grouped_df[adam_grouped_df['FID']<min_threshold]['name'].values
        adam_optimizer_df = adam_optimizer_df[adam_optimizer_df['name'].isin(adam_min_name)]

        adam_rolling_metrics_df = adam_optimizer_df \
                                    .groupby('name')[['FID', 'IS_mean']] \
                                    .rolling(smoothing) \
                                    .mean() \
                                    .reset_index() \
                                    .drop(columns="name") \
                                    .rename(columns={"level_1":"index", "FID":"roll_FID", "IS_mean":"roll_IS_mean"}) \
                                    .dropna()

        adam_optimizer_df = pd.merge(adam_optimizer_df.reset_index(), adam_rolling_metrics_df, on='index')

        adam_optimizer_df['parameters'] = 'A'+ \
                                        ', ηG '+adam_optimizer_df['lrG'].map(lambda x: "{:.0e}".format(x)).astype(str)+ \
                                        ', ηD '+adam_optimizer_df['lrD'].map(lambda x: "{:.0e}".format(x)).astype(str)+ \
                                        ', b '+adam_optimizer_df['beta'].astype(str)

        adam_optimizer_df = adam_optimizer_df.sort_values(by=['optimizer', 'beta', 'lrG', 'lrD'], ascending=True)

        # save FID, IS figures with only top-4
        if min_threshold<13:
            #save FID figure
            save_figure(base_path, df=adam_optimizer_df, y_col='roll_FID', x_col='iter', hue_col='parameters',
                        ylim=(0, 750), fig_name=f'Adam_FID', fig_size=(4,3.5),
                        xlabel="Number of mini-batches", ylabel="FID",
                        fontsize_labels='10', fontsize_legend='9')
            #save IS figure
            save_figure(base_path, df=adam_optimizer_df, y_col='roll_IS_mean', x_col='iter', hue_col='parameters',
                    ylim=(1, 3.75), fig_name=f'Adam_IS', fig_size=(4,3.5),
                    xlabel="Number of mini-batches", ylabel="Inception Score",
                    fontsize_labels='10', fontsize_legend='9')

        else:
        # save FID figure with top-10 parameters
            save_figure(base_path, df=adam_optimizer_df, y_col='roll_FID', x_col='iter', hue_col='parameters',
                        ylim=(0, 750), fig_name=f'TOP10_Adam_FID', fig_size=(10,6),
                        xlabel="Number of mini-batches", ylabel="FID",
                        fontsize_labels='14', fontsize_legend='10')


# ExtraAdam figures
def create_ExtraAdam_figures(base_path, metrics_df, min_threshold_list=[[12, 14.4],[16, 16]]):
    smoothing = 4

    for min_threshold_1, min_threshold_2 in min_threshold_list:

        cols_filter = (metrics_df['optimizer']=='ExtraAdam') & \
                        (metrics_df['bsz']==128) & \
                        (metrics_df['iter']<150000) & \
                        (metrics_df['loss']=="St")

        extra_adam_optimizer_df = metrics_df[cols_filter]
        extra_adam_grouped_df = extra_adam_optimizer_df.groupby('name')[['FID', 'beta']].min('FID').reset_index()

        col_filter = ((extra_adam_grouped_df['FID']<min_threshold_1) & (extra_adam_grouped_df['beta']==0.9)) | \
                        ((extra_adam_grouped_df['FID']<min_threshold_2) & (extra_adam_grouped_df['beta']==0.5))


        extra_adam_min_name = extra_adam_grouped_df[col_filter]['name'].values
        extra_adam_optimizer_df = extra_adam_optimizer_df[extra_adam_optimizer_df['name'].isin(extra_adam_min_name)]

        extra_adam_rolling_metrics_df = extra_adam_optimizer_df \
                                    .groupby('name')[['FID', 'IS_mean']] \
                                    .rolling(smoothing) \
                                    .mean() \
                                    .reset_index() \
                                    .drop(columns="name") \
                                    .rename(columns={"level_1":"index", "FID":"roll_FID", "IS_mean":"roll_IS_mean"}) \
                                    .dropna()

        extra_adam_optimizer_df = pd.merge(extra_adam_optimizer_df.reset_index(), extra_adam_rolling_metrics_df, on='index')

        extra_adam_optimizer_df['parameters'] = 'E-A'+ \
                                        ', ηG '+extra_adam_optimizer_df['lrG'].map(lambda x: "{:.0e}".format(x)).astype(str)+ \
                                        ', ηD '+extra_adam_optimizer_df['lrD'].map(lambda x: "{:.0e}".format(x)).astype(str)+ \
                                        ', b '+extra_adam_optimizer_df['beta'].astype(str)

        extra_adam_optimizer_df = extra_adam_optimizer_df.sort_values(by=['optimizer', 'beta', 'lrG', 'lrD'], ascending=True)


        # save FID, IS figures with only top-4
        if min_threshold_1<16:
            #save FID figure
            save_figure(base_path, df=extra_adam_optimizer_df, y_col='roll_FID', x_col='iter', hue_col='parameters',
                        ylim=(0, 750), fig_name=f'ExtraAdam_FID', fig_size=(4,3.5),
                        xlabel="Number of mini-batches", ylabel="FID",
                        fontsize_labels='10', fontsize_legend='9')
            #save IS figure
            save_figure(base_path, df=extra_adam_optimizer_df, y_col='roll_IS_mean', x_col='iter', hue_col='parameters',
                    ylim=(1, 3.75), fig_name=f'ExtraAdam_IS', fig_size=(4,3.5),
                    xlabel="Number of mini-batches", ylabel="Inception Score",
                    fontsize_labels='10', fontsize_legend='9')

        else:
        # save FID figure with top-10 parameters
            save_figure(base_path, df=extra_adam_optimizer_df, y_col='roll_FID', x_col='iter', hue_col='parameters',
                        ylim=(0, 800), fig_name=f'TOP10_ExtraAdam_FID', fig_size=(10,6),
                        xlabel="Number of mini-batches", ylabel="FID",
                        fontsize_labels='14', fontsize_legend='10')


# Loss figures
def create_loss_figures(base_path, metrics_df, beta_list=[0.5, 0.9]):

    for beta in beta_list:
        cols_filter = (metrics_df['optimizer']=='Adam') & \
                        (metrics_df['bsz']==128) & \
                        (metrics_df['beta']==beta) & \
                        (metrics_df['lrD']==0.0001) & \
                        (metrics_df['iter']<150000)

        loss_df = metrics_df[cols_filter]
        loss_df["loss"] = loss_df["loss"].replace({"St": "Standard"})

        loss_df['parameters'] = "A"+ \
                                        ', ηG '+loss_df['lrG'].map(lambda x: "{:.0e}".format(x)).astype(str)+ \
                                        ', ηD '+loss_df['lrD'].map(lambda x: "{:.0e}".format(x)).astype(str)+ \
                                        ', b '+loss_df['beta'].astype(str)+ \
                                        ', bsz '+loss_df['bsz'].astype(str)+ \
                                        ', loss '+loss_df['loss'].astype(str)

        loss_df = loss_df.sort_values(by=['optimizer','lrG', 'lrD'], ascending=True)

        figure_name = 'wgangp_vs_standard_Adam_FID' if beta==0.9 else 'wgangp_vs_standard_Adam_b05_FID'
        save_figure(base_path, df=loss_df, y_col='FID', x_col='iter', hue_col='parameters',
            ylim=(0, 700), fig_name=figure_name, fig_size=(10,6),
            xlabel="Number of mini-batches", ylabel="FID",
            fontsize_labels='14', fontsize_legend='10')


# create batch size figures
def tumbling_avg(df, max_bsz):
    cur_bsz = df['bsz'].unique()[0]
    num_of_batches = int(max_bsz/cur_bsz)

    count=0
    sum_fid=0
    df['avg_FID'] = -1

    for index, row in df.iterrows():
        count+=1
        sum_fid += row["FID"]
        if count==num_of_batches:
            df.loc[index,'avg_FID'] = sum_fid/num_of_batches
            count=0
            sum_fid=0

    return df

def create_batch_size_figures(base_path, metrics_df):

    cols_filter = (metrics_df['optimizer']=='ExtraAdam')& \
                    (metrics_df['lrD']==0.0002) & \
                    (metrics_df['lrG']==0.0005) & \
                    (metrics_df['beta']==0.5) & \
                    (metrics_df['loss']=="St")

    batchsize_df = metrics_df[cols_filter]
    batchsize_df = batchsize_df.sort_values(by=['iter'], ascending=True)

    scaled_FID_df = batchsize_df[batchsize_df['bsz']!=1024]
    scaled_FID_df = scaled_FID_df.groupby('name').apply(lambda df: tumbling_avg(df, 1024))
    scaled_FID_df = scaled_FID_df[scaled_FID_df["avg_FID"]!=-1]

    max_bsz_df = batchsize_df[batchsize_df['bsz']==1024]
    max_bsz_df['avg_FID'] = max_bsz_df['FID']

    batchsize_df = pd.concat([scaled_FID_df, max_bsz_df])
    min_iter = min(batchsize_df.groupby('name').size().values)
    batchsize_df = batchsize_df.groupby('name').head(min_iter)

    bsz_1024_iters = list(batchsize_df[batchsize_df["bsz"]==1024]["iter"].values)

    new_iter_list = []
    for i in range(5):
        new_iter_list += bsz_1024_iters

    batchsize_df = batchsize_df.sort_values(by=['name', 'iter'], ascending=True).assign(new_iter = new_iter_list)
    batchsize_df['parameters'] = "E-A"+ \
                                    ', ηG '+batchsize_df['lrG'].map(lambda x: "{:.0e}".format(x)).astype(str)+ \
                                    ', ηD '+batchsize_df['lrD'].map(lambda x: "{:.0e}".format(x)).astype(str)+ \
                                    ', b '+batchsize_df['beta'].astype(str)+ \
                                    ', bsz '+batchsize_df['bsz'].astype(str)

    save_figure(base_path, df=batchsize_df.sort_values(by=['bsz'], ascending=False), y_col='FID', x_col='new_iter', hue_col='parameters',
        ylim=(0, 600), fig_name='batch_size_comparison_ExtraAdam_FID', fig_size=(10,6),
        xlabel="Number of mini-batches of size 1024", ylabel="FID",
        fontsize_labels='14', fontsize_legend='10')


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
    print("Create SGD_ESGD figures...")
    create_SGD_ESGD_figures(args.figs_output, fid_df, optimizer_name='SGD', min_threshold_list=[310, 400], figure_name='SGD', is_ylim=(1, 1.14))

    #create Extra SGD figures
    print("Create SGD_ESGD figures...")
    create_SGD_ESGD_figures(args.figs_output, fid_df, optimizer_name='ExtraSGD', min_threshold_list=[315, 400], figure_name='Extra_SGD', is_ylim=(1, 1.23))

    #create ExtraAdam figures
    print("Create ExtraAdam figures...")
    create_ExtraAdam_figures(args.figs_output, fid_df)

    #create ADAM figures
    print("Create Adam figures...")
    create_Adam_figures(args.figs_output, fid_df)

    # Create batch size figures
    print("Create batch_size figures...")
    create_batch_size_figures(args.figs_output, fid_df)

    # Create loss figures
    print("Create loss figures...")
    create_loss_figures(args.figs_output, fid_df)


