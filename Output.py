"""
TESI DI LAUREA IN INFORMATICA PRESSO L'UNIVERSITÀ DI PISA
@Author: Giuseppe Muschetta
@Thesis_Supervisor: Professor Riccardo Guidotti
@Libraries: Pandas 1.4.4
            MatPlotLib 3.5.2
"""

import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import time


def _max_depth_varies(tables, graphs, couple, tree_type, criterion, mode, bins, mss, msl):

    # leggo il nome dei file .csv appartenenti allo stesso dataset su cui andrò ad effettuare l'analisi
    mixed_csv = couple[0]
    sklearn_csv = couple[1]

    # dataset name:
    temp_ds_name = re.split('/', couple[0])[1]
    ds_name = re.split('_', temp_ds_name)[0]

    # distinguo i due alberi
    if tree_type == 'MixedDecisionTree':
        df = pd.read_csv(mixed_csv)
    elif tree_type == 'SklearnDT':
        df = pd.read_csv(sklearn_csv)
    else:
        raise Exception("Wrong Decision Tree name...")
    df.insert(0, 'DATASET', ds_name)
    df.insert(1, 'DT_TYPE', tree_type)

    # qui parte la query
    if mode != 'classic':
        df_queried = df[(df['Tree_Mode'] == mode) & (df['Criterion'] == criterion)
                        & (df['Bins'] == bins) & (df['Min_sample_split'] == mss) &
                        (df['Min_sample_leaf'] == msl)]
    else:
        if bins != 20:
            return
        else:
            bins = 0
            df_queried = df[(df['Tree_Mode'] == mode) & (df['Criterion'] == criterion)
                            & (df['Bins'] == bins) & (df['Min_sample_split'] == mss) &
                            (df['Min_sample_leaf'] == msl)]
        # end_if
    # end_if

    # PARTE RELATIVA ALLE TABELLE:
    if tables:
        # creo le directories in cui salvare le tabelle:
        dir_path = 'Tables/' + ds_name + '/'
        try:
            os.makedirs(dir_path, exist_ok=True)
        except OSError as error:
            print(error)
            print("Path '%s' can not be created" % dir_path)

        table_path = dir_path + 'max_depth_varies.csv'
        df_queried.to_csv(table_path, index=False, header=True, mode='a')
    # end_if

    # PARTE RELATIVA AI GRAFICI:
    if graphs:
        # creo le directories in cui salvare i grafici:
        dir_path = 'Graphs/' + ds_name + '/' + 'max_depth_varies/'
        try:
            os.makedirs(dir_path, exist_ok=True)
        except OSError as error:
            print(error)
            print("Path '%s' can not be created" % dir_path)

        metrics_list = ['ACCURACY', 'F1_SCORE']
        for metric in metrics_list:
            graph_path = dir_path + tree_type + '_' + metric + '_' + criterion + '_' + mode + '_' \
                         + str(bins) + '_' + str(mss) + '_' + str(msl) + '.png'

            x = df_queried['Max_Depth']
            y = df_queried[metric]

            plt.plot(x, y, label=tree_type, marker='o')
            plt.xlabel('max_depth')
            plt.ylabel(metric.lower())
            plt.savefig(graph_path)
            plt.close()
    # end_if

    return


def _msl_varies(tables, graphs, couple, tree_type, criterion, mode, bins, max_depth, mss):

    # leggo il nome dei file .csv appartenenti allo stesso dataset su cui andrò ad effettuare l'analisi
    mixed_csv = couple[0]
    sklearn_csv = couple[1]

    # dataset name:
    temp_ds_name = re.split('/', couple[0])[1]
    ds_name = re.split('_', temp_ds_name)[0]

    # distinguo i due alberi
    if tree_type == 'MixedDecisionTree':
        df = pd.read_csv(mixed_csv)
    elif tree_type == 'SklearnDT':
        df = pd.read_csv(sklearn_csv)
    else:
        raise Exception("Wrong Decision Tree name...")
    df.insert(0, 'DATASET', ds_name)
    df.insert(1, 'DT_TYPE', tree_type)

    # qui parte la query
    if mode != 'classic':
        df_queried = df[(df['Min_sample_split'] > df['Min_sample_leaf']) &
                        (df['Tree_Mode'] == mode) & (df['Criterion'] == criterion)
                        & (df['Bins'] == bins) & (df['Max_Depth'] == max_depth)
                        & (df['Min_sample_split'] == mss)]
    else:
        if bins != 20:
            return
        else:
            bins = 0
            df_queried = df[(df['Min_sample_split'] > df['Min_sample_leaf']) &
                            (df['Tree_Mode'] == mode) & (df['Criterion'] == criterion)
                            & (df['Bins'] == bins) & (df['Max_Depth'] == max_depth)
                            & (df['Min_sample_split'] == mss)]
        # end_if
    # end_if

    # PARTE RELATIVA ALLE TABELLE:
    if tables:
        # creo le directories in cui salvare le tabelle:
        dir_path = 'Tables/' + ds_name + '/'
        try:
            os.makedirs(dir_path, exist_ok=True)
        except OSError as error:
            print(error)
            print("Path '%s' can not be created" % dir_path)

        table_path = dir_path + 'min_sample_leaf_varies.csv'
        df_queried.to_csv(table_path, index=False, header=True, mode='a')
    # end_if

    # PARTE RELATIVA AI GRAFICI:
    if graphs:
        # creo le directories in cui andare a salvare i grafici:
        dir_path = 'Graphs/' + ds_name + '/' + 'min_sample_leaf_varies/'
        try:
            os.makedirs(dir_path, exist_ok=True)
        except OSError as error:
            print(error)
            print("Path '%s' can not be created" % dir_path)

        metrics_list = ['ACCURACY', 'F1_SCORE']
        for metric in metrics_list:
            graph_path = dir_path + tree_type + '_' + metric + '_' + criterion + '_' + mode + '_' \
                         + str(bins) + '_' + str(max_depth) + '_' + str(mss) + '.png'

            x = df_queried['Min_sample_leaf']
            y = df_queried[metric]

            plt.plot(x, y, label=tree_type, marker='o')
            plt.xlabel('min_sample_leaf')
            plt.ylabel(metric.lower())
            plt.savefig(graph_path)
            plt.close()
    # end_if

    return


def _mss_varies(tables, graphs, couple, tree_type, criterion, mode, bins, max_depth, msl):

    # leggo il nome dei file .csv appartenenti allo stesso dataset su cui andrò ad effettuare l'analisi
    mixed_csv = couple[0]
    sklearn_csv = couple[1]

    # dataset name:
    temp_ds_name = re.split('/', couple[0])[1]
    ds_name = re.split('_', temp_ds_name)[0]

    # distinguo i due alberi
    if tree_type == 'MixedDecisionTree':
        df = pd.read_csv(mixed_csv)
    elif tree_type == 'SklearnDT':
        df = pd.read_csv(sklearn_csv)
    else:
        raise Exception("Wrong Decision Tree name...")
    df.insert(0, 'DATASET', ds_name)
    df.insert(1, 'DT_TYPE', tree_type)

    # qui parte la query
    if mode != 'classic':
        df_queried = df[(df['Min_sample_split'] > df['Min_sample_leaf']) &
                        (df['Tree_Mode'] == mode) & (df['Criterion'] == criterion)
                        & (df['Bins'] == bins) & (df['Max_Depth'] == max_depth)
                        & (df['Min_sample_leaf'] == msl)]
    else:
        if bins != 20:
            return
        else:
            bins = 0
            df_queried = df[(df['Min_sample_split'] > df['Min_sample_leaf']) &
                            (df['Tree_Mode'] == mode) & (df['Criterion'] == criterion)
                            & (df['Bins'] == bins) & (df['Max_Depth'] == max_depth)
                            & (df['Min_sample_leaf'] == msl)]
        # end_if
    # end_if

    # PARTE RELATIVA ALLE TABELLE:
    if tables:
        dir_path = 'Tables/' + ds_name + '/'
        try:
            os.makedirs(dir_path, exist_ok=True)
        except OSError as error:
            print(error)
            print("Path '%s' can not be created" % dir_path)

        table_path = dir_path + 'min_sample_split_varies.csv'
        df_queried.to_csv(table_path, index=False, header=True, mode='a')
    # end_if

    # PARTE RELATIVA AI GRAFICI:
    if graphs:
        # creo le directories in cui andare a salvare i grafici:
        dir_path = 'Graphs/' + ds_name + '/' + 'min_sample_split_varies/'
        try:
            os.makedirs(dir_path, exist_ok=True)
        except OSError as error:
            print(error)
            print("Path '%s' can not be created" % dir_path)

        metrics_list = ['ACCURACY', 'F1_SCORE']
        for metric in metrics_list:
            graph_path = dir_path + tree_type + '_' + metric + '_' + criterion + '_' + mode + '_' \
                         + str(bins) + '_' + str(max_depth) + '_' + str(msl) + '.png'

            x = df_queried['Min_sample_split']
            y = df_queried[metric]

            plt.plot(x, y, label=tree_type, marker='o')
            plt.xlabel('min_sample_split')
            plt.ylabel(metric.lower())
            plt.savefig(graph_path)
            plt.close()
    # end_if

    return


def max_depth_varies(tables, graphs, couples_list, tree_type_list, criterion_list, tree_mode_list, bins_list,
                     mss_list, msl_list):

    for couple in couples_list:
        for criterion in criterion_list:
            for mode in tree_mode_list:
                for bins in bins_list:
                    for mss in mss_list:
                        for msl in msl_list:
                            for tree_type in tree_type_list:
                                if mss > msl:
                                    _max_depth_varies(tables, graphs, couple, tree_type, criterion,
                                                      mode, bins, mss, msl)
    return


def msl_varies(tables, graphs, couples_list, tree_type_list, criterion_list, tree_mode_list, bins_list,
               max_depth_list, mss_list):

    for couple in couples_list:
        for criterion in criterion_list:
            for mode in tree_mode_list:
                for bins in bins_list:
                    for max_depth in max_depth_list:
                        for mss in mss_list:
                            for tree_type in tree_type_list:
                                if mss < 10:
                                    continue
                                else:
                                    _msl_varies(tables, graphs, couple, tree_type, criterion,
                                                mode, bins, max_depth, mss)
    return


def mss_varies(tables, graphs, couples_list, tree_type_list, criterion_list, tree_mode_list, bins_list,
               max_depth_list, msl_list):

    for couple in couples_list:
        for criterion in criterion_list:
            for mode in tree_mode_list:
                for bins in bins_list:
                    for max_depth in max_depth_list:
                        for msl in msl_list:
                            for tree_type in tree_type_list:
                                if msl > 5:
                                    continue
                                else:
                                    _mss_varies(tables, graphs, couple, tree_type,
                                                criterion, mode, bins, max_depth, msl)
    return


def main():
    # tramite questi due booleani decido l'ouput che voglio generare:
    tables = False
    graphs = False
    if tables:
        print("- Verranno generate tabelle")
    if graphs:
        print("- Verrano generati grafici")
    if not tables and not graphs:
        print("- Non verrà generato alcun output, termino l'esecuzione.")
        return

    folder_name = 'Results/'
    #                 prima coppia
    couples_list = [(folder_name + 'Iris_Mixed_DT_result.csv',
                     folder_name + 'Iris_Sklearn_DT_result.csv'),
                    # seconda coppia
                    (folder_name + 'breast-cancer_Mixed_DT_result.csv',
                     folder_name + 'breast-cancer_Sklearn_DT_result.csv'),
                    # terza coppia
                    (folder_name + 'Carseats_Mixed_DT_result.csv',
                     folder_name + 'Carseats_Sklearn_DT_result.csv'),
                    # quarta coppia
                    (folder_name + 'diabetes_Mixed_DT_result.csv',
                     folder_name + 'diabetes_Sklearn_DT_result.csv'),
                    # quinta coppia
                    (folder_name + 'HouseVotes84_Mixed_DT_result.csv',
                     folder_name + 'HouseVotes84_Sklearn_DT_result.csv'),
                    # sesta coppia
                    (folder_name + 'prostate_Mixed_DT_result.csv',
                     folder_name + 'prostate_Sklearn_DT_result.csv'),
                    # settima coppia
                    (folder_name + 'parkinsons_Mixed_DT_result.csv',
                     folder_name + 'parkinsons_Sklearn_DT_result.csv'),
                    # ottava coppia
                    (folder_name + 'Sacramento_Mixed_DT_result.csv',
                     folder_name + 'Sacramento_Sklearn_DT_result.csv'),
                    # nona coppia
                    (folder_name + 'Auto_Mixed_DT_result.csv',
                     folder_name + 'Auto_Sklearn_DT_result.csv'),
                    # decima coppia
                    (folder_name + 'churn_Mixed_DT_result.csv',
                     folder_name + 'churn_Sklearn_DT_result.csv')]

    tree_type_list = ['MixedDecisionTree', 'SklearnDT']
    tree_mode_list = ['classic', 'median', 'binned']
    criterion_list = ['entropy', 'gini']
    bins_list = [20, 11]  # [20,15,11,7] ma ho sovuto accorciarla
    max_depth_list = [2, 3, 4, 5, 6, 7, 8, 25]  # [2,3,4,5,6,7,8,25] ho tolto il valore 50
    mss_list = [2, 3, 5, 10, 25, 50]  # min_sample_split list
    msl_list = [1, 3, 5, 10, 25, 50]  # min_sample_leaf list

    # calcoliamo il tempo di esecuzione:
    start = time.time()
    max_depth_varies(tables, graphs, couples_list, tree_type_list, criterion_list,
                     tree_mode_list, bins_list, mss_list, msl_list)
    msl_varies(tables, graphs, couples_list, tree_type_list, criterion_list,
               tree_mode_list, bins_list, max_depth_list, mss_list)
    mss_varies(tables, graphs, couples_list, tree_type_list, criterion_list,
               tree_mode_list, bins_list, max_depth_list, msl_list)
    end = time.time()
    elapsed_time = end - start

    if elapsed_time >= 60:
        print("Execution time =", elapsed_time / 60, "minutes")
    else:
        print("Execution time =", elapsed_time, "seconds")
    # end_if

    return


if __name__ == '__main__':
    main()
