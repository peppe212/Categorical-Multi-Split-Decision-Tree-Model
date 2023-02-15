"""
TESI DI LAUREA IN INFORMATICA PRESSO L'UNIVERSITÀ DI PISA
@Author: Giuseppe Muschetta
@Thesis_Supervisor: Prof. Riccardo Guidotti
@Libraries: Pandas 1.4.4
            MatPlotLib 3.5.2
"""

import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import time


def _max_depth_varies(couple, criterion, _bins, mss, msl):
    # leggo il nome dei file .csv appartenenti allo stesso dataset su cui andrò ad effettuare l'analisi
    mixed_csv = couple[0]
    sklearn_csv = couple[1]

    # dataset name:
    temp_ds_name = re.split('/', couple[0])[1]
    ds_name = re.split('_', temp_ds_name)[0]

    # distinguo i due alberi:
    df_mdt = pd.read_csv(mixed_csv)
    df_sk = pd.read_csv(sklearn_csv)

    # parto con le queries:
    # si scrive più volte sullo stesso plot in modo da sovrapporre le linee del grafico
    tree_mode_list = ['classic', 'median', 'binned']
    for mode in tree_mode_list:
        if mode == 'classic':
            bins = 0
        else:
            bins = _bins
        # end_if
        df_mix_queried = df_mdt[(df_mdt['Tree_Mode'] == mode) & (df_mdt['Criterion'] == criterion)
                                & (df_mdt['Bins'] == bins) & (df_mdt['Min_sample_split'] == mss) &
                                (df_mdt['Min_sample_leaf'] == msl)]

        df_sk_queried = df_sk[(df_sk['Tree_Mode'] == mode) & (df_sk['Criterion'] == criterion)
                              & (df_sk['Bins'] == bins) & (df_sk['Min_sample_split'] == mss) &
                              (df_sk['Min_sample_leaf'] == msl)]

        x = df_mix_queried['Max_Depth']
        y = df_mix_queried['F1_SCORE']
        plt.plot(x, y, label='mix-%s' % mode, marker='o')

        x = df_sk_queried['Max_Depth']
        y = df_sk_queried['F1_SCORE']
        plt.plot(x, y, label='skl-%s' % mode, marker='s')
    # end_for

    dir_path = 'Overlaps_F1_score/' + ds_name + '/' + 'max_depth_varies/'
    try:
        os.makedirs(dir_path, exist_ok=True)
    except OSError as error:
        print(error)
        print("Path '%s' can not be created" % dir_path)

    graph_path = dir_path + 'F1_SCORE' + '_' + criterion + '_' + str(_bins) \
                 + '_' + str(mss) + '_' + str(msl) + '.png'

    # qui finalmente creo i grafici sovrapposti (overlapped)
    plt.xlabel('max_depth')
    plt.ylabel('F1_SCORE'.lower())
    plt.legend()
    plt.savefig(graph_path)
    plt.close()

    return


def _mss_varies(couple, criterion, _bins, max_depth, msl):
    # leggo il nome dei file .csv appartenenti allo stesso dataset su cui andrò ad effettuare l'analisi
    mixed_csv = couple[0]
    sklearn_csv = couple[1]

    # dataset name:
    temp_ds_name = re.split('/', couple[0])[1]
    ds_name = re.split('_', temp_ds_name)[0]

    # distinguo i due alberi
    df_mdt = pd.read_csv(mixed_csv)
    df_sk = pd.read_csv(sklearn_csv)

    # parto con le queries:
    # si scrive più volte sullo stesso plot in modo da sovrapporre le linee del grafico
    tree_mode_list = ['classic', 'median', 'binned']
    for mode in tree_mode_list:
        if mode == 'classic':
            bins = 0
        else:
            bins = _bins
        # end_if
        df_mix_queried = df_mdt[(df_mdt['Tree_Mode'] == mode) & (df_mdt['Criterion'] == criterion)
                                & (df_mdt['Bins'] == bins) & (df_mdt['Max_Depth'] == max_depth) &
                                (df_mdt['Min_sample_leaf'] == msl)]

        df_sk_queried = df_sk[(df_sk['Tree_Mode'] == mode) & (df_sk['Criterion'] == criterion)
                              & (df_sk['Bins'] == bins) & (df_sk['Max_Depth'] == max_depth) &
                              (df_sk['Min_sample_leaf'] == msl)]

        x = df_mix_queried['Min_sample_split']
        y = df_mix_queried['F1_SCORE']
        plt.plot(x, y, label='mix-%s' % mode, marker='o')

        x = df_sk_queried['Min_sample_split']
        y = df_sk_queried['F1_SCORE']
        plt.plot(x, y, label='skl-%s' % mode, marker='s')
    # end_for

    dir_path = 'Overlaps_F1_score/' + ds_name + '/' + 'min_sample_split_varies/'
    try:
        os.makedirs(dir_path, exist_ok=True)
    except OSError as error:
        print(error)
        print("Path '%s' can not be created" % dir_path)

    graph_path = dir_path + 'F1_SCORE' + '_' + criterion + '_' + str(_bins) \
                 + '_' + str(max_depth) + '_' + str(msl) + '.png'

    # qui finalmente creo i grafici sovrapposti (overlapped)
    plt.xlabel('min_sample_split')
    plt.ylabel('F1_SCORE'.lower())
    plt.legend()
    plt.savefig(graph_path)
    plt.close()

    return


def _msl_varies(couple, criterion, _bins, max_depth, mss):
    # leggo il nome dei file .csv appartenenti allo stesso dataset su cui andrò ad effettuare l'analisi
    mixed_csv = couple[0]
    sklearn_csv = couple[1]

    # dataset name:
    temp_ds_name = re.split('/', couple[0])[1]
    ds_name = re.split('_', temp_ds_name)[0]

    # distinguo i due alberi
    df_mdt = pd.read_csv(mixed_csv)
    df_sk = pd.read_csv(sklearn_csv)

    # parto con le queries:
    # si scrive più volte sullo stesso plot in modo da sovrapporre le linee del grafico
    tree_mode_list = ['classic', 'median', 'binned']
    for mode in tree_mode_list:
        if mode == 'classic':
            bins = 0
        else:
            bins = _bins
        # end_if
        df_mix_queried = df_mdt[(df_mdt['Tree_Mode'] == mode) & (df_mdt['Criterion'] == criterion)
                                & (df_mdt['Bins'] == bins) & (df_mdt['Max_Depth'] == max_depth) &
                                (df_mdt['Min_sample_split'] == mss)]

        df_sk_queried = df_sk[(df_sk['Tree_Mode'] == mode) & (df_sk['Criterion'] == criterion)
                              & (df_sk['Bins'] == bins) & (df_sk['Max_Depth'] == max_depth) &
                              (df_sk['Min_sample_split'] == mss)]

        x = df_mix_queried['Min_sample_leaf']
        y = df_mix_queried['F1_SCORE']
        plt.plot(x, y, label='mix-%s' % mode, marker='o')

        x = df_sk_queried['Min_sample_leaf']
        y = df_sk_queried['F1_SCORE']
        plt.plot(x, y, label='skl-%s' % mode, marker='s')
    # end_for

    dir_path = 'Overlaps_F1_score/' + ds_name + '/' + 'min_sample_leaf_varies/'
    try:
        os.makedirs(dir_path, exist_ok=True)
    except OSError as error:
        print(error)
        print("Path '%s' can not be created" % dir_path)

    graph_path = dir_path + 'F1_SCORE' + '_' + criterion + '_' + str(_bins) \
                 + '_' + str(max_depth) + '_' + str(mss) + '.png'

    # qui finalmente creo i grafici sovrapposti (overlapped)
    plt.xlabel('min_sample_leaf')
    plt.ylabel('F1_SCORE'.lower())
    plt.legend()
    plt.savefig(graph_path)
    plt.close()

    return


def max_depth_varies(couples_list, criterion_list, bins_list, mss_list, msl_list):
    for couple in couples_list:
        for criterion in criterion_list:
            for bins in bins_list:
                for mss in mss_list:
                    for msl in msl_list:
                        if mss > msl:
                            _max_depth_varies(couple, criterion, bins, mss, msl)
    return


def mss_varies(couples_list, criterion_list, bins_list, max_depth_list, msl_list):
    for couple in couples_list:
        for criterion in criterion_list:
            for bins in bins_list:
                for max_depth in max_depth_list:
                    for msl in msl_list:
                        if msl > 5:
                            continue
                        else:
                            _mss_varies(couple, criterion, bins, max_depth, msl)
    return


def msl_varies(couples_list, criterion_list, bins_list, max_depth_list, mss_list):
    for couple in couples_list:
        for criterion in criterion_list:
            for bins in bins_list:
                for max_depth in max_depth_list:
                    for mss in mss_list:
                        if mss < 10:
                            continue
                        else:
                            _msl_varies(couple, criterion, bins, max_depth, mss)
    return


def main():
    # imposto il nome della cartella da cui leggere le tabelle:
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

    criterion_list = ['entropy', 'gini']
    bins_list = [20, 11]
    max_depth_list = [2, 3, 4, 5, 6, 7, 8, 25, 50]  # Tree's max depth list
    mss_list = [2, 3, 5, 10, 25, 50]  # min_sample_split list
    msl_list = [1, 3, 5, 10, 25, 50]  # min_sample_leaf list

    # calcoliamo il tempo di esecuzione:
    start = time.time()
    max_depth_varies(couples_list, criterion_list, bins_list, mss_list, msl_list)
    mss_varies(couples_list, criterion_list, bins_list, max_depth_list, msl_list)
    msl_varies(couples_list, criterion_list, bins_list, max_depth_list, mss_list)
    end = time.time()
    elapsed_time = end - start

    if elapsed_time >= 60:
        print("Execution time =", elapsed_time / 60, "minutes")
    else:
        print("Execution time =", elapsed_time, "seconds")
    return

# il problema al momento è che non mi plotta la modalità classica ne del mixed e ne dell'sklearn tree


if __name__ == '__main__':
    main()
