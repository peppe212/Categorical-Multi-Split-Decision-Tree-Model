import pandas as pd
import matplotlib.pyplot as plt
import re


def varia_max_depth(metric, couple, tree_type, mode, criterion, bins, mss, msl):
    # mss deve essere strettamente maggiore di msl
    if mss <= msl:
        return

    # leggo la coppia di dataset di partenza:
    mixed_csv_name = couple[0]
    sklearn_csv_name = couple[1]

    # dataset name:
    temp_ds_name = re.split('/', couple[0])[1]
    ds_name = re.split('_', temp_ds_name)[0] + '_'

    # cartella di output, dove andrò a salvare le tabelle
    folder = 'Graphs/'

    # parte della stringa del file di output, per distinguere meglio i tipi di albero
    mixed_string = 'MDT_' + metric + '_'
    sklearn_string = 'SKL_' + metric + '_'

    if mode == 'classic':
        bins = 0
    # end_if

    if tree_type == 'MixedDecisionTree':
        df = pd.read_csv(mixed_csv_name)
        file_out = ds_name + mixed_string + mode + '_' + criterion + '_bins' + str(bins) \
                   + '_mss' + str(mss) + '_msl' + str(msl) + '.png'
    elif tree_type == 'SklearnDT':
        df = pd.read_csv(sklearn_csv_name)
        file_out = ds_name + sklearn_string + mode + '_' + criterion + '_bins' + str(bins) \
                   + '_mss' + str(mss) + '_msl' + str(msl) + '.png'
    else:
        raise Exception("UNEXPECTED ERROR: valid names are MixedDecisionTree or SklearnDT...")
    # end_if_then_else

    # qui parte la query che ci interessa:
    dft = df[(df['Tree_Mode'] == mode) & (df['Criterion'] == criterion) & (df['Bins'] == bins)
             & (df['Min_sample_split'] == mss) & (df['Min_sample_leaf'] == msl)]

    # dato che qui posso scremare a monte i valori per cui mss > msl,
    # se arrivo a questo punto, allora avrò solo dataframe contenenti 9 righe.
    # assert(len(dft) == 9)
    x = dft['Max_Depth']
    y = dft[metric]

    plt.plot(x, y, label=tree_type, marker='o')
    plt.xlabel('max_depth')
    plt.ylabel(metric.lower())
    plt.savefig(folder + file_out)
    plt.close()
    print(file_out, ' has been created correctly')

    plt.legend()
    plt.show()

    return


def varia_min_sample_split(metric, couple, tree_type, mode, criterion, bins, depth, msl):
    # leggo la coppia di dataset di partenza
    mixed_csv_name = couple[0]
    sklearn_csv_name = couple[1]

    # dataset name:
    temp_ds_name = re.split('/', couple[0])[1]
    ds_name = re.split('_', temp_ds_name)[0] + '_'

    # cartella di output
    folder = 'Graphs/'

    # parte della stringa del file di output, per distinguere meglio i tipi di albero
    mixed_string = 'MDT_' + metric + '_'
    sklearn_string = 'SKL_' + metric + '_'

    if mode == 'classic':
        bins = 0

    if tree_type == 'MixedDecisionTree':
        df = pd.read_csv(mixed_csv_name)
        file_out = ds_name + mixed_string + mode + '_' + criterion + '_bins' + str(bins) \
                   + '_depth' + str(depth) + '_msl' + str(msl) + '.png'
    elif tree_type == 'SklearnDT':
        df = pd.read_csv(sklearn_csv_name)
        file_out = ds_name + sklearn_string + mode + '_' + criterion + '_bins' + str(bins) \
                   + '_depth' + str(depth) + '_msl' + str(msl) + '.png'
    else:
        raise Exception("UNEXPECTED ERROR: valid names are MixedDecisionTree or SklearnDT...")

    dft = df[(df['Tree_Mode'] == mode) & (df['Criterion'] == criterion) & (df['Bins'] == bins)
             & (df['Max_Depth'] == depth) & (df['Min_sample_leaf'] == msl)]

    # se il dataset non è vuoto e, sapendo che mss > msl, mi mantengo almeno una precisione di 3 punti
    # nel far variare min_sample_split tenendo "fermi" depth, msl e il resto
    if len(dft) > 2:
        x = dft['Min_sample_split']
        y = dft[metric]

        plt.plot(x, y, label=tree_type, marker='o')
        plt.xlabel('min_sample_split')
        plt.ylabel(metric.lower())
        plt.savefig(folder + file_out)
        plt.close()
        print(file_out, ' has been created correctly')

        plt.legend()
        plt.show()

    # end_if

    return


def varia_min_sample_leaf(metric, couple, tree_type, mode, criterion, bins, depth, mss):
    # leggo la coppia di dataset di partenza
    mixed_csv_name = couple[0]
    sklearn_csv_name = couple[1]

    # dataset name:
    temp_ds_name = re.split('/', couple[0])[1]
    ds_name = re.split('_', temp_ds_name)[0] + '_'

    # cartella di output
    folder = 'Graphs/'

    # parte della stringa del file di output, per distinguere meglio i tipi di albero
    mixed_string = 'MDT_' + metric + '_'
    sklearn_string = 'SKL_' + metric + '_'

    if mode == 'classic':
        bins = 0

    if tree_type == 'MixedDecisionTree':
        df = pd.read_csv(mixed_csv_name)
        file_out = ds_name + mixed_string + mode + '_' + criterion + '_bins' + str(bins) \
                   + '_depth' + str(depth) + '_mss' + str(mss) + '.png'
    elif tree_type == 'SklearnDT':
        df = pd.read_csv(sklearn_csv_name)
        file_out = ds_name + sklearn_string + mode + '_' + criterion + '_bins' + str(bins) \
                   + '_depth' + str(depth) + '_mss' + str(mss) + '.png'
    else:
        raise Exception("UNEXPECTED ERROR: valid names are MixedDecisionTree or SklearnDT...")

    dft = df[(df['Tree_Mode'] == mode) & (df['Criterion'] == criterion) & (df['Bins'] == bins)
             & (df['Max_Depth'] == depth) & (df['Min_sample_split'] == mss)]

    # se il dataset non è vuoto e, sapendo che mss > msl, mi mantengo almeno una precisione di 3 punti
    # nel far variare min_sample_leaf tenendo "fermi" depth, mss e il resto
    if len(dft) > 2:
        x = dft['Min_sample_leaf']
        y = dft[metric]

        plt.plot(x, y, label=tree_type, marker='o')
        plt.xlabel('min_sample_leaf')
        plt.ylabel(metric.lower())
        plt.savefig(folder + file_out)
        plt.close()
        print(file_out, ' has been created correctly')

        plt.legend()
        plt.show()

    # end_if

    return


def tables(metrics_list, result_couples, tree_types_list, tree_mode_list, criterion_list,
           bins_list, max_depth_list, mss_list, msl_list):
    # faccio variare la max_depth e tengo fissi mss e msl ai valori:
    for metric in metrics_list:
        for couple in result_couples:
            for tree_type in tree_types_list:
                for mode in tree_mode_list:
                    for criterion in criterion_list:
                        for bins in bins_list:
                            for mss in mss_list:
                                for msl in msl_list:
                                    varia_max_depth(metric, couple, tree_type, mode, criterion, bins, mss, msl)
                                # end
                            # end
                        # end
                    # end
                # end
            # end
        # end
    # end_for

    # faccio variare min sample split e tengo fissi max_depth e min sample leaf ai valori:
    for metric in metrics_list:
        for couple in result_couples:
            for tree_type in tree_types_list:
                for mode in tree_mode_list:
                    for criterion in criterion_list:
                        for bins in bins_list:
                            for depth in max_depth_list:
                                for msl in msl_list:
                                    varia_min_sample_split(metric, couple, tree_type, mode, criterion, bins, depth, msl)
                                # end
                            # end
                        # end
                    # end
                # end
            # end
        # end
    # end_for

    # faccio variare min sample leaf e tengo fissi max_depth e min sample split:
    for metric in metrics_list:
        for couple in result_couples:
            for tree_type in tree_types_list:
                for mode in tree_mode_list:
                    for criterion in criterion_list:
                        for bins in bins_list:
                            for depth in max_depth_list:
                                for mss in mss_list:
                                    varia_min_sample_leaf(metric, couple, tree_type, mode, criterion, bins, depth, mss)
                                # end
                            # end
                        # end
                    # end
                # end
            # end
        # end
    # end_for

    return


def _sinks(couple, dt, mode, criterion):
    # settings:
    bins = 20
    depth = 3
    mss = 50
    msl = 25

    # leggo il nome dei file .csv appartenenti allo stesso dataset su cui andrò ad effettuare l'analisi
    mixed_csv = couple[0]
    sklearn_csv = couple[1]

    # dataset name:
    temp_ds_name = re.split('/', couple[0])[1]
    ds_name = re.split('_', temp_ds_name)[0]

    # Cartella su cui andrò a salvare i massimi e minimi delle metriche
    folder_name = 'PeaksSinks/Sinks/'

    if dt == 'MixedDecisionTree':
        df = pd.read_csv(mixed_csv)
        save_string = folder_name + dt + '_' + ds_name + '.txt'
        if mode == 'classic':
            df_mixed = df[(df['Tree_Mode'] == mode) & (df['Criterion'] == criterion)
                          & (df['Bins'] == 0) & (df['Max_Depth'] == depth)
                          & (df['Min_sample_split'] == mss) & (df['Min_sample_leaf'] == msl)].min()
        else:
            df_mixed = df[(df['Tree_Mode'] == mode) & (df['Criterion'] == criterion)
                          & (df['Bins'] == bins) & (df['Max_Depth'] == depth)
                          & (df['Min_sample_split'] == mss) & (df['Min_sample_leaf'] == msl)].min()
        # enbd_if_then_else
        print(dt.upper())
        print(df_mixed)
        df_mixed.to_csv(save_string, sep='\t', encoding='utf-8', mode="a")

    elif dt == 'SklearnDT':
        df = pd.read_csv(sklearn_csv)
        save_string = folder_name + dt + '_' + ds_name + '.txt'
        if mode == 'classic':
            df_sklearn = df[(df['Tree_Mode'] == mode) & (df['Criterion'] == criterion)
                            & (df['Bins'] == 0) & (df['Max_Depth'] == depth)
                            & (df['Min_sample_split'] == mss) & (df['Min_sample_leaf'] == msl)].min()
        else:
            df_sklearn = df[(df['Tree_Mode'] == mode) & (df['Criterion'] == criterion)
                            & (df['Bins'] == bins) & (df['Max_Depth'] == depth)
                            & (df['Min_sample_split'] == mss) & (df['Min_sample_leaf'] == msl)].min()
        # end_if_then_else
        print(dt.upper())
        print(df_sklearn)
        df_sklearn.to_csv(save_string, sep='\t', encoding='utf-8', mode='a')
    else:
        raise Exception("Wrong DT name...")
    # end_if_then_else
    print("\n")
    return


def _peaks(couple, dt, mode, criterion):
    # settings:
    bins = 11
    depth = 3
    mss = 50
    msl = 25

    # leggo il nome dei file .csv appartenenti allo stesso dataset su cui andrò ad effettuare l'analisi
    mixed_csv = couple[0]
    sklearn_csv = couple[1]

    # dataset name:
    temp_ds_name = re.split('/', couple[0])[1]
    ds_name = re.split('_', temp_ds_name)[0]

    # Cartella su cui andrò a salvare i massimi e minimi delle metriche
    folder_name = 'PeaksSinks/Peaks/'

    if dt == 'MixedDecisionTree':
        df = pd.read_csv(mixed_csv)
        save_string = folder_name + dt + '_' + ds_name + '.txt'
        if mode == 'classic':
            df_mixed = df[(df['Tree_Mode'] == mode) & (df['Criterion'] == criterion)
                          & (df['Bins'] == 0) & (df['Max_Depth'] == depth)
                          & (df['Min_sample_split'] == mss) & (df['Min_sample_leaf'] == msl)].max()
        else:
            df_mixed = df[(df['Tree_Mode'] == mode) & (df['Criterion'] == criterion)
                          & (df['Bins'] == bins) & (df['Max_Depth'] == depth)
                          & (df['Min_sample_split'] == mss) & (df['Min_sample_leaf'] == msl)].max()
        # enbd_if_then_else
        print(dt.upper())
        print(df_mixed)
        df_mixed.to_csv(save_string, sep='\t', encoding='utf-8', mode="a")  # a+ è anche per la modalità lettura

    elif dt == 'SklearnDT':
        df = pd.read_csv(sklearn_csv)
        save_string = folder_name + dt + '_' + ds_name + '.txt'
        if mode == 'classic':
            df_sklearn = df[(df['Tree_Mode'] == mode) & (df['Criterion'] == criterion)
                            & (df['Bins'] == 0) & (df['Max_Depth'] == depth)
                            & (df['Min_sample_split'] == mss) & (df['Min_sample_leaf'] == msl)].max()
        else:
            df_sklearn = df[(df['Tree_Mode'] == mode) & (df['Criterion'] == criterion)
                            & (df['Bins'] == bins) & (df['Max_Depth'] == depth)
                            & (df['Min_sample_split'] == mss) & (df['Min_sample_leaf'] == msl)].max()
        # end_if_then_else
        print(dt.upper())
        print(df_sklearn)
        df_sklearn.to_csv(save_string, sep='\t', encoding='utf-8', mode='a')
    else:
        raise Exception("Wrong DT name...")
    # end_if_then_else
    print("\n")
    return


def peaks_and_sinks(couples_list, tree_types_list, tree_mode_list, criterion_list):
    for couple in couples_list:
        for dt in tree_types_list:
            for mode in tree_mode_list:
                for criterion in criterion_list:
                    _peaks(couple, dt, mode, criterion)
                    _sinks(couple, dt, mode, criterion)

    return


def main():
    print("Qui verranno generate tabelle e test per ogni dataset")
    print("Vedremo le differenze di prestazioni tra il mio MixedDecisionTree e quello di scikit-learn")
    folder_name = 'Results/'
    metrics_list = ['ACCURACY', 'F1_SCORE']

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

    tree_types_list = ['MixedDecisionTree', 'SklearnDT']
    tree_mode_list = ['classic', 'median', 'binned']
    criterion_list = ['entropy', 'gini']
    bins_list = [20, 15, 11, 7]
    max_depth_list = [2, 3, 4, 5, 6, 7, 8, 25, 50]
    mss_list = [2, 3, 5, 10, 25, 50]  # min_sample_split list
    msl_list = [1, 3, 5, 10, 25, 50]  # min_sample_leaf list
    peaks_and_sinks(couples_list, tree_types_list, tree_mode_list, criterion_list)
    tables(metrics_list, couples_list, tree_types_list, tree_mode_list, criterion_list, bins_list,
           max_depth_list, mss_list, msl_list)
    return


if __name__ == '__main__':
    main()
