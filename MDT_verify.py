"""
TESI DI LAUREA IN INFORMATICA PRESSO L'UNIVERSITÀ DI PISA
@Author: Giuseppe Muschetta
@Thesis_Supervisor: Professor Riccardo Guidotti
"""

import time
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# qui sto importando tutto il file mixed_DT.py
from MixedDecisionTree import MixedDecisionTree


def start_test(total_rows, file_name, dataframe, class_name, cut_threshold, criterion_list, tree_mode_list, bins_list,
               max_depth_list, min_sample_split_list, min_sample_leaf_list):
    list_of_lists = list()
    rows_counter = 0
    percentage = 0
    for tree_mode in tree_mode_list:
        if tree_mode != 'classic':
            for criterion in criterion_list:
                for bins in bins_list:
                    for max_depth in max_depth_list:
                        for sample_split in min_sample_split_list:
                            for sample_leaf in min_sample_leaf_list:
                                if sample_split > sample_leaf:
                                    rows_list = list()

                                    # ISTANZIO IL MIXED_DECISION_TREE:
                                    tree = MixedDecisionTree(class_name, max_depth, sample_split, sample_leaf,
                                                             tree_mode, bins, cut_threshold, criterion)

                                    # CONFIGURO ESTERNAMENTE IL DATASET IN BASE ALLA MODALITÀ SCELTA:
                                    X_train, X_test, y_train, y_test = tree.select_tree_modes(dataframe)

                                    ##################################
                                    # QUI PARTE IL TRAINING DELL'ALBERO:
                                    # faccio partire il cronometro che calcolerà il tempo di training dell'albero
                                    start_chrono_ID3 = time.time()
                                    # adesso faccio partire il training dell'albero
                                    tree.root = tree.fit(X_train, y_train)
                                    # stoppo il "cronometro"
                                    end_chrono_ID3 = time.time()
                                    # il tempo viene calcolato in secondi
                                    elapsed_time = end_chrono_ID3 - start_chrono_ID3

                                    # FASE DI PREDIZIONE:
                                    # calcolo accuracy_score e f1_score usando le funzioni di libreria
                                    y_pred = tree.predict(X_test, y_test)
                                    accuracy = accuracy_score(y_test, y_pred)
                                    f1val = f1_score(y_test, y_pred, average='weighted')
                                    ##################################

                                    # inserisco in lista i dati, ogni lista sarà una riga nel dataframe
                                    rows_list.append(tree_mode)
                                    rows_list.append(criterion)
                                    rows_list.append(bins)
                                    rows_list.append(max_depth)
                                    rows_list.append(sample_split)
                                    rows_list.append(sample_leaf)
                                    rows_list.append(accuracy)
                                    rows_list.append(f1val)
                                    rows_list.append(elapsed_time * 1000)  # tempo misurato in millisecondi

                                    # faccio il conto delle righe:
                                    rows_counter += 1

                                    if rows_counter == int(total_rows / 100):
                                        percentage += 1
                                        rows_counter = 0
                                        if percentage >= 100:
                                            percentage = 100

                                    print("\rCOMPLETAMENTO TEST:", percentage, "%", end='', flush=True)

                                    # inserisco la lista appena creata nella lista di liste
                                    list_of_lists.append(rows_list)

                                # end_if
        else:
            bins = 0
            for criterion in criterion_list:
                for max_depth in max_depth_list:
                    for sample_split in min_sample_split_list:
                        for sample_leaf in min_sample_leaf_list:
                            if sample_split > sample_leaf:
                                rows_list = list()

                                # ISTANZIO IL MIXED_DECISION_TREE:
                                tree = MixedDecisionTree(class_name, max_depth, sample_split, sample_leaf,
                                                         tree_mode, bins, cut_threshold, criterion)

                                # CONFIGURO ESTERNAMENTE IL DATASET IN BASE ALLA MODALITÀ SCELTA:
                                X_train, X_test, y_train, y_test = tree.select_tree_modes(dataframe)

                                ##################################
                                # QUI PARTE IL TRAINING DELL'ALBERO:
                                # faccio partire il cronometro che calcolerà il tempo di training dell'albero
                                start_chrono_ID3 = time.time()
                                # adesso faccio partire il training dell'albero
                                tree.root = tree.fit(X_train, y_train)
                                # stoppo il "cronometro"
                                end_chrono_ID3 = time.time()
                                # il tempo viene calcolato in secondi
                                elapsed_time = end_chrono_ID3 - start_chrono_ID3

                                # FASE DI PREDIZIONE:
                                # calcolo accuracy_score e f1_score usando le funzioni di libreria
                                y_pred = tree.predict(X_test, y_test)
                                accuracy = accuracy_score(y_test, y_pred)
                                f1val = f1_score(y_test, y_pred, average='weighted')
                                ##################################

                                # inserisco in lista i dati, ogni lista sarà una riga nel dataframe
                                rows_list.append(tree_mode)
                                rows_list.append(criterion)
                                rows_list.append(bins)
                                rows_list.append(max_depth)
                                rows_list.append(sample_split)
                                rows_list.append(sample_leaf)
                                rows_list.append(accuracy)
                                rows_list.append(f1val)
                                rows_list.append(elapsed_time * 1000)  # tempo misurato in millisecondi

                                # faccio il conto delle righe:
                                rows_counter += 1

                                if rows_counter == int(total_rows / 100):
                                    percentage += 1
                                    rows_counter = 0
                                    if percentage >= 100:
                                        percentage = 100

                                print("\rCOMPLETAMENTO TEST:", percentage, "%", end='', flush=True)

                                # inserisco nella lista di liste
                                list_of_lists.append(rows_list)

                            # end_if

        # end_if_then_else

    # end_outer_for

    # Creo un DataFrame a partire dalla lista di liste contenente tutti i dati che ci servono:
    dataframe = pd.DataFrame(list_of_lists, columns=['Tree_Mode', 'Criterion', 'Bins', 'Max_Depth', 'Min_sample_split',
                                                     'Min_sample_leaf', 'ACCURACY', 'F1_SCORE',
                                                     'ELAPSED_TIME_MS'])

    # Voglio aggiungere una colonna che calcola il tempo totale del test facendo sostanzialmente la somma
    # della riga ELAPSED_TIME_MS
    test_total_time_ms = dataframe['ELAPSED_TIME_MS'].sum()  # il valore sarà in ms
    test_total_time_sec = test_total_time_ms / 1000  # il valore sarà in secondi
    test_total_time_min = test_total_time_sec / 60  # il valore sarà in minuti
    dataframe['TEST_TOTAL_TIME_MIN'] = test_total_time_min
    dataframe['TEST_TOTAL_TIME_SEC'] = test_total_time_sec

    # serve per tranquillizzare il compilatore
    pd.set_option('mode.chained_assignment', None)
    j = 1
    while j < len(dataframe['TEST_TOTAL_TIME_MIN']):
        dataframe['TEST_TOTAL_TIME_MIN'][j] = ' '
        j += 1

    i = 1
    while i < len(dataframe['TEST_TOTAL_TIME_SEC']):
        dataframe['TEST_TOTAL_TIME_SEC'][i] = ' '
        i += 1

    # esportiamo un file .csv contenente tutte le informazioni circa il test effettuato
    input_csv = file_name.split('.')
    output_csv = input_csv[0] + '_Mixed_DT_result.csv'
    dataframe.to_csv('Results/' + output_csv, index=False, header=True, mode='w')
    print("\nTEST RESULT FILE:", output_csv, "has been created successfully!")
    return


def main():
    # devi rifare il test dellla prostata con entrambi gli alberi

    # TEST DEL MIXED_DECISION_TREE
    folder_name = 'Datasets/'
    file_name = 'ChickWeight.csv'
    dataframe = pd.read_csv(folder_name + file_name, skipinitialspace=True)

    # qui va inserito il nome della colonna di classificazione (la variabile dipendente Y)
    class_name = 'class'  # il nome di default è class
    cut_threshold = 50000  # vale solo per la modalità 'classic'

    criterion_list = ['entropy', 'gini']

    tree_mode_list = ['classic', 'median', 'binned']

    bins_list = [20, 15, 11, 7]

    max_depth_list = [2, 3, 4, 5, 6, 7, 8, 25, 50]

    min_sample_split_list = [2, 3, 5, 10, 25, 50]

    min_sample_leaf_list = [1, 3, 5, 10, 25, 50]

    # Le configurazioni attuali sono 6*6*9*4*2*2 - 6*6*9*4*1*2 = 2592
    for_configurations = len(min_sample_leaf_list) * len(min_sample_split_list) * \
                         len(max_depth_list) * len(bins_list) * (len(tree_mode_list) - 1) * len(criterion_list) - \
                         len(min_sample_leaf_list) * len(min_sample_split_list) * \
                         len(max_depth_list) * len(bins_list) * (len(tree_mode_list) - 2) * len(criterion_list)

    # QUI FACCIO PARTIRE IL TEST MISURANDO ANCHE IL SUO TEMPO COMPLESSIVO:
    print("STARTING TEST: dataset file", file_name, "is being tested...")
    print("Execution time may vary from a few minutes to even several hours")
    print("Please wait...\n")

    start_time_measure = time.time()
    start_test(for_configurations, file_name, dataframe, class_name, cut_threshold, criterion_list, tree_mode_list,
               bins_list, max_depth_list, min_sample_split_list, min_sample_leaf_list)
    end_time_measure = time.time()
    elapsed_test_time = end_time_measure - start_time_measure
    if elapsed_test_time > 60:
        print("il test è durato", elapsed_test_time / 60, "minuti")
    else:
        print("il test è durato", elapsed_test_time, "secondi")
    return


if __name__ == '__main__':
    main()
