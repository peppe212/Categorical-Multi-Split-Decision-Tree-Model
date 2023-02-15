"""
TESI DI LAUREA IN INFORMATICA PRESSO L'UNIVERSITÀ DI PISA
@Author: Giuseppe Muschetta
@Thesis_Supervisor: Professor Riccardo Guidotti
@Libraries: Pandas 1.4.4
            MatPlotLib 3.5.2
            Sklearn 1.1.1
"""

# funzioni che calcolano accuracy e media armonica
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
# qui importo l'albero di decisione preso dalla libreria
from sklearn.tree import DecisionTreeClassifier
# per la stsampa testuale dell'albero
from sklearn import tree
# per la visualizzazione grafica dell'albero
from matplotlib import pyplot as plt
import time
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


def median(folder_name, file_name, bins, class_name):
    dataframe = pd.read_csv(folder_name + file_name, skipinitialspace=True)
    assert (dataframe[class_name] is not None)
    dataframe.columns = dataframe.columns.str.replace(' ', '_')

    X_columns = list()
    for column in dataframe.columns:
        if column != class_name:
            X_columns.append(column)
    Y_target = class_name

    X = dataframe[X_columns]  # X è il df contenente tutte le colonne tranne quella di classificazione
    y = dataframe[Y_target]

    # mi scelgo tutte le colonne con valori continui, cioè quelle i cui valori sono float o int
    int_features = X.select_dtypes(include=int)
    float_features = X.select_dtypes(include=float)

    # lista contenente tutte le feature continue, interi e float
    continuous_features = int_features.columns.tolist() + float_features.columns.tolist()
    # lista contenente tutte le feature categoriche, cioè le stringhe object
    object_features = X.select_dtypes(include=object).columns.tolist()

    # adesso dobbiamo distinguere vari casi:
    if len(continuous_features) > 0:
        if len(object_features) > 0:
            # siamo nel caso in cui il dataset è misto:
            X_cont = X[continuous_features]
            X_obj = X[object_features]
            # applico la get dummies al dataframe contenente solo colonne object
            X_obj = pd.get_dummies(X_obj)
            # applico la cut e la mid al dataset contenente solo variabili continue:
            for c in X_cont.columns:
                X_cont[c] = pd.cut(X_cont[c], bins)
                X_cont[c] = X_cont[c].apply(lambda x: x.mid)
            # unisco i due dataframe:
            df = X_cont.join(X_obj)
            X_train, X_test, y_train, y_test = train_test_split(df, y,
                                                                test_size=0.33, random_state=43, stratify=y)
        else:
            # sono nel caso in cui il dataset è composto solo da valori continui:
            X_cont = X[continuous_features]
            # applico la cut e la mid al dataset contenente solo variabili continue:
            for c in X_cont.columns:
                X_cont[c] = pd.cut(X_cont[c], bins)
                X_cont[c] = X_cont[c].apply(lambda x: x.mid)
            # creo adesso le porzioni che ci interessano per il train e il test
            X_train, X_test, y_train, y_test = train_test_split(X_cont, y,
                                                                test_size=0.33, random_state=43, stratify=y)
    else:
        if len(object_features) > 0:
            # sono nel caso in cui il dataset è tutto categorico
            X_obj = X[object_features]
            X_obj = pd.get_dummies(X_obj)
            X_train, X_test, y_train, y_test = train_test_split(X_obj, y,
                                                                test_size=0.33, random_state=43, stratify=y)
        else:
            raise Exception("il dataset è vuoto...")

    return X_train, X_test, y_train, y_test


def classic(folder_name, file_name, class_name):
    dataframe = pd.read_csv(folder_name + file_name, skipinitialspace=True)
    assert (dataframe[class_name] is not None)
    dataframe.columns = dataframe.columns.str.replace(' ', '_')

    X_columns = list()
    for column in dataframe.columns:
        if column != class_name:
            X_columns.append(column)
    Y_target = class_name

    X = dataframe[X_columns]
    y = dataframe[Y_target]

    # al dataframe contenente solo colonne dai valori object faccio la getdummies
    X = pd.get_dummies(X, columns=None)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                        random_state=43, stratify=y)

    return X_train, X_test, y_train, y_test


def binned(folder_name, file_name, bins, class_name):
    dataframe = pd.read_csv(folder_name + file_name, skipinitialspace=True)
    assert (dataframe[class_name] is not None)
    dataframe.columns = dataframe.columns.str.replace(' ', '_')

    X_columns = list()
    for column in dataframe.columns:
        if column != class_name:
            X_columns.append(column)
    Y_target = class_name

    X = dataframe[X_columns]  # X è il df contenente tutte le colonne tranne quella di classificazione
    y = dataframe[Y_target]

    # mi scelgo tutte le colonne con valori continui, cioè quelle i cui valori sono float o int
    int_features = X.select_dtypes(include=int)
    float_features = X.select_dtypes(include=float)

    # lista contenente tutte le feature continue, interi e float
    continuous_features = int_features.columns.tolist() + float_features.columns.tolist()

    # lista contenente tutte le feature categoriche, cioè le stringhe object
    object_features = X.select_dtypes(include=object).columns.tolist()

    # adesso dobbiamo distinguere vari casi:
    if len(continuous_features) > 0:

        if len(object_features) > 0:
            # siamo nel caso in cui il dataset è misto:
            X_cont = X[continuous_features]
            X_obj = X[object_features]
            # applico la get dummies al dataframe contenente solo colonne object
            X_obj = pd.get_dummies(X_obj)
            # applico la cut al dataset contenente solo variabili continue:
            for c in X_cont.columns:
                X_cont[c] = pd.cut(X_cont[c], bins)
            # faccio la getdummies delle variabili continue ormai trasformate in stringhe d'intervalli
            X_cont = pd.get_dummies(X_cont)
            # unisco i due dataframe:
            df = X_cont.join(X_obj)
            X_train, X_test, y_train, y_test = train_test_split(df, y,
                                                                test_size=0.33, random_state=43, stratify=y)
        else:
            # sono nel caso in cui il dataset è solo fatto di valori continui:
            X_cont = X[continuous_features]
            # al dataframe contenente solo colonne continue applico la cut
            for c in X_cont.columns:
                X_cont[c] = pd.cut(X_cont[c], bins)

            # faccio la getdummies delle variabili continue a intervallo
            X_cont = pd.get_dummies(X_cont)
            # creo adesso le porzioni che ci interessano per il train e il test
            X_train, X_test, y_train, y_test = train_test_split(X_cont, y,
                                                                test_size=0.33, random_state=43, stratify=y)
    else:
        if len(object_features) > 0:
            # sono nel caso in cui il dataset è tutto categorico
            X_obj = X[object_features]
            X_obj = pd.get_dummies(X_obj)
            X_train, X_test, y_train, y_test = train_test_split(X_obj, y,
                                                                test_size=0.33, random_state=43, stratify=y)
        else:
            raise Exception("il dataset è vuoto...")

    return X_train, X_test, y_train, y_test


def select_mode(folder_name, file_name, bins, class_name, tree_mode):
    if tree_mode == 'classic':
        X_train, X_test, y_train, y_test = classic(folder_name, file_name, class_name)
    elif tree_mode == 'median':
        X_train, X_test, y_train, y_test = median(folder_name, file_name, bins, class_name)
    elif tree_mode == 'binned':
        X_train, X_test, y_train, y_test = binned(folder_name, file_name, bins, class_name)
        return X_train, X_test, y_train, y_test
    else:
        return Exception("valid modes are 'classic', 'median' or 'binned'")
    return X_train, X_test, y_train, y_test


def start_test(total_rows, folder_name, file_name, bins_list, max_depth_list, min_sample_split_list,
               min_sample_leaf_list, criterion_list, tree_mode_list, class_name='class'):
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
                                    # PER QUANTO RIGUARDA IL MIO ALBERO DI DECISIONE:
                                    # istanzio una nuova lista la lista ad ogni iterazione del for più interno
                                    rows_list = list()
                                    # recupero i training_set e testing_set in base alla modalità dell'albero
                                    X_train, X_test, y_train, y_test = select_mode(folder_name, file_name, bins,
                                                                                   class_name, tree_mode)

                                    # ISTANZIO IL DecisionTreeClassifier di Sklearn
                                    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                                                 min_samples_split=sample_split,
                                                                 min_samples_leaf=sample_leaf)

                                    # QUI PARTE IL TRAINING DELL'ALBERO DI SKLEARN
                                    start_algo_time = time.time()
                                    # inizio il training dell'albero
                                    clf.fit(X_train, y_train)
                                    end_algo_time = time.time()
                                    elapsed_algo_time = end_algo_time - start_algo_time

                                    # QUI PARTE LA FASE DI PREDIZIONE CON LE RELATIVE METRICHE DI PERFORMANCE
                                    y_predicted = clf.predict(X_test)
                                    accuracy = accuracy_score(y_true=y_test, y_pred=y_predicted)
                                    f1val = f1_score(y_true=y_test, y_pred=y_predicted, average='weighted')

                                    # inserisco in lista i dati, ogni lista sarà una riga nel dataframe
                                    rows_list.append(tree_mode)
                                    rows_list.append(criterion)
                                    rows_list.append(bins)
                                    rows_list.append(max_depth)
                                    rows_list.append(sample_split)
                                    rows_list.append(sample_leaf)
                                    rows_list.append(accuracy)
                                    rows_list.append(f1val)
                                    rows_list.append(elapsed_algo_time * 1000)  # tempo misurato in millisecondi

                                    # faccio il conto delle righe:
                                    rows_counter += 1

                                    if rows_counter == int(total_rows/100):
                                        percentage += 1
                                        rows_counter = 0
                                        if percentage >= 100:
                                            percentage = 100

                                    print("\rCOMPLETAMENTO TEST:", percentage, "%", end='', flush=True)

                                    # inserisco nella lista di liste
                                    list_of_lists.append(rows_list)

                                # end_if
        else:
            bins = 0
            for criterion in criterion_list:
                for max_depth in max_depth_list:
                    for sample_split in min_sample_split_list:
                        for sample_leaf in min_sample_leaf_list:
                            if sample_split > sample_leaf:
                                # PER QUANTO RIGUARDA IL MIO ALBERO DI DECISIONE:
                                # istanzio una nuova lista la lista ad ogni iterazione del for più interno
                                rows_list = list()
                                # recupero i training_set e testing_set in base alla modalità dell'albero
                                X_train, X_test, y_train, y_test = select_mode(folder_name, file_name, bins,
                                                                               class_name, tree_mode)

                                # ISTANZIO IL DecisionTreeClassifier di Sklearn
                                clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                                             min_samples_split=sample_split,
                                                             min_samples_leaf=sample_leaf)

                                # QUI PARTE IL TRAINING DELL'ALBERO DI SKLEARN
                                start_algo_time = time.time()
                                # inizio il training dell'albero
                                clf.fit(X_train, y_train)
                                end_algo_time = time.time()
                                elapsed_algo_time = end_algo_time - start_algo_time

                                # QUI PARTE LA FASE DI PREDIZIONE CON LE RELATIVE METRICHE DI PERFORMANCE
                                y_predicted = clf.predict(X_test)
                                accuracy = accuracy_score(y_true=y_test, y_pred=y_predicted)
                                f1val = f1_score(y_true=y_test, y_pred=y_predicted, average='weighted')

                                # inserisco in lista i dati, ogni lista sarà una riga nel dataframe
                                rows_list.append(tree_mode)
                                rows_list.append(criterion)
                                rows_list.append(bins)
                                rows_list.append(max_depth)
                                rows_list.append(sample_split)
                                rows_list.append(sample_leaf)
                                rows_list.append(accuracy)
                                rows_list.append(f1val)
                                rows_list.append(elapsed_algo_time * 1000)  # tempo misurato in millisecondi

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
    output_csv = input_csv[0] + '_Sklearn_DT_result.csv'
    dataframe.to_csv('Results/' + output_csv, index=False, header=True, mode='w')
    print("\nTEST RESULT FILE:", output_csv, "HAS BEEN CREATED SUCCESSFULLY!")
    return


def plotDefaultTree(folder_name, file_name, bins, class_name, tree_mode):
    X_train, X_test, y_train, y_test = select_mode(folder_name, file_name, bins,
                                                   class_name, tree_mode)

    # ISTANZIO UN SKLEARN DT CON PARAMETRI DEFAULT: 50 max depth, 2 max_sample_split, 1 max_sample_leaf
    clf = DecisionTreeClassifier(max_depth=50, min_samples_split=2, min_samples_leaf=1)

    # visualizzo graficamente l'albero aopena istanziato
    graphic_tree_visualization(clf, file_name, X_train, y_train)
    return


def graphic_tree_visualization(clf, file_name, X_train, y_train):
    # cartella nella quale creare i file grafici:
    folder_name = 'Visual_sklearn/'

    # faccio il training dell'albero con parametri default
    clf.fit(X_train, y_train)

    # esporto la versione testuale stampata dell'albero:
    text_representation = tree.export_text(clf)
    input_log = file_name.split('.')
    output_log = input_log[0] + '_textTREE.log'
    fout = open(folder_name + output_log, "w")
    if fout is None:
        raise OSError("impossibile to open the file")
    fout.write(text_representation)
    print("\nil file di log", output_log, "è stato creato con successo")

    # con matplotlib visualizzo graficamente l'albero
    fig = plt.figure(figsize=(25, 20))
    _ = tree.plot_tree(clf)

    input_fig = file_name.split('.')
    output_fig = input_fig[0] + '_graphicTREE.png'
    fig.savefig(folder_name + output_fig)
    print("il file .png", output_fig, "è stato creato con successo")
    fout.close()
    return


def main():
    # TESTIAMO L'ALBERO DI SKLEARN
    folder_name = 'Datasets/'
    file_name = 'Auto.csv'
    # qui va inserito il nome della colonna di classificazione (la variabile dipendente Y)
    class_name = 'class'  # il nome di default è class

    criterion_list = ['entropy', 'gini']

    tree_mode_list = ['classic', 'median', 'binned']

    bins_list = [20, 15, 11, 7]

    max_depth_list = [2, 3, 4, 5, 6, 7, 8, 25, 50]

    min_sample_split_list = [2, 3, 5, 10, 25, 50]

    min_sample_leaf_list = [1, 3, 5, 10, 25, 50]

    # le configurazioni attuali sono 6*6*9*4*2*2 - 6*6*9*4*1*2 = 2592
    for_configurations = len(min_sample_leaf_list) * len(min_sample_split_list) * \
                         len(max_depth_list) * len(bins_list) * (len(tree_mode_list) - 1) * len(criterion_list) - \
                         len(min_sample_leaf_list) * len(min_sample_split_list) * \
                         len(max_depth_list) * len(bins_list) * (len(tree_mode_list) - 2) * len(criterion_list)

    # QUI FACCIO PARTIRE IL TEST MISURANDO ANCHE IL SUO TEMPO COMPLESSIVO:
    print("STARTING TEST: dataset file", file_name, "is being tested...")
    print("Execution time may vary from a few seconds to even several minutes")
    print("Please wait...\n")

    start_time_measure = time.time()  # giusto come info console
    start_test(for_configurations, folder_name, file_name, bins_list, max_depth_list, min_sample_split_list,
               min_sample_leaf_list, criterion_list, tree_mode_list, class_name)
    end_time_measure = time.time()
    elapsed_test_time = end_time_measure - start_time_measure
    if elapsed_test_time > 60:
        print("il test è durato ", elapsed_test_time / 60, "minuti")
    else:
        print("il test è durato", elapsed_test_time, "secondi")

    # QUI PLOTTO L'ALBERO CLASSICO CON PARAMETRI DEFAULT: 50 max depth, 2 max_sample_split, 1 max_sample_leaf
    plotDefaultTree(folder_name, file_name, 0, class_name, 'classic')
    return


if __name__ == '__main__':
    main()
