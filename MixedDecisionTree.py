"""
TESI DI LAUREA IN INFORMATICA PRESSO L'UNIVERSITÀ DI PISA
@Author: Giuseppe Muschetta
@Thesis_Supervisor: Professor Riccardo Guidotti
"""

# importante per distinguere gli attributi continui: float da quelli categorici: stringhe object
from pandas.core.dtypes.common import is_numeric_dtype
# importante per preparare a modo il dataset iniziale
from sklearn.model_selection import train_test_split
# funzioni che calcolano accuracy e media armonica
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
# re serve per costruire un nome colonna del tipo: nome_attributo <= valore_continuo
import re
import time

"""
Le specifiche per questo albero di decisione sono le seguenti:
1 - ogni attributo categorico deve essere rappresentato da una stringa nel dataset d'ingresso
2 - ogni attributo intero o float presente nel dataset verrà convertito in float e sarà considerato continuo
3 - ogni nodo può avere molteplici figli, ergo molteplici split, non si tratta quindi
    del solito albero di decisione che splitta soltanto in modalità binaria come quello di Slearn
"""


class MixedDecisionTree:
    class Node:
        def __init__(self):
            self.split_condition = None
            self.gain: float = float()
            self.split_value = None
            self.isLeaf: bool = False
            self.isRoot: bool = False
            self.classification = None
            self.parent = None
            self.depth: int = int()
            self.children = list()

    def __init__(self, class_name='class', max_allowed_depth=50, min_samples_split=2, min_samples_leaf=1,
                 tree_mode='classic', number_of_bins=0, cut_threshold=10000, criterion='entropy'):
        self.root = None
        self.class_name = class_name
        self.initial_depth = 0
        self.max_allowed_depth = max_allowed_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree_mode = tree_mode
        self.number_of_bins = number_of_bins
        self.cut_threshold = cut_threshold
        self.criterion = criterion
        self.most_common_value = None

    # auxiliary method
    def _get_impurity(self, attribute):
        # il parametro attribute è di tipo pd.Series
        # con la potenza di numpy calcoliamo l'entropia
        values, counts = np.unique(attribute, return_counts=True)
        impurity_values = []
        if self.criterion == 'entropy':
            for i in range(len(values)):
                impurity_values.append(-(counts[i] / np.sum(counts)) * (np.log2(counts[i] / np.sum(counts))))
            impurity_measure = np.sum(impurity_values)
        elif self.criterion == 'gini':
            for i in range(len(values)):
                impurity_values.append((counts[i] / np.sum(counts)) ** 2)
            impurity_measure = 1.0 - np.sum(impurity_values)
        else:
            raise Exception("Wrong criteria input, possible choices are: 'entropy' or 'gini'")
        return impurity_measure

    # auxiliary method called by the inner _fit recursive function
    def _getGain(self, data, split_attribute_name):
        # calcolo l'entropia del padre
        I_parent = self._get_impurity(data[self.class_name])
        assert (I_parent >= 0)

        values, counts = np.unique(data[split_attribute_name], return_counts=True)
        impurity_values = []
        for i in range(len(values)):
            impurity_values.append(
                (counts[i] / np.sum(counts)) * self._get_impurity(
                    data.where(data[split_attribute_name] == values[i]).dropna()[self.class_name]))

        # calcolo l'entropia pesata dopo lo split (quella del figlio in sostanza...)
        I_child = np.sum(impurity_values)
        # finalmente ottengo l'information gain dell'attributo passato in ingresso
        gain = I_parent - I_child  # a volte questa differenza diventa negativa quando siamo alla -16 di esponente
        epsilon = 0.0000001  # epsilon piccolo a piacere > 0 al max prenderlo uguale alla precisione di macchina
        if gain < 0.0:
            if gain + epsilon >= 0.0:
                gain = 0.0
                return gain
            else:
                raise Exception("Gain cannot be negative...")
        else:  # se il gain è >=0 restituiscilo
            return gain

    # metodo ausiliario
    def _getMostCommonValue(self, data):
        class_values, counts = np.unique(data[self.class_name], return_counts=True)
        index_max = np.argmax(counts)
        return class_values[index_max]

    # metodo ausiliario invocato dalla _fit
    def _stoppingCriteria(self, data, attributes):
        # PRIMO CRITERIO DI STOP
        # se rimangono valori di etichetta tutti uguali, mi fermo
        # e la classificazione attribuita alla foglia sara' proprio quel valore di etichetta
        if len(np.unique(data[self.class_name])) <= 1:
            node = MixedDecisionTree.Node()
            node.isLeaf = True
            node.classification = np.unique(data[self.class_name])[0]
            return node, True
        # SECONDO CRITERIO DI STOP:
        # se la lunghezza dei subdata (cioè i dati che si vanno restringendo via via) == 0
        # restituisco il valore più comune di tutto il dataset, quindi dei dati originali
        if len(data) <= 0:
            node = MixedDecisionTree.Node()
            node.isLeaf = True
            node.classification = self.most_common_value
            return node, True
        # TERZO CRITERIO DI STOP
        # Se 'attributes' è vuota, restituisci una foglia avente come valore di classificazione
        # il valore piu' comune in tutto l'insieme dei training examples
        if len(attributes) <= 0:
            node = MixedDecisionTree.Node()
            node.isLeaf = True
            node.classification = self._getMostCommonValue(data)
            return node, True
        # PRUNING: MIN_SAMPLES_SPLIT
        # The min_samples_split parameter will evaluate the number of samples in the node,
        # and if the number is less than the minimum the split will be avoided and the node will be a leaf.
        if len(data) < self.min_samples_split:
            node = MixedDecisionTree.Node()
            node.isLeaf = True
            node.classification = self._getMostCommonValue(data)
            return node, True
        # Se nessuna delle condizioni si verifica allora, restituisco la tupla (None, False)
        return None, False

    # METODO D'ISTANZA CHE SI OCCUPA DELLA MANIPOLAZIONE DEL DATASET E DEL RRAINING DELL'ALBERO
    def fit(self, X_train, y_train):
        assert (X_train is not None and y_train is not None)
        X_train[self.class_name] = y_train
        # FASE DI TRAINING DELL'ALBERO:
        # spiegazione parametri che prende in ingresso il classificatore:
        # il primo parametro rappresentsa il training_set che si restringerà a colpi di ricorsione
        # il secondo parametro è una lista di tutti i nomi delle features tranne quella di classificazione
        # il terzo parametro rappresenta la profondità di partenza dell'albero
        self.root = self._fit(X_train, X_train.columns[:-1], self.initial_depth)
        return self.root

    # routine interna ricorsiva, si occupa del training dell'albero a partire da esempi di training
    def _fit(self, data, features, current_depth):
        # Se almeno uno degli stopping criteria si è verificato restituisco una foglia classificata
        node, i_have_to_stop = self._stoppingCriteria(data, features)
        if i_have_to_stop:
            if current_depth == self.initial_depth:
                node.isRoot = True
                assert (node.classification is not None)
            return node

        # PRUNING: MAX_DEPTH
        if current_depth >= self.max_allowed_depth:
            node = MixedDecisionTree.Node()
            node.isLeaf = True
            node.classification = self._getMostCommonValue(data)
            if current_depth == self.initial_depth:
                # non dovrebbe mai verificarsi questa if perché c'é il maggiore stretto nella condizione
                node.isRoot = True
            return node

        # QUI INIZIA IL CALCOLO DEL BEST-ATTRIBUTE SCELTO in base al criterio scelto
        gain_dict = dict()
        for feature in features:
            gain_dict[feature] = self._getGain(data, feature)
        best_split_attribute = max(gain_dict, key=gain_dict.get)
        assert (best_split_attribute is not None)

        # QUI VADO A RIDURRE L'INSIEME DEGLI ATTRIBUTI
        # non fa differenza tra attributi categorici e continui
        remaining_features = list()
        for feature in features:
            if feature != best_split_attribute:
                remaining_features.append(feature)

        sub_data_pairs_list = list()
        flag_leaf = False
        # per ogni valore dell'attributo di split facciamo crescere un ramo:
        for value in np.unique(data[best_split_attribute]):
            # questo valore puo' essere o di tipo numeric o no
            # per come è stato concepito questo albero, ogni attributo categorico è una stringa (str)
            # e ogni attributo continuo è un float
            split_attribute_value = value

            # considero il sottinsieme del dataset su cui andrò ad operare la ricorsione
            sub_data = data.where(data[best_split_attribute] == split_attribute_value).dropna()
            # PRUNING: MIN_SAMPLE_LEAF
            # The min_samples_leaf parameter checks, before the node is generated, if the possible
            # split results in a child with fewer samples, the split will be avoided
            # (since the minimum number of samples for the child to be a leaf has not been reached)
            # and the node will be replaced by a leaf.
            # I sample dei sottodati devono essere strettamente minori dell'iperparametro
            if len(sub_data) < self.min_samples_leaf:
                flag_leaf = True
                break
            sub_data_pairs_list.append((split_attribute_value, sub_data))

        if flag_leaf:
            # codice per fare foglia
            node = MixedDecisionTree.Node()
            node.isLeaf = True
            node.classification = self._getMostCommonValue(data)
            if current_depth == self.initial_depth:
                node.isRoot = True
            return node

        # SE I VARI CRITERI DI STOP NON SONO TRUE ARRIVIAMO A QUESTO PUNTO DEL CODICE
        # DOVE INIZIA LA VERA E PROPRIA CRESCITA DELL'ALBERO
        root = MixedDecisionTree.Node()
        root.split_condition = best_split_attribute
        root.gain = gain_dict[best_split_attribute]
        root.depth = current_depth
        if current_depth == self.initial_depth:
            root.isRoot = True

        # per ogni valore dell'attributo di split facciamo crescere un ramo:
        # for value in np.unique(input[best_split_attribute]):
        for pair in sub_data_pairs_list:
            split_attribute_value = pair[0]
            sub_data = pair[1]

            # PARTONO LE CHIAMATE RICORSIVE CHE FARANNO CRESCERE L'ALBERO IN PROFONDITÀ
            # nodo child o subroot
            child = self._fit(sub_data, remaining_features, current_depth + 1)

            child.split_value = split_attribute_value
            child.parent = root
            child.depth = current_depth + 1
            if is_numeric_dtype(split_attribute_value):
                # la funzione is_numeric_type_ considera il valore ? come un attributo numerico e non come stringa
                # bisogna stare attenti ai ? nei dataset e toglierli
                if split_attribute_value != '?':
                    if split_attribute_value == 1.0:
                        root.children.insert(0, child)  # inserisco come primo figlio o figlio di posizione 0 left
                    elif split_attribute_value == 0.0:
                        root.children.insert(1, child)  # inserisco come secondo figlio o figlio di posizione 1 right
                    else:
                        raise Exception("fatal error... mi è arrivato un split attribute value che non è numerico...")
                else:
                    raise Exception("Ho trovato un valore ? che va sostituito con un valore valido nel file .csv")
            else:  # se invece l'attributo è categorico, ergo object in python
                root.children.append(child)
        # end_for
        return root

    # auxiliary method
    # in median_mode il training_set verrà binnato con valori reali singoli, quindi non avremo stringhe d'intervalli
    # il testing_set non verrà modificato
    def _median_mode(self, dataframe):
        dataframe.columns = dataframe.columns.str.replace(' ', '_')
        dataframe[self.class_name] = dataframe[self.class_name].map(str)
        int_features = dataframe.select_dtypes(include=int)
        float_features = dataframe.select_dtypes(include=float)
        continuous_features = int_features.columns.tolist() + float_features.columns.tolist()
        assert (self.class_name not in continuous_features)
        for continuous_feature in continuous_features:
            dataframe[continuous_feature] = dataframe[continuous_feature].map(float)

        # mi ricavo dal dataframe generale, le porzioni di training_data e testing_data
        y = dataframe[self.class_name].map(str)
        columns = [c for c in dataframe.columns if c != self.class_name]
        X = dataframe[columns]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=43, stratify=y)

        # STO APPLICANDO LA CUT E LA MID SOLO AL TRAINING-SET
        for feature in continuous_features:
            X_train[feature] = pd.cut(X_train[feature], self.number_of_bins)
            X_train[feature] = X_train[feature].apply(lambda x: x.mid)

        # INIZIO LA MANIPOLAZIONE TENENDO CONTO DELL'OTTIMIZZAZIONE OFFERTA DALLA MEDIANA:
        cond_var_list = []
        for feature in continuous_features:
            feature_values_list = (X_train[feature].values.tolist())
            training_classes = np.unique(X_train[feature])
            for i in range(len(training_classes)):
                # in questo modo toglio gli spazio a destra e a sinistra della stringa
                # e in piu' sostituisco ad ogni spazio un trattino basso
                attribute_name = feature + " <= " + str(training_classes[i])
                cond_var_list.clear()
                for value in feature_values_list:
                    if value <= training_classes[i]:
                        cond_var_list.append(1.0)
                    else:
                        cond_var_list.append(0.0)
                # end_inner_for
                # questa è la soluzione che ho trovato al problema della memoria
                # con la join evito il problema della insert ripetuta, la join internamente ottimizza tantissimo
                to_be_added_df = pd.DataFrame({attribute_name: cond_var_list})
                X_train = X_train.reset_index(drop=True).join(to_be_added_df.reset_index(drop=True),
                                                              how='left')
            # end_middle_for
            del X_train[feature]
        # end_outer_for
        # useremo il training set come copia non framentata del training_data
        training_temp = X_train.copy(deep=True)
        del X_train
        X_train = training_temp

        # Resetto gli indici del training_set con drop=True
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        return X_train, X_test, y_train, y_test

    # aux method
    # in questa modalità non verrà applicata ne' la cut ne' la mid, il dataset verrà processato cosi com'è
    def _classic_mode(self, dataframe):
        dataframe.columns = dataframe.columns.str.replace(' ', '_')
        dataframe[self.class_name] = dataframe[self.class_name].map(str)

        # RIDIMENSIONO IL DATASET:
        # se il dataset è più grande di cut-threshold viene tagliata tutta la parte che va da soglia a fine dataset
        if len(dataframe) > self.cut_threshold:
            # provvedo a mantenere solo cut_threshold righe del dataset
            dataframe = dataframe.drop(labels=range(self.cut_threshold, len(dataframe)), axis=0)
        # end_if

        # INIZIA LA MANIPOLAZIONE DEL DATASET:
        # considerando che nel mio albero la variabile dipendente sarà sempre categorica,
        # mi parso subito la colonna a stringa
        y = dataframe[self.class_name].map(str)
        columns = [c for c in dataframe.columns if c != self.class_name]
        X = dataframe[columns]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=43, stratify=y)

        # INIZIO CREAZIONE COLONNE PER GESTIRE DATI CONTINUI:
        # MI PRENDO TUTTE LE COLONNE CONTENENTI VALORI DI TIPO FLOAT O INT
        int_features = dataframe.select_dtypes(include=int)
        float_features = dataframe.select_dtypes(include=float)
        continuous_features = int_features.columns.tolist() + float_features.columns.tolist()
        for continuous_feature in continuous_features:
            X_train[continuous_feature] = X_train[continuous_feature].map(float)
        cond_var_list = []
        for feature in continuous_features:
            feature_values_list = (X_train[feature].values.tolist())
            training_classes = np.unique(X_train[feature])
            for i in range(len(training_classes)):
                # in questo modo toglio gli spazio a destra e a sinistra della stringa
                # e in piu' sostituisco ad ogni spazio un trattino basso
                attribute_name = feature + " <= " + str(training_classes[i])
                cond_var_list.clear()
                for value in feature_values_list:
                    if value <= training_classes[i]:
                        cond_var_list.append(1.0)
                    else:
                        cond_var_list.append(0.0)
                # end_inner_for
                # questa è la soluzione che ho trovato al problema della memoria
                # con la join evito il problema della insert ripetuta, la join internamente ottimizza tantissimo
                to_be_added_df = pd.DataFrame({attribute_name: cond_var_list})
                X_train = X_train.reset_index(drop=True).join(to_be_added_df.reset_index(drop=True), how='left')
            # end_middle_for
            del X_train[feature]
        # end_outer_for
        # useremo il training set come copia non framentata del training_data
        training_temp = X_train.copy(deep=True)
        del X_train
        # rimetto la colonna di classificazione alla fine del training_set per motivi d'implementazione
        X_train = training_temp
        # Resetto gli indici del training_set con drop=True
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        return X_train, X_test, y_train, y_test

    # In questa modalità sia il training-set che il testing_set verranno splittati in stringhe d'intervalli
    # ogni dato continuo diventerà un intervallo. Gli attributi in partenza già categorici rimnarrano tali e quali.
    def _binned_mode(self, dataframe):
        dataframe.columns = dataframe.columns.str.replace(' ', '_')
        dataframe[self.class_name] = dataframe[self.class_name].map(str)
        int_features = dataframe.select_dtypes(include=int)
        float_features = dataframe.select_dtypes(include=float)
        continuous_features = int_features.columns.tolist() + float_features.columns.tolist()
        assert (self.class_name not in continuous_features)
        for feature in continuous_features:
            dataframe[feature] = pd.cut(dataframe[feature], self.number_of_bins)

        # mi ricavo dal dataframe generale, le porzioni di training_data e testing_data
        y = dataframe[self.class_name].map(str)
        columns = [c for c in dataframe.columns if c != self.class_name]
        X = dataframe[columns]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=43, stratify=y)

        return X_train, X_test, y_train, y_test

    def select_tree_modes(self, dataframe):
        if self.tree_mode == 'binned':
            X_train, X_test, y_train, y_test = self._binned_mode(dataframe)
        elif self.tree_mode == 'median':
            X_train, X_test, y_train, y_test = self._median_mode(dataframe)
        elif self.tree_mode == 'classic':
            X_train, X_test, y_train, y_test = self._classic_mode(dataframe)
        else:
            raise Exception("valid keywords are 'binned', 'median', 'classic'")
        self.most_common_value = self._getMostCommonValue(pd.DataFrame(data=y_train))
        return X_train, X_test, y_train, y_test

    # metodo ausiliario invocato dalla predict
    def _predict(self, root, example):
        if root is None:
            raise Exception("istanza l'albero e facci il training prima di lanciare la fnzione predict..!")
        if root.isLeaf:
            return root.classification
        if ' <= ' in root.split_condition:
            temp_string = root.split_condition
            # se ho una split condition continua alla radice:
            whole_condition = re.split(' <= ', temp_string)
            condition_name: str = str(whole_condition[0])
            condition_number: float = float(whole_condition[1])
            if len(root.children) > 1:
                if float(example[condition_name]) <= condition_number:
                    return self._predict(root.children[0], example)
                else:
                    return self._predict(root.children[1], example)
            # se arrivo a questo punto del codice per attributi continui, ci sarà un degrado nelle performance
            # infatti qui arrivo con i continui solo quado median ha 7 bins soltanto e ci sta.
            return self.most_common_value
        else:
            # se la split condition è categorica
            example_value = example[root.split_condition]
            for child in root.children:
                # il problema qui è che il test set puo avere un valore che non è presente nel training set
                if child.split_value == example_value:
                    return self._predict(child, example)
            # se arrivo a questo punto del codice per attributi categorici è perché nel test set
            # non corrisponde qualche valore di classe presente nel training set
            return self.most_common_value

    # METODO D'ISTANZA
    def predict(self, X_test, y_test):
        X_test[self.class_name] = y_test
        # Crea gli n esempi unseen da fornire all'albero convertendoli n dizionari
        examples = X_test.iloc[:, :-1].to_dict(orient="records")
        predicted_list = list()
        for i in range(len(X_test)):
            predicted_list.append(self._predict(self.root, examples[i]))
        # end_for
        return predicted_list

    # METODO D'ISTANZA:
    def displayDT(self, root, file, width):
        if root is None:
            raise Exception("a tree must have been instanciated..!")
        # Definisco il numero di spazi:
        const = int(root.depth * width ** 1.5)
        spaces = "-" * const
        # se sono sul nodo root
        if root.isRoot:
            assert (root.split_value is None)
            # se sono la radice e al tempo stesso una foglia, l'albero non si è nemmeno formato
            # restituisco la classificazione (essendo una foglia) del valore piu' comune nel training-set
            if root.isLeaf:
                assert (root.split_condition is None and root.classification is not None)
                # sono sul nodo root di un albero che non si è nemmeno potuto formare
                root_string = "sono root e allo stesso tempo foglia, posso solo darti la classificazione " \
                              f"del most common value che è {root.classification}"
                print(root_string)
                file.write(root_string)
                file.write("\n")
            else:
                # sono la radice di un albero che si sta formando, ergo verró splittato
                if self.criterion == 'entropy':
                    root_string = f"Root at depth:{root.depth} with Split Condition: |{root.split_condition}| " \
                                  f"with info_gain = {root.gain}"
                    print(root_string)
                    file.write(root_string)
                    file.write("\n")
                elif self.criterion == 'gini':
                    root_string = f"Root at depth: {root.depth} with Split Condition: |{root.split_condition}| " \
                                  f"with gini_gain = {root.gain}"
                    print(root_string)
                    file.write(root_string)
                    file.write("\n")
                else:
                    raise Exception("Wrong criteria input, possible choices are: 'entropy' or 'gini'")
        else:  # se non sono la radice posso essere o una foglia o un nodo interno con split condition e figli
            assert (root.split_value is not None)
            # se sono una foglia
            if root.isLeaf:
                if is_numeric_dtype(root.split_value):
                    assert (root.split_value == 1.0 or root.split_value == 0.0)
                    if root.split_value == 1.0:
                        word: str = "yes"
                    elif root.split_value == 0.0:
                        word: str = "no"
                    else:
                        raise Exception("valid values at this point can either be 1.0 or 0.0")
                else:  # se è di tipo string or object
                    word: str = str(root.split_value)

                string_leaf = 'depth:', str(root.depth), f"|{spaces} if {root.parent.split_condition} == " \
                                                         f"{word} --> Classification: {root.classification}"
                print(string_leaf)
                file.writelines(string_leaf)
                file.write("\n")
            # se sono un nodo interno
            else:
                if self.criterion == 'entropy':
                    string_not_leaf = 'depth:', str(root.depth), \
                                      f"|{spaces} Split Condition: |{root.split_condition}| " \
                                      f"with info_gain = {root.gain}"
                    print(string_not_leaf)
                    file.writelines(string_not_leaf)
                    file.write("\n")
                elif self.criterion == 'gini':
                    string_not_leaf = 'depth:', str(root.depth), \
                                      f"|{spaces} Split Condition: |{root.split_condition}| " \
                                      f"with gini_gain = {root.gain}"
                    print(string_not_leaf)
                    file.writelines(string_not_leaf)
                    file.write("\n")
                else:
                    raise Exception("Wrong criteria input, possible choices are: 'entropy' or 'gini'")
        # qui parte la ricorsione
        # se ho dei figli, allora child sarà il prossimo root
        if len(root.children) > 0:
            for child in root.children:
                assert (child.parent == root)
                self.displayDT(child, file, width)
        return


# end_class


def initial_infos(file):
    (dataset_folder_name, dataset_file_name, class_name, max_depth, min_sample_split, min_sample_leaf, criterion,
     tree_mode, number_of_bins, cut_threshold) = settings()
    print("Il file ", dataset_folder_name + dataset_file_name, "è stato letto con successo!\n")
    file.writelines(f"Il file '{dataset_folder_name + dataset_file_name}' è stato letto con successo!\n")
    print("L'albero di decisione è stato istanziato con i seguenti parametri:")
    file.write("\nL'albero di decisione è stato istanziato con i seguenti parametri:\n")
    print("MODALITÀ = ", tree_mode)
    file.writelines(f"MODALITÀ = {tree_mode} \n")
    print("CRITERIO = ", criterion)
    file.writelines(f"CRITERIO = {criterion} \n")
    print("SOGLIA_DI_TAGLIO = ", cut_threshold)
    file.writelines(f"SOGLIA_DI_TAGLIO = {cut_threshold} \n")
    print("BINS = ", number_of_bins)
    file.writelines(f"BINS = {number_of_bins} \n")
    print("\nGli iper-parametri per il pruning dell'albero sono:")
    file.write("\nGli iper-parametri per il pruning dell'albero sono:\n")
    print("MAX_DEPTH = ", max_depth)
    file.writelines(f"MAX_DEPTH = {max_depth} \n")
    print("MIN_SAMPLE_SPLIT = ", min_sample_split)
    file.writelines(f"MIN_SAMPLE_SPLIT = {min_sample_split} \n")
    print("MIN_SAMPLE_LEAF = ", min_sample_leaf)
    file.writelines(f"MIN_SAMPLE_LEAF = {min_sample_leaf} \n")
    print("\n")
    file.write("\n")
    return


# QUI È POSSIBILE SETTARE OGNI SINGOLO PARAMETRO DI CONFIGURAZIONE DEL MIXED_DECISION_TREE:
def settings() -> tuple:
    # selezionare il file .csv contenente il dataset
    dataset_folder_name = 'Datasets/'
    dataset_file_name = 'Iris.csv'

    # qui va inserito il nome della colonna di classificazione (la variabile dipendente Y)
    class_name = 'class'  # il nome di default è class

    # IPER-PARAMETRI utili per effettuare il PRUNING DELL'ALBERO
    max_depth = 50
    min_sample_split = 2
    min_sample_leaf = 1

    # qui inseriamo il criterio d'impurità: valori possibili 'entropy' o 'gini'
    criterion = 'entropy'
    # criterion = 'gini'

    # qui inseriamo la modalità dell'albero: 'classic', 'binned' o 'median'
    tree_mode = 'classic'
    # tree_mode = 'binned'
    # tree_mode = 'median'

    # qui inseriamo il numero di bin da utilizzare nella modalità binned o median
    # in modalità classic tale valore verrà ignorato internamente.
    number_of_bins = 20

    # qui inseriamo la soglia di taglio del dataset che riguarda pero' solo la modalità 'classic'
    # in modalità classica verrà ignorato il numero di bins
    cut_threshold = 100000

    return (dataset_folder_name, dataset_file_name, class_name, max_depth, min_sample_split, min_sample_leaf, criterion,
            tree_mode, number_of_bins, cut_threshold)


def main():
    # leggo i parametri di configurazione
    (dataset_folder_name, dataset_file_name, class_name, max_depth, min_sample_split, min_sample_leaf, criterion,
     tree_mode, number_of_bins, cut_threshold) = settings()

    # creo un file per stampare le informazioni circa l'albero
    file_txt_folder = 'visual_mixedDT/'
    file_txt_name = dataset_file_name.split('.')[0] + '_' + criterion + '_mixed_textTREE.txt'
    file = open(file_txt_folder + file_txt_name, 'w')
    if file is None:
        raise OSError("Error opening the file ", file_txt_name)

    # CREO UNA ISTANZA del MixedDecisionTree classifier:
    tree = MixedDecisionTree(class_name, max_depth, min_sample_split, min_sample_leaf, tree_mode,
                             number_of_bins, cut_threshold, criterion)

    # stampo a schermo la configurazione corrente del mixed decision tree appena istanziato
    initial_infos(file)

    # qui viene letto il dataset in formato .csv e restituito il relativo dataframe
    dataframe = pd.read_csv(dataset_folder_name + dataset_file_name, skipinitialspace=True)

    # qui prepariamo i vari dataset in base alla modalità di albero scelta:
    X_train, X_test, y_train, y_test = tree.select_tree_modes(dataframe)

    # AVVIO LA FASE DI TRAINING DELL'ALBERO A PARTIRE DAL DATAFRAME ricavato dal dataset
    start = time.time() * 1000  # ottengo il tempo in milli-secondi
    tree.root = tree.fit(X_train, y_train)
    end = time.time() * 1000
    elapsed_time = end - start
    if elapsed_time > 1000:
        print("Il training dell'albero è durato ", elapsed_time / 1000, "secondi")
    else:
        print("Il training dell'albero è durato ", elapsed_time, "millisecondi")

    # Invoco il metodo che mi va a stampare l'albero di decisione
    tree.displayDT(tree.root, file, 3)

    # FASE DI TEST AND PREDICT, da qui vediamo come generalizza l'albero
    y_pred_list = tree.predict(X_test, y_test)  # lista contenente le predizioni
    # precisione di predizione:
    accuracy = accuracy_score(y_test, y_pred_list)
    print("\nMIXED_DECISION_TREE PERFORMANCE:")
    print("accuracy_score = ", accuracy)
    file.writelines("accuracy_score = " + str(accuracy) + "\n")
    # media armonica:
    f1val = f1_score(y_test, y_pred_list, average='weighted')
    print("f1_score  = ", f1val)
    file.writelines("f1_score = " + str(f1val) + "\n")
    file.close()
    return


if __name__ == '__main__':
    main()
