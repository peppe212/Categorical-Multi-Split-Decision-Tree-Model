# Mixed Decision Tree for Classification

---

## Bachelor's Degree Thesis Project in Computer Science

**University of Pisa**

* **Author**: Giuseppe Muschetta
* **Thesis Supervisor**: Prof. Riccardo Guidotti
* **Key Libraries Used**:
    * **Pandas**: 1.4.4
    * **Numpy**: 1.21.5
    * **Scikit-learn (Sklearn)**: 1.1.1

---

## Project Overview: A Unique Approach to Decision Trees

This project presents a custom-built **Mixed Decision Tree** classifier, developed as part of a Bachelor's degree thesis at the University of Pisa. This implementation stands apart from standard decision tree libraries, such as **scikit-learn**, primarily due to its inherent design for **multi-way splitting** and its flexible handling of mixed data types.

Traditional decision trees, like the `DecisionTreeClassifier` in scikit-learn, are predominantly designed for binary splits (where each node divides data into two branches, typically based on a "yes/no" condition or a numerical threshold). My **Mixed Decision Tree**, on the other hand, is designed to perform **multi-way splits** from the ground up. This means a single node can naturally branch into *multiple* children, accommodating all unique values of a categorical attribute or multiple ranges for a continuous one.

Furthermore, this implementation gives **categorical attributes primary importance in its splitting mechanism**, while also robustly handling continuous data through various transformation modes. This hybrid approach makes it particularly well-suited for datasets with a mix of attribute types, offering a more nuanced and potentially more intuitive tree structure than strictly binary splitting algorithms.

### Key Features and Design Principles

1.  **Flexible Attribute Handling**:
    * **Categorical Attributes**: Must be represented as strings (`str`) in the input dataset. These are handled naturally with multi-way splits.
    * **Continuous Attributes**: Integer (`int`) or float (`float`) attributes are automatically converted to float and can be handled in different ways depending on the `tree_mode` selected.
2.  **Multi-way Splitting**: Each node can have multiple children, enabling more complex decision boundaries compared to strictly binary splits. This is a core differentiator from many off-the-shelf implementations.
3.  **Customizable Tree Construction**:
    * **Impurity Criteria**: Supports both **Gini impurity** and **Entropy** for calculating information gain.
    * **Tree Modes**:
        * `'classic'`: Processes continuous data as is, creating binary splits (e.g., `attribute <= value`). Can apply a dataset `cut_threshold`. This mode is still multi-way for categorical features.
        * `'binned'`: Discretizes continuous attributes into a specified number of bins, treating each bin as a categorical value for multi-way splitting.
        * `'median'`: Converts continuous attributes into binary features based on their median value (e.g., `attribute <= median_value`), but still within the multi-split framework for other features.
    * **Pruning Hyperparameters**: `max_allowed_depth`, `min_samples_split`, and `min_samples_leaf` for controlling tree complexity and preventing overfitting.

---

## Code Structure

The core of the project is encapsulated within the `MixedDecisionTree` class, defined in the main Python script (likely `main.py` or `mixed_decision_tree.py`).

### `MixedDecisionTree` Class

* **`Node` Inner Class**: Represents a single node in the decision tree, storing information such as `split_condition`, `gain`, `isLeaf`, `classification`, `parent`, `depth`, and `children`.
* **`__init__`**: Initializes the tree with various parameters like `class_name`, `max_allowed_depth`, `min_samples_split`, `min_samples_leaf`, `tree_mode`, `number_of_bins`, `cut_threshold`, and `criterion`.
* **Auxiliary Methods (`_get_impurity`, `_getGain`, `_getMostCommonValue`, `_stoppingCriteria`)**: These private methods handle impurity calculation (Entropy/Gini), information gain calculation, determining the most common class value for leaves, and checking the stopping conditions for tree growth (pruning).
* **`fit(X_train, y_train)`**: The main method for training the decision tree. It prepares the input data and calls the recursive `_fit` method.
* **`_fit(data, features, current_depth)`**: The recursive function responsible for building the tree. It selects the best split attribute, creates child nodes, and continues recursively until a stopping criterion is met.
* **Data Preprocessing Methods (`_median_mode`, `_classic_mode`, `_binned_mode`, `select_tree_modes`)**: These methods prepare the dataset (`X_train`, `X_test`, `y_train`, `y_test`) according to the chosen `tree_mode` and handle continuous attribute transformation.
* **`predict(X_test, y_test)`**: Makes predictions on new, unseen data (`X_test`) by traversing the trained tree for each example. It calls the recursive `_predict` method.
* **`_predict(root, example)`**: A recursive helper for `predict` that navigates the tree based on an input example's features.
* **`displayDT(root, file, width)`**: A method for visualizing the structure of the trained decision tree, printing it to the console and writing it to a text file.

### Helper Functions

* **`initial_infos(file)`**: Prints and writes initial configuration parameters of the decision tree to a file.
* **`settings()`**: A utility function to configure all hyperparameters and dataset paths for the `MixedDecisionTree`. This is the primary place to adjust settings.
* **`main()`**: The entry point of the script, handling dataset loading, tree instantiation, training, and prediction. It also measures the training time and outputs results.

---
