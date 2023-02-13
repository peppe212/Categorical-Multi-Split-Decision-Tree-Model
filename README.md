# MixedDecisionTree
My Computer Science Bachelor's Thesis: A full Python Implementation of a Categoric, Continuous, Multi-Split, Decision Tree

Python implementation of a full functional Machine Learning Decision Tree Classifier with extended capabilities that makes it differ 
from the one implemented in the famous library Scikit- Learn. 
My own Decision Tree can get as input any type of attribute, both categoric and continuous. 
It also supports multi-split nodes, which is the real deal, in fact, this tree has been programmed from scratch with this 
philosophy in mind: “make it to be as close as possible to the theory of how Decision Trees are presented”. 
It means native support for categoric attributes, and every node (root or any other internal node) can have an arbitrary number of children; 
that is, it even supports multi-split. 
Scikit-Learn built-int Decision Tree Classifier does not support multi-split (all its splits are binary) and categorical 
attributes (it can only process, internally, continuous attributes).
