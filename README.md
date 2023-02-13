# What is this all about?
I developed from scratch, using the Python programming language, a full "Decision Tree Classifier" with extended capabilities 
that makes it differ from the one implemented in the famous machine learning library called Scikit-Learn. 
So in the end we're talking about machine learning algorithms here.

My own Decision Tree can get as input a dataset with any kind of attribute, both categoric and continuous. 
It also supports multi-split nodes, which is the real deal, in fact, this tree has been written wirth this  
philosophy in mind: “make it to be as close as possible to the theory of how Decision Trees are presented”. 

It means native support for categoric attributes and every node (root or any other internal node) can have an 
arbitrary number of children; that is, it even supports multi-split and not just binary splits. 

Scikit-Learn built-in Decision Tree Classifier does not support directly multi-split and categorical attributes, 
infact it can only process, internally, continuous attributes with only binary splits.
