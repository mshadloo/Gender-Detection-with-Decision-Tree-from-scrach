# Gender Detection with Decision Tree from scrach

In this repo, I implement a decision tree classifier from scratch. Decision trees are built using a recursive algorithm known as divide and conquer algorithm. 
- I select the best feature based on information gain for root node. I create branch for two possible outcomes of the test (has or doesn't have that feature).
- I split instances into two subsets. One for each branch extending from the node.
- I repeat recursively for each branch, using only instances that reach the branch.
- I stop recursion for a branch if all its instances have the same class or I prune the tree at some given depth. 
##  Dataset
I evaluate the decision tree on gender detection based on names. I use NationalNames data [https://www.kaggle.com/kaggle/us-baby-names?select=NationalNames.csv](https://www.kaggle.com/kaggle/us-baby-names?select=NationalNames.csv) which is  released by data.gov. 
## Experiment

First I extract features from names using some heuristics. I used the first letter, first two letters, first three letters, last letter, last two letters, last three letters, etc. as features.
I built diffrent trees based on the the maximum depth that tree can be extended to. The experiments show that by increasing the maximum depth training accuracy increases as I expected. 

![](/decision_tree.png)
