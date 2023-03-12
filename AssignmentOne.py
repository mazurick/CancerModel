import pandas as pd
import numpy as np
from sklearn import datasets 
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_breast_cancer



# Write code that fetches the breast cancer wisconsin dataset. 
 
X,y=datasets.load_breast_cancer(return_X_y = True) 
cancer = load_breast_cancer()   #This line of code is loading the Breast Cancer Wisconsin (Diagnostic) dataset from the scikit-learn library. This dataset contains information about various features of breast cancer tumors, such as radius, texture, smoothness, etc., and whether or not the tumor is malignant. This data can then be used to build various machine learning models to predict if a tumor is malignant or benign.
number_of_instances = len(X) # This line of code is counting the number of elements in the list X and assigning that number to the variable number_of_instances. The len() function is used to count the number of elements in a list.
number_of_features = len(cancer['feature_names']) #This line of code is calculating the number of features in the cancer dataset. The variable "cancer" is a dictionary containing the data and the feature names are one of the items in that dictionary. The function len() is used to calculate the length of the list containing the feature names, which is then assigned to the variable "number_of_features".

# Check how many instances we have in the dataset, and how many features describe these instances
print("There are", number_of_instances, "instances described by", number_of_features, "features.")  #(5 points)  

# Create a training and test set such that the test set has 40% of the instances from the 
# complete breast cancer wisconsin dataset and that the training set has the remaining 60% of  
# the instances from the complete breast cancer wisconsin dataset, using the holdout method. 
# In addition, ensure that the training and test sets # contain approximately the same 
# percentage of instances of each target class as the complete set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state = 42, stratify = y)   #(5 points) 

# Create a decision tree classifier. Then Train the classifier using the training dataset created earlier.
# To measure the quality of a split, using the entropy criteria.
# Ensure that nodes with less than 6 training instances are not further split
clf = tree.DecisionTreeClassifier(criterion = 'entropy', min_samples_split = 6)  #(5 points) 
clf = clf.fit(X_train, y_train)  #(4 points) 

# Apply the decision tree to classify the data 'testData'.
y_test_pred = clf.predict(X_test) #(4 points) 

# Compute the accuracy of the classifier on 'testData'
print("The accuracy of the classifier is", accuracy_score(y_test, y_test_pred))  #(3 points) 

# Visualize the tree created. Set the font size the 12 (5 points) 

#### IMPORTANT ########

""" 

Please note I did this plot below on lines 66 and 67, however, when I run the file and then complete the 
plot in Part 2.1 it creates a graphic overlay problem that I think fixing is beyond this project's scope.

If you want to verify that this plot works, please do steps 1 and 2 below.

1. comment out lines 74-157 

2. Uncomment lines 66 and 67 and run the script 

To verfify this works, I will upload plots upon submission at Plot1.png

"""

# plt.figure(figsize=(30,20)) # then we plot it and give it dimensions suitable to see.
# tree.plot_tree(clf, fontsize=12) # we enlarge the tree plot font for reading comprehension 

text_representation = tree.export_text(clf) # we make a text visualization 
print(text_representation)

#### THANK YOU ########

### PART 2.1 ###
# Visualize the training and test error as a function of the maximum depth of the decision tree
# Initialize 2 empty lists where you will save the training and testing accuracies 
# as we iterate through the different decision tree depth options.
testAccuracy = []  #(1 point) 
trainAccuracy =[] #(1 point) 
# Use the range function to create different depths options, ranging from 1 to 15, for the decision trees
depthOptions = list(range(1,16))  #(1 point) 

for depth in depthOptions: #(1 point) 
    # Use a decision tree classifier that still measures the quality of a split using the entropy criteria.
    # Also, ensure that nodes with less than 6 training instances are not further split
    cltree = tree.DecisionTreeClassifier(criterion = 'entropy', min_samples_split = 6, max_depth=depth) # Here I use min_sample_split  #(1 point) 
    # Decision tree training
    cltree = cltree.fit(X_train, y_train)  #(1 point) 
    # Training error
    y_predTrain = cltree.predict(X_train)#(1 point) 
    # Testing error
    y_predTest = cltree.predict(X_test)   #(1 points) 
    # Training accuracy
    trainAccuracy.append(accuracy_score(y_train, y_predTrain))  #(2 points) 
    # Testing accuracy
    testAccuracy.append(accuracy_score(y_test, y_predTest)) #(2 points) 

x = np.arange(len(depthOptions)) + 1 # Create domain for plot
plt.plot(x, testAccuracy, label='Testing Accuracy')
plt.plot(x, trainAccuracy, label='Training Accuracy') # Plot testing error over domain
plt.xlabel('Maximum Depth') # Label x-axis
plt.ylabel('Total Error') # Label y-axis
plt.legend() # Show plot labels as legend
plt.show() # Show graph

# Fill out the following blanks: #(6 points (3 points per blank)) 
""" 
According to the test error, the best model to select is when the maximum depth is equal to 13, approximately. 
But, we should not use select the hyperparameters of our model using the test data, because this is considered cheating.
"""

### PART 2.2 ###
# Use sklearn's GridSearchCV function to perform an exhaustive search to find the best tree depth.
# Hint: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# First define the parameter to be optimized: the max depth of the tree
parameters = {
    'max_depth': list(range(1,16)),
    'criterion': ["entropy"], 
    #'max_features': list(range(1,number_of_features+1)), 
    'splitter': ["best"], 
    'min_samples_split':[6]

} #(6 points)


# We will still grow a decision tree classifier by measuring the quality of a split using the entropy criteria. 
# Also, continue to ensure that nodes with less than 6 training instances are not further split.
clf = GridSearchCV(
    estimator=tree.DecisionTreeClassifier(),
    param_grid=parameters,
    cv=10, 
    n_jobs=5,
    verbose=3,
)

clf.fit(X_train, y_train)


tree_model = clf.best_estimator_   
print("The maximum depth of the tree should be", clf.best_params_['max_depth']) 
 
print(clf.best_params_) #(5 points)

# The best model is tree_model. Visualize that decision tree (tree_model). Set the font size the 12 
plt.figure(figsize=(23,12))
tree.plot_tree(tree_model, feature_names=cancer['feature_names'], filled=True, fontsize=12)  #(5 points)

# Fill out the following blank: #(3 points)
""" 
This method for tuning the hyperparameters of our model is acceptable, because it identifies the optimal hyperparameters for a model while guaranteeing the same percentage of class labels in the training and test set. 
"""

# Explain below was is tenfold Stratified cross-validation?  #(5 points)
"""
Yes, this file uses tenfold stratified cross-validation. Tenfold stratified cross validation is a method of model evaluation in which we divide one data set randomly into 10 parts. We use 9 of those parts to train a model and save the other 1/10th for testing. We repeat this process 10 times, each time reserving a different 1/10th for testing.
______
"""

############## FOR GRADUATE STUDENTS ONLY (the students enrolled in CPS 8318) ##############
### PART 3: decision tree classifier to solve a classification problem ###
""" 

For this section please see the following files:

1.  report_Adam_Mazurick.pdf    - This is my write-up.

2.  part_three_classifier.py    - This is the classifier I built for Part 3.

3.  data_gen.py                 - This is the script I used to generate my dataset. 

4.  test_data.csv               - This is the dataset I used.


"""
