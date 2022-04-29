# The Super Learner
 The super-learner model is a variation of stacking or k-fold cross-validation where individual models would be trained on k-fold data split. A final meta-model would then be trained on their output, also called an out-of-fold prediction from each model. Overall, the super learner is an important tool implemented in a very limited number of studies to reduce parametric assumptions, boost predictive accuracy, and avoid overfitting. In this work, a variant of the super learner model was developed strictly to predict 3C steel metal's corrosion rate in seawater using five algorithms chosen by the developed selection framework. All five algorithms are trained, tested individually on the same dataset, and compared with the super learner model; they are combined to create five other algorithms trained and tested on the same dataset. These five selected algorithms performed the best individually on the same dataset to be put forward by the selection framework for implementation in developing the super learner model. The algorithms chosen by the developed selection framework are:

 - Linear regression
 - Ridge regression
 - Lasso regression
 - Support vector regression using the polynomial kernel
 - Extra Gradient Boosting

 # Random Search Hyperparameter Optimization
 Random search is a basic improvement on grid search. It is based on a random search of hyperparameter values from user-defined distributions. It involves the user providing a statistical distribution of the hyperparameter values for the optimization method or algorithm to test. These distributions are created for each hyperparameter, after which the optimization algorithm or method tests them at every iteration by generating a model. The iteration continues until the predefined (by user) distributions for each hyperparameter are exhausted or until the desired accuracy is reached. Random search was proven to be better than grid search because,: firstly, in random search, a range can be assigned independently according to the distribution of search space, therefore making random search perform better in cases where some hyperparameters are not uniformly distributed

 # Computation time and Memory Consumption Optimization
 Computation time and memory consumption have been a genuine issue over the years in transferring machine learning models onto portable physical systems. Most machine learning models' size mitigates their utilization in basic physical systems, therefore requiring complex-expensive systems with greater hardware requirements to carry these models. In comparison with the model developed to predict the same dataset's corrosion rate, the super learner’s architecture facilitates efficient utilization of memory at less computation time by its models using multilayered ensembles (ML-Ensembles).
 Computation time and memory consumption issues are common, mostly with a moderately sized dataset, becoming more prominent with an increase in the training dataset's size. When ensembles such as bagging, voting, etc., are executed, serialization of the training dataset occurs (i.e., training ensembles in parallel), which are then stored subprocess (memory). As the number of ensembles in parallel increases, multiple copies of the training dataset are stored in the same subprocess, leading to increased memory consumption at slower computation time when the model is called for execution.
 Multilayered Ensembles (ML-Ensemble) is a python library that utilizes memmapping to builds models in the form of a feed-forward network. Layers are stacked sequentially, with each layer taking the previous layer’s output as input. With the aid of multilayered ensembles, ensembles of any shape and form can be built, features can be propagated through layers, estimation method can be varied between layers, preprocessing can be differentiated between a subset of base learners but most importantly, the computation time and memory consumption are optimized. Also, the serialization of the training data, sending the serialized data to the subprocess, and copying of the dataset (leading to an increase in the number of subprocesses) are all avoided, leading to the memory consumption being constant.
 Ultimately, the knowledge on the architecture of the super leaner provided in this chapter is essential in developing a super learner model that not only provides a comparable prediction accuracy (compared to other model trained on the same dataset) but also efficiently utilizes system memory (at lesser computation time) while predicting the corrosion rate of 3C steel in different environmental conditions.

## Below are the models developed from supervised learning algorithms utilized in the development of the super learner model:
 - bagging meta estimator model
 - adaboost ensemble model
 - gradient boosting machine ensemble model
 - extra gradient boosting model (xgbm) ensemble model
 - classification and regression tree (cart) model: maximum depth of 2
 - classification and regression tree (cart) model: maximum depth of 5
 - convolutional neural network model
 - feed forward neural network model
 - hybrid voting regressor model
 - multivariate adaptive regression spline model
 - support vector regression model using linear kernel
 - support vector regression model using polynomial kernel
 - support vector regression model using rbf kernel
 - huber regression model
 - lasso regression model
 - linear regression model
 - multilayer perceptron model
 - polynomial regression model
 - ridge regression model
