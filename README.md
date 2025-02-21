# module-13 : Logistic Regression
## Overview
Logistic regression is somewhat like linear regression, but linear regression is ‘unbounded,’ meaning that the value the model returns could be anything based on the input of the data. In logistic regression, the value that the model returns is between 0 and 1. This allows you to set a threshold between two classes more easily. Later in this module, you will apply logistic regression to more than two classes.

## Motivation:
Logistic regression is a supervised machine learning model for determining the probability of the dependent variable. Its major motivation is predicting the likelihood of categorical outcomes that often present binary data such as yes/no and true/false. Some well-known use cases for logistic regression are email spam classifiers and tools for medical diagnoses (e.g., malignant or benign cancers).

The main difference between KNN and Logistic Regression is that KNN classifies a data point based on its proxmity to its nearest neighbors, which can be an expensive computation.
Logistic Regression, like linear regression uses coefficients to classify data points.  What that means is that the model's output will be a continuously-valued probability that the input belongs to a certain class.

The advantage that logistic regression has over linear regression is as follows:
- When trying to determine the decision threshold between two classes, the linear regression model is more sensitive to outliers meaning the threshold changes significantly when outliers are introduced to the dataset.
- The logistic regression model produces "class membership probabilities" instead of bear decision thresholds which becomes significant when building categories of more than two classes of objects.

## Easy Logistic Regression
Simple logistic regression is a supervised machine learning method for predicting the probability of a certain event or class. Although it uses an equation that is similar to linear regression, it uses a logistic function or sigmoid function, which models the odds of the probability of the event. It is used when the data is linearly separable, and the outcome is a dichotomous variable, meaning the outcome is simply yes or no (or 1 or 0). It works on the basic assumptions of linearity and independence (independent variables are not highly correlated). Therefore, logistic regression is typically used for problems that involve binary classification, which helps predict output variables that are discrete into two classes. Examples of binary classifications are yes/no, spam/no spam, and cancerous/noncancerous.

## One Feature, Two Classes
### Classification using bins
From the range of feature values you determine the number of bins and then calculate the probability from that feature whether the object is of class 0 or 1.

The formula that determine if Y equals Class 1 is as follows:
P(Y | bin = i) = Yi / n  : where Yi = total number of objects of class 1 found in bin i and n = total number of objects in bin i

Note: when calculating the number of objects of Class 0 in bin i, the forumla would then be:  1 - P(Y | bin = i)

Once this is dermined the threshold for the feature value should be placed in the bin where there are equals number of objects for each class.  This is where the probability for each class = 0.5

p(X|Y = 0) ~ N(u0, v0^2).  This reads as the the probability X ( p(X) ) given Y = 0 is normally distributed with mean: u0 and variance: v0^2
p(X|Y = 1) ~ N(u1, v1^2).  This reads as the the probability X ( p(X) ) given Y = 1 is normally distributed with mean: u1 and variance: v1^2

where:
X = feature
Y = the class where in this class there are 2 classes (0 and 1)
dx = marginal distribution of X

assumptions:
v0^2 = v1^2 and they shall both be denoted as v^2

We now must determin the probability Y = class 0 or 1 in some bin i.  This shall be denoted as sigma or o

formula for the sigmoid function (aka: Logistic function):
p(Y|X = x)  = o(X) 
            = Y1 / (Y1 + Y0)  // which equals the ratio of class 1
            = 1 / (1+(Y0/Y1))
            = Y1 / Y0
            = P(X=x | Y=0) / P(X=x | Y=1) //this is known as the ODDS RATIO
            = 1 / (1 + ODDS RATIO)
            = 1 / (1 + e^-z)
assumption:
- Marginal distributions are gausian
- ODDS RATIO:  e^-z where z = B0 + B1x //or a linear function of x with coeficients B0 and B1

where:
B0 = (u0^2 - u1^2) / 2v^2  // B = beta
B1 = (u1^2 - u0^2) / v^2  // B = beta

when: o(xbar)   = 0.5  // threshold
                = 1 / (1 + e^-z)    => zbar = 0
                                    => 0 = B0 + (B1 xbar)
                                    => xbar = -(B0 / B1)

To determin the approximate values of B0 and B1 we use the APPROXIMATE SOLUTION where the betas shall be denoted using aB0 and aB1 and 
the letter: a shall be equivalent to ^.

formula:
aB0 = (au1^2 - au2^2) / 2av
aB1 = (au2 - au1) / av

From here we take the likelihood fx to optimize the candidate logistic function for determining the probability of Y.  This is done by finding B0 and B1 that minimizing the CROSS ENTROPY.

Code:
from sklearn.linear_model import LogisticRegression
...

df = some_dataset()
X_features = df[['x1']]
y_classes = df['y_classes']

// run LR class and inspect the coefficients
lr = LogisticRegression().fit(X, y)
lr_beta0 = lr.intercept_[0]
lr_beta1 = lr.coef_[0, 0]
lr_thresh = -ls_beta0/lr_beta1
lr_beta0

// compare to approximate values
X1 = X[y==1].to_numpy()
X2 = X[y==2].to_numpy()

mu1 = np.mean(X1)
mu2 = np.menan(X2)

v1 = np.var(X1)
v2 = np.var(X2)
v = np.mean((v1, v2)) // mean variance

approx_beta0 = (mu1**2 - mu2**2) / 2 / v
approx_beta1 = (mu2 - mu1) / v
approx_threshold = -approx_beta0 / approx_beta1

// sigmoid function
def sigmoid(beta0, beta1, x) :
    return 1 / (1 + np.exp(-beta0 - beta1*x))

// chart output
x = np.linspace(3, 7)
plt.figure(figsize=(8, 6))
plt.scatter( X1, np.zeros(50), label='versicolor' )
plt.scatter( X2, np.zeros(50), label='virginica' )
plt.plot(x, sigmoid( lr_beta0, lr_beta1, x), linewidth=3, color='green', label='logistic regression' )
plt.plot(x, sigmoid( approx_beta0, approx_beta1, x), linewidth=3, color='magenta', label='approximation' )
plt.legend(fontsize=14)
plt.grid()





