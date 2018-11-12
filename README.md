# Machine Learning with Octave/Matlab

[The credit for preparing the exercises goes to Andrew Ng of Stanford University and DeepLearning.ai through the Machine Learning course on Coursera. I have completed the exercises to complete the tasks.]

## sample1
This file will run samples to perform 
- Plotting Data
- Conputing Cost and Gradient descent
- Visualizing J(theta_0, theta_1)
For a given set of points we first plot the data points and then find a linear regression that minimizes the sum of the squares of distances (cost: least square) of the points from the line, using gradient descent. Then we will visulalize the cost function J.
<img src="sample1_fig1.jpg" width="600" alt="Data points and the line of regression" align="middle">
Here is the cost as a function of y-intercept (theta_0) and the slope (theta_1) of the line:
<img src="sample1_fig2.jpg" width="600" alt="Cost Function" align="middle">
And the contour lines with the minimum value found:
<img src="sample1_fig3.jpg" width="600" alt="Contour lines of the cost function" align="middle">
This is the output:

    Plotting Data ...
    Program paused. Press enter to continue.

    Testing the cost function ...
    With theta = [0 ; 0]
    Cost computed = 32.072734
    Expected cost value (approx) 32.07

    With theta = [-1 ; 2]
    Cost computed = 54.242455
    Expected cost value (approx) 54.24
    Program paused. Press enter to continue.

    Running Gradient Descent ...
    Theta found by gradient descent:
    -3.630291
    1.166362
    Expected theta values (approx)
     -3.6303
      1.1664

    For population = 35,000, we predict a profit of 4519.767868
    For population = 70,000, we predict a profit of 45342.450129
    Program paused. Press enter to continue.
    Visualizing J(theta_0, theta_1) ...



## sample1_multi
This file will run samples to perform 
- Feature Normalization
- Gradient Descent
- Normal Equations
Here we load a set of data of some features of houses and their prices. First, we will scale the features to normalize them, then run a gradient descent to minimize the cost function. And eventually compute the regression line directly to compare. At the end you can see the estimate for price of a house not in the data set.

Here you can see the cost vs. number of iterations. This is good way to check if our code is working correctly:
<img src="sample1_multi_fig1.jpg" width="600" alt="Cost vs. Number of iterations" align="middle">


This is the output:

    Loading data ...
    First 10 examples from the dataset: 
     x = [2104 3], y = 399900 
     x = [1600 3], y = 329900 
     x = [2400 3], y = 369000 
     x = [1416 2], y = 232000 
     x = [3000 4], y = 539900 
     x = [1985 4], y = 299900 
     x = [1534 3], y = 314900 
     x = [1427 3], y = 198999 
     x = [1380 3], y = 212000 
     x = [1494 3], y = 242500 
    Program paused. Press enter to continue.
    Normalizing Features ...
    Running gradient descent ...
    Theta computed from gradient descent: 
     334302.063993 
     100087.116006 
     3673.548451 


    question =

       -0.4413   -0.2237

    Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):
     $289314.620338
    Program paused. Press enter to continue.
    Solving with normal equations...
    Theta computed from the normal equations: 
     89597.909544 
     139.210674 
     -8738.019113 

    Predicted price of a 1650 sq-ft, 3 br house (using normal equations):
     $293081.464335


## sample2
This file will run samples to perform logistic regression. It includes:
- Plotting Data
- Compute Cost and Gradient
- Optimizing using fminunc
- Predict and Accuracies

Here we have data on students being admitted to a program and two test scores. We predict the probability that a student with some scores will get admitted or not. 

Here you can see the test scores and whether a student is admitted or not. Then, we have found a "cut-off line" and based or predicctions on which side of the line a new student will fall, based on their test scores. 
<img src="sample2_fig1.jpg" width="600" alt="Cost vs. Number of iterations" align="middle">

Then we predict for a student with scores 45 and 85, an admission probability of 0.776291, with train accuracy: 89%.
<img src="sample2_fig2.jpg" width="600" alt="Prediction for a new student" align="middle">

Here is the output:

	Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.

	Program paused. Press enter to continue.
	Cost at initial theta (zeros): 0.693147
	Expected cost (approx): 0.693
	Gradient at initial theta (zeros): 
	 -0.100000 
	 -12.009217 
	 -11.262842 
	Expected gradients (approx):
	 -0.1000
	 -12.0092
	 -11.2628

	Cost at test theta: 0.218330
	Expected cost (approx): 0.218
	Gradient at test theta: 
	 0.042903 
	 2.566234 
	 2.646797 
	Expected gradients (approx):
	 0.043
	 2.566
	 2.647

	Program paused. Press enter to continue.

	Local minimum found.

	Optimization completed because the size of the gradient is less than
	the default value of the optimality tolerance.

	<stopping criteria details>

	Cost at theta found by fminunc: 0.203498
	Expected cost (approx): 0.203
	theta: 
	 -25.161343 
	 0.206232 
	 0.201472 
	Expected theta (approx):
	 -25.161
	 0.206
	 0.201

	Program paused. Press enter to continue.
	For a student with scores 45 and 85, we predict an admission probability of 0.776291
	Expected value: 0.775 +/- 0.002

	Train Accuracy: 89.000000
	Expected accuracy (approx): 89.0
	
## sample2_reg
This file will run samples to perform logistic regression with regularization. It includes:
- Regularized Logistic Regression
- Polynomial Features
- Regularization and Accuracies

Here we perform two tests on a set of microchips and we want to classify the microchips. The data looks like this:
<img src="sample2_reg_fig1.jpg" width="600" alt="Test scores for microschips and their failure" align="middle">

Then we train a logistic regression algorithm with regulariziation paramter lambda to classify them. The change in lambda clearly shows the sensitivity of the algorithm for over/under fitting. Here are the decision boundaries for a few sample lambdas:

<img src="sample2_reg_fig2.jpg" width="270" alt="Test scores for microschips with decision boundary" align="middle"> <img src="sample2_reg_fig3.jpg" width="270" alt="Test scores for microschips with decision boundary" align="middle"> <img src="sample2_reg_fig4.jpg" width="270" alt="Test scores for microschips with decision boundary" align="middle">
<img src="sample2_reg_fig5.jpg" width="270" alt="Test scores for microschips with decision boundary" align="middle">
<img src="sample2_reg_fig6.jpg" width="270" alt="Test scores for microschips with decision boundary" align="middle">
<img src="sample2_reg_fig7.jpg" width="270" alt="Test scores for microschips with decision boundary" align="middle">
<img src="sample2_reg_fig8.jpg" width="270" alt="Test scores for microschips with decision boundary" align="middle">
<img src="sample2_reg_fig9.jpg" width="270" alt="Test scores for microschips with decision boundary" align="middle">

Here is the output for lambda = 1:

	Cost at initial theta (zeros): 0.693147
	Expected cost (approx): 0.693
	Gradient at initial theta (zeros) - first five values only:
	 0.008475 
	 0.018788 
	 0.000078 
	 0.050345 
	 0.011501 
	Expected gradients (approx) - first five values only:
	 0.0085
	 0.0188
	 0.0001
	 0.0503
	 0.0115

	Program paused. Press enter to continue.

	Cost at test theta (with lambda = 10): 3.164509
	Expected cost (approx): 3.16
	Gradient at test theta - first five values only:
	 0.346045 
	 0.161352 
	 0.194796 
	 0.226863 
	 0.092186 
	Expected gradients (approx) - first five values only:
	 0.3460
	 0.1614
	 0.1948
	 0.2269
	 0.0922

	Program paused. Press enter to continue.

	Local minimum found.

	Optimization completed because the size of the gradient is less than
	the default value of the optimality tolerance.

	<stopping criteria details>

	Train Accuracy: 83.050847
	Expected accuracy (with lambda = 1): 83.1 (approx)