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
