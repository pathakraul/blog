+++
date = "2017-03-19T15:54:54+05:30"
title = "Understanding Multivariate Linear Regression"
#categories = [""]
tags = ["machinelearning", "regression"]
summary = """Multivariate Linear regression introdues the scenarios where a output response **y** is dependent on many feature
variables **x**"""
mathjax = true
+++

### Multivariate Linear Regression

Linear regression with multiple variables is also known as "multivariate linear regression". Lets introduce notation for equations where we can have any number of input variables.

{{% alert note %}}
Note: \\(\theta^T\\) is a 1 by (n+1) matrix and not an (n+1) by 1 matrix
{{% /alert %}}

<div>
\begin{align*}
x_j^{(i)} &= \text{value of feature } j \text{ in the }i^{th}\text{ training example} \newline 
x^{(i)}& = \text{the column vector of all the feature inputs of the }i^{th}\text{ training example} \newline 
m &= \text{the number of training examples} \newline 
n &= \left| x^{(i)} \right| ; \text{(the number of features)} 
\end{align*}
</div>

The multivariable form of the hypothesis function accommodating these multiple features is as follows:

<div>
\begin{align*}
h_\theta (x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + \cdots + \theta_n x_n
\end{align*}
</div>

In order to develop intuition about this function, we can think about $\theta\_0$ as the basic price of a house, $\theta\_1$ as the price per square meter, $\theta\_2$ as the price per floor, etc. $x\_1$ will be the number of square meters in the house, $x\_2$ the number of floors, etc.

Using the definition of matrix multiplication, our multivariable hypothesis function can be concisely represented as:

<div>
\begin{align*}
    h_\theta(x) =
    \begin{bmatrix}
        \theta_0 \hspace{2em} \theta_1 \hspace{2em} ... \hspace{2em} \theta_n
    \end{bmatrix}
    \begin{bmatrix}
        x_0 \newline 
        x_1 \newline 
        \vdots \newline 
        x_n
    \end{bmatrix}= \theta^T x
\end{align*}
</div>

This is a vectorization of our hypothesis function for one training example; see the lessons on vectorization to learn more.

Remark: Note that for convenience reasons in this course we assume $x_{0}^{(i)} =1 \text{ for } (i\in { 1,\dots, m } )$. This allows us to do matrix operations with $\theta$ and $x$. Hence making the two vectors $\theta$ and $x(i)$ match each other element-wise (that is, have the same number of elements: $n+1$).

The following example shows us the reason behind setting $x_{0}^{(i)}=1$ :

<div>
\begin{align*}
    X = \begin{bmatrix}
            x^{(1)}_0 & x^{(2)}_0 & x^{(3)}_0 \newline 
            x^{(1)}_1 & x^{(2)}_1 & x^{(3)}_1 
        \end{bmatrix} &,\theta = 
        \begin{bmatrix}
            \theta_0 \newline 
            \theta_1 \newline
        \end{bmatrix}
\end{align*}
</div>

As a result, you can calculate the hypothesis as a vector with $h_{θ}(X)=θ^TX$

<br \>

#### Gradient Descent For Multiple Variables ####

The gradient descent equation itself is generally the same form; we just have to repeat it for our 'n' features:

<div>
\begin{align*} 
\text{repeat until convergence:} \; \lbrace \newline 
\; & \theta_0 := \theta_0 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_0^{(i)} \newline 
\; & \theta_1 := \theta_1 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_1^{(i)} \newline 
\; & \theta_2 := \theta_2 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_2^{(i)} \newline 
& \cdots \newline 
\rbrace 
\end{align*}
</div>


In other words:

<div>
\begin{align*}
    \text{repeat until convergence:} \; \lbrace \newline 
    \; & \theta_j := \theta_j - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)} \\
    \; & \text{for j := 0...n}\newline 
    \rbrace
\end{align*}
</div>


### Gradient Descent in Practice I - Feature Scaling

{{% alert note %}}
We can speed up gradient descent by having each of our input values in roughly the same range. This is because \\(\theta\\) will descend quickly on small ranges and slowly on large ranges, and so will oscillate inefficiently down to the optimum when the variables are very uneven.
{{% /alert %}}


The way to prevent this is to modify the ranges of our input variables so that they are all roughly the same. Ideally:

$$ −1 ≤ x_i ≤ 1 $$

or

$$ −0.5 ≤ x_i ≤ 0.5 $$

These aren't exact requirements; we are only trying to speed things up. The goal is to get all input variables into roughly one of these ranges, give or take a few.

Two techniques to help with this are feature scaling and mean normalization. Feature scaling involves dividing the input values by the range (i.e. the maximum value minus the minimum value) of the input variable, resulting in a new range of just 1. Mean normalization involves subtracting the average value for an input variable from the values for that input variable resulting in a new average value for the input variable of just zero. To implement both of these techniques, adjust your input values as shown in this formula:

<div>
\begin{align*}
    x_i & :=(x_i−μ_i)/s_i \newline
    & s_i = \text{Standard Deviation} \newline
    & μ_i = \text{Mean}
\end{align*}
</div>

Where μi is the average of all the values for feature (i) and si is the range of values (max - min), or si is the standard deviation.

Note that dividing by the range, or dividing by the standard deviation, give different results. The quizzes in this course use range - the programming exercises use standard deviation.

For example, if xi represents housing prices with a range of 100 to 2000 and a mean value of 1000, then, xi:=price−10001900.

### Gradient Descent in Practice II - Learning Rate

Note: [5:20 - the x -axis label in the right graph should be θ rather than No. of iterations ]

Debugging gradient descent. Make a plot with number of iterations on the x-axis. Now plot the cost function, J(θ) over the number of iterations of gradient descent. If J(θ) ever increases, then you probably need to decrease α.

Automatic convergence test. Declare convergence if J(θ) decreases by less than E in one iteration, where E is some small value such as 10−3. However in practice it's difficult to choose this threshold value.

<div class="imgcap">
<img src="/assets/lr-gd/costfunction-alpha1.png" style="border:none; width:60%;">
</div>

It has been proven that if learning rate α is sufficiently small, then \\( J(\theta)\\) will decrease on every iteration.

<div class="imgcap">
<img src="/assets/lr-gd/costfunction-alpha2.png" style="border:none; width:60%;">
</div>

To summarize:
If α is too small: slow convergence.
If α is too large: \\( J(\theta)\\) may not decrease on every iteration and thus may not converge.

### Features and Polynomial Regression

We can improve our features and the form of our hypothesis function in a couple different ways. We can combine multiple features into one. For example, we can combine $x\_1$ and $x\_2$ into a new feature $x\_3$ by taking $x\_1*x\_2$.

#### Polynomial Regression

Our hypothesis function need not be linear (a straight line) if that does not fit the data well.

We can change the behavior or curve of our hypothesis function by making it a quadratic, cubic or square root function (or any other form).

For example, if our hypothesis function is $h\_{\theta}(x) = \theta\_0 + \theta\_1 x\_1$ then we can create additional features based on $x\_1$ , to get the quadratic function $h\_{\theta}(x) = \theta\_0 + \theta\_1 x\_1 + \theta\_2 x\_1^2$ or the cubic function $h\_{\theta}(x) = \theta\_0 + \theta\_1 x\_1 + \theta\_2 x\_1^2 + \theta\_3 x\_1^3$. 

<br \> <br \>

In the cubic version, we have created new features $x\_2$ and $x\_3$ where $x\_2=x\_1^2$ and $x\_3=x\_1^3$. To make it a square root function, we could do: $h\_{\theta}(x) = \theta\_0 + \theta\_1 x\_1 + \theta\_2 \sqrt{x\_1}$
<br \>
{{% alert note %}}
One important thing to keep in mind is, if you choose your features this way then feature scaling becomes very important.
{{% /alert %}}
eg. if $x\_1$ has range 1 - 1000 then range of $x\_1^2$ becomes 1 - 1000000 and that of $x\_1^3$ becomes 1 - 1000000000
