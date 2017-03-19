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

Linear regression with multiple variables is also known as "multivariate linear regression". We now introduce notation for equations where we can have any number of input variables.

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

In order to develop intuition about this function, we can think about θ0 as the basic price of a house, θ1 as the price per square meter, θ2 as the price per floor, etc. x1 will be the number of square meters in the house, x2 the number of floors, etc.

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


