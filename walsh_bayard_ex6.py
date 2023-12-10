
# Note that we import the numpy package as 'np', so it is referenced by 'np'
# in the subsequent code
import numpy as np
import math


# This function takes a vector as input and returns the softmax output for that
# vector
def softmax(v):
    dist = []
    for i in v:
        prob = math.exp(i)/sum([math.exp(e) for e in v])
        dist.append(prob)
    return np.array(dist)

# This function takes a vector and applies the sigmoid to each element,
# returning the vector of sigmoid outputs
def sigmoid_elementwise(v):
    return np.array([1/(1+math.exp(-1*e)) for e in v])

def logistic_regression(w,b,x):
    # 1) Implement the logistic regression function, as a function of the
    # vector w of weights, bias b, and input vector x. You can use
    # np.dot() to do the dot product between the vectors. You will need to
    # implement the sigmoid function, for which you can use math.exp(x) for
    # e**x. Make sure that the final value you get matches the one reported in
    # the textbook section 5.1.1.
    # Assign the final output value to the variable 'y', which the
    # function will return.

    z=np.dot(w, x)+b
    y= 1/(1+math.exp(-z))
    return y

def ffn(W,U,b,x):
    # 2) Implement the two-layer feed-forward network described in the
    # equations in 7.10. You can use np.matmul() for the multiplication of the
    # weight matrices and vectors (Wx and Uh). I've implemented an elementwise
    # sigmoid function ('sigmoid_elementwise()') which you can use to apply the
    # sigmoid to the relevant vector. I've also implemented the softmax function
    # for you ('softmax()'), so you can use softmax() to apply the softmax
    # function to the relevant vector.
    # Assign the final output value to the variable 'y', which the
    # function will return.

    h=sigmoid_elementwise(np.matmul(W,x)+b)
    z=np.dot(U, h)
    y=softmax(z)
    return y

# This is x, the input vector from Sec 5.1.1
x = np.array([3,2,1,3,0,4.19])

# These are the logistic regression weights used in Sec 5.1.1
w_lr = np.array([2.5,-5.0,-1.2,0.5,2.0,0.7])
# This is the bias from Sec 5.1.1
b_lr = .1

# This is a bias vector for the neural network
b = [.1,.1,.1]

# This is the W matrix of layer-1 weights, to transform the input. Note that the
# first row of weights is the same as the logistic regression weights, so the
# first dimension of your hidden layer h should have the same value as your
# logistic regression output.
W = np.array([
[2.5,-5.0,-1.2,0.5,2.0,0.7],
[1.0,-2.0,-3.3,1.5,1.8,0.2],
[2.1,1.0,-1.2,-0.8,2.0,1.4]
])

# This is the U matrix of layer-2 weights, to transform the output of the
# first layer
U = np.array([
[1.,2.,1.,],
[2.,1.,2.]
])

lr_output = logistic_regression(w_lr,b_lr,x)
print('LOGISTIC REGRESSION OUTPUT: %s'%lr_output)

nn_output = ffn(W,U,b,x)
print('NEURAL NETWORK OUTPUT: %s'%nn_output)

# 3) Below are three questions and one task: please print answers to the three
# questions, and complete the task.

# 3a) You can think of the neural network output as a probability distribution for a
# discrete random variable with two values. Let's say the variable is Sentiment,
# with the following ordering on the domain of the variable: <positive, negative>.
# (Remember Sec 13.2.2 in the Russell & Norvig reading). Which sentiment label
# (positive or negative) is the model currently assigning the highest probability?
# Print your answer.
answer3a= ("\n3a: Currently the model is assigning the probabilites [0.28097038 0.71902962].\n"
           "Therefore, if we have labels <positive, negative>, then the sentiment would\n"
           "favor negative with 71.9% probability based on discrete outputs from the model\n"
           "Therefore negative is assigned the highest probability by the model.\n \n")

# 3b) We could just as easily have said that the first dimension of the output is
# negative sentiment, and the second dimension is positive. In practice, how do we
# connect models' output dimensions to the labels in our task? Print your answer.

answer3b=("\n3b: An aspect of determining the output of the model is considering the ordering of training labels.\n"
            "When training a neural network, because of the linear algebraic operations preformed,\n"
            "the output of the network's neurons correspond to the order in which labels are\n"
            "presented in the training data. If the training data labels negative or positive sentiments (ranging from 0 to 1) for a given feature,\n" 
            "then the same level of the corresponding feature dimension would align with this distribution.\n"
            "Note that how and why certain features are more or less prominent in model behavior is an open question in research\n")


# 3c) Suppose that the correct sentiment for this input x is actually the opposite
# of the label currently receiving highest probability. Modify weight matrix U2
# below such that the updated network assigns higher probability to the correct label.
answer3c=("\n3c: In this case, with only two features, we can switch the order that the featyres are fed in\n"
          "to get the exact opposite outcomes, of [0.71902962 0.28097038]. Note that this approach is not \n "
          "usually sucesfull for a larger network but in this case the manual change works. \n \n")

# 3d) Above you manually changed the weights to get a better output. In practice,
# what more efficient alternative do we have for adjusting the weights of the
# network to improve the outputs? Print your answer.

answer3d= ("\n3d: Instead of manually changing the weights, a better practice would be\n"
           "to employ backpropagation. This entails the process of adding or subtracting from the\n"
           "weights based on how the model has performed after various epochs (or runs through the model).\n"
           "This is possible in supervised learning because we can measure the difference between our output\n"
           "And the true output and then change the weights (rate of change/ derivative) to better model the training data\n"
           "Oftentimes it's not clear exactly how a specific weights correlate to specific outputs in a larger model\n"
           "and manually finetuning a large model is usually a futile effort. Additionally, based on loss\n"
           "functions we can use optimization (such as gradient descent) to minimize this loss, causing us to eventually converge on ideal\n"
           "weights after enough iterations.\n \n")



U2 = np.array([
[2.,1.,2.],    
[1.,2.,1.,]
])

nn_output2 = ffn(W,U2,b,x)
print('UPDATED NEURAL NETWORK OUTPUT: %s'%nn_output2)

print(answer3a+answer3b+answer3c+answer3d)
