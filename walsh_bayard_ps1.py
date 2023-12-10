import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        ## 1a) Here, add one line to initialize the Linear layer of the desired
        ## 8x8 size (with bias), and one line to initialize the Sigmoid
        ## activation function, both to be used in the forward pass below
        self.linear_layer = nn.Linear(8, 8, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ## 1b) Here, define the forward pass of the network using the two
        ## functions defined above. This pass should take x as input, pass it
        ## through the Linear layer and the Sigmoid, and then assign the output
        ## also to x to be returned by the forward pass
        x = self.sigmoid(self.linear_layer(x))

        return x

## Here you create a network object of class Net(). Creating this also runs the
## __init__ method of that class, so your Linear and Sigmoid functions will
## be initialized as well
RM_net = Net()

## Here you define the optimizer as stochastic gradient descent on all
## parameters of the network 'RM_net', with learning rate of 10
RM_optimizer = optim.SGD(RM_net.parameters(), lr=10.)

## Here we define the loss function as L1 loss
loss_criterion = nn.L1Loss()

## This function trains the network on a specified set of training pairs (input/output mappings), for
## a specified number of epochs
def run_training(mapping_pairs,num_epochs):
    for epoch in range(num_epochs):
        for input_rep,target_rep in mapping_pairs:
            input_rep = input_rep.view(1,-1) # here we just add a "batch" dimension (batch size of 1, so each item is its own "batch")
            target_rep = target_rep.view(1, -1) # here we just add a "batch" dimension (batch size of 1, so each item is its own "batch")

            ## 2a) The lines below are pasted directly from the Neural Networks
            ## section of the tutorial. For the training to run, you need to
            ## modify these lines to reconcile with the different names assigned
            ## to the relevant components above
            RM_optimizer.zero_grad()
            output = RM_net(input_rep)
            loss = loss_criterion(output, target_rep)
            loss.backward()
            RM_optimizer.step()

            ## 2b) Here, print out the value that tells you how far your
            ## network's output is from the target value
            print(f"Loss value: {loss.item()}")

    ## 2c) Here use the .parameters() method to print out the weights and biases
    ## of the trained network. Print out the weight parameters first, and then
    ## print out the bias parameters separately.
    for name, param in RM_net.named_parameters():
        if 'weight' in name:
            print("Weights:", param.data)

    for name, param in RM_net.named_parameters():
        if 'bias' in name:
            print("Biases:", param.data)

## These are the two input/output tensor pairs for Phase 1 of training
phase1_mappings = [
#1,4,7 -> 1,4,7
(torch.tensor([1.,0.,0.,1.,0.,0.,1.,0.]),torch.tensor([1.,0.,0.,1.,0.,0.,1.,0.])),
#2,5,8 -> 2,5,7
(torch.tensor([0.,1.,0.,0.,1.,0.,0.,1.]),torch.tensor([0.,1.,0.,0.,1.,0.,1.,0.]))
]

## These are the raw input triples for Phase 2 of training. They need to be
## converted to tensors before being input to the network
phase2_inputs = [
(1,4,7),
(1,4,8),
(1,5,7),
(1,5,8),
(1,6,7),
(1,6,8),
(2,4,7),
(2,4,8),
(2,5,7),
(2,5,8),
(2,6,7),
(2,6,8),
(3,4,7),
(3,4,8),
(3,5,7),
(3,5,8),
(3,6,7),
(3,6,8)
]

## This is the function that takes the raw input triples and converts them
## to input/output pairs in the form of torch.tensor objects
def rule_of_78_mappings(inputs):
    all_forms = []
    for a,b,c in inputs:
        if a == 1 and b == 4 and c == 7:
            c2 = 7
        else:
            if c == 7: c2 = 8
            if c == 8: c2 = 7
        inp = [0.]*8
        for i in (a,b,c): inp[i-1] = 1.
        out = [0.]*8
        for i in (a,b,c2): out[i-1] = 1.
        all_forms.append((torch.tensor(inp),torch.tensor(out)))
    return all_forms


phase2_mappings = rule_of_78_mappings(phase2_inputs)

## Here we run the training for each quadrant of Table 4 of R&M

print('TRAINING ON TWO EXAMPLES, 20 EPOCHS')
## 3a) Train the model on just the Phase 1 training data, for 20 epochs
run_training(phase1_mappings,20)


print('TRAINING ON ALL EXAMPLES, 10 MORE EPOCHS')
## 3b) Train the model on the Phase 2 training data, for 10 epochs
run_training(phase2_mappings,10)

print('TRAINING ON ALL EXAMPLES, 30 MORE EPOCHS')
## 3c) Train the model on the Phase 2 training data, for 30 epochs
run_training(phase2_mappings,30)

print('TRAINING ON ALL EXAMPLES, 460 MORE EPOCHS')
## 3d) Train the model on the Phase 2 training data, for 460 epochs
run_training(phase2_mappings,460)


## 4) Print out answers to the following:
## a. Is your model successfully learning during the training? How can you tell?
## b. Do the patterns in your model's learned weights resemble the patterns in
## Table 4 of R&M? Describe the patterns that you observe, and say a bit about why
## the model learns those particular weights.
answer_a= """

Answer_4a: 
Yes, it is successfully learning because the error rate is decreasing
from iteration to iteration (as shown by printing the current loss rate at each interval)
and heading towards 0. By the last
iteration the error rate is 0.0003945621137972921 which is a strong performance, and indicates the model
is training towards generating the data, as a loss rate that low would not be randomly generated.
While the model varies error rate between runs due to noise and a small batch size
the overall trend is a considerable decrease approaching zero.


"""


answer_b= """

Answer_4b: 
Yes, however by different degrees of magnitude, although the overall resemblance to the Table 4 is 
made by a constant factor. The sign and general size
of weights in the final table are similar to the ones for table 4, D) though
with about 1/6-1/7 of the size of 4. However this relationship holds for all the weights
in the table, suggesting a strong correlation in terms of the distribution of the table to other features within it.
A reason why these weights are learned in
the same shape is because the values were inputted in the same linear
fashion (meaning features lined up in terms of how the tables were trained)
in both the model and the paper. 
The reason for the difference in magnitude is pytorch precision optimizations which favor smaller more precise values
compared to the larger whole numbers in the paper.


"""


print(answer_a)
print(answer_b)
