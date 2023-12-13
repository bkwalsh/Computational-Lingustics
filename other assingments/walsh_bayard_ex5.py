


# Here is a DICTIONARY representing the trained decision tree. This is what the
# code below uses to determine what prediction the model makes for each item
# in the test set
trained_decision_tree = {
'ftr': 'isSystems',
'left': 'LIKE',
'right': {
          'ftr':'takenOtherSys',
          'left': { 'ftr': 'morning','left': 'LIKE', 'right': 'NAH'},
          'right': { 'ftr': 'likedOtherSys', 'left':'NAH', 'right': 'LIKE'}

         }
}

# Here is the LIST of test items. Each item in the list is a TUPLE consisting of
# a) the true label for the item (a string 'LIKE' or 'NAH'), and b) the feature
# values for the item (a DICTIONARY mapping feature names to yes/no values), in
# the following order: (true_label, item_features).
# 1) Add to 'test_set' one more item TUPLE, on which the
# 'trained_decision_tree' will make the WRONG prediction. Your item tuple should
# take the same form as the existing test items in the list, with either 'LIKE'
# or 'NAH' as the label, and a complete dictionary of features with values of
# 'YES' or 'NO'. At the moment the average loss is .2 -- your new item
# should bring the average loss to .33.
test_set = [
('NAH',{'isSystems': 'YES','takenOtherSys': 'NO','morning':'YES','likedOtherSys':'NO'}),
('NAH',{'isSystems': 'NO','takenOtherSys': 'NO','morning':'YES','likedOtherSys':'NO'}),
('LIKE',{'isSystems': 'YES','takenOtherSys': 'YES','morning':'YES','likedOtherSys':'YES'}),
('NAH',{'isSystems': 'YES','takenOtherSys': 'NO','morning':'YES','likedOtherSys':'YES'}),
('LIKE',{'isSystems': 'NO','takenOtherSys': 'NO','morning':'YES','likedOtherSys':'NO'}),
('NAH',{'isSystems': 'NO','takenOtherSys': 'NO','morning':'NO','likedOtherSys':'NO'}) # this is the entry I added 
]

# This function takes a trained tree and a test item and produces a predicted
# label for the item based on the tree. This is from the algorithm in Ch. 1 of
# A Course in Machine Learning (Daume III)
def DT_test(tree, test_point):
    if type(tree) == str:
        return tree
    elif type(tree) == dict:
        f = tree['ftr']
        if test_point[f] == 'NO':
            print('%s -- NO'%f)
            return DT_test(tree['left'],test_point)
        else:
            print('%s -- YES'%f)
            return DT_test(tree['right'],test_point)

# This function computes the average loss for a list of
# model-prediction/true-label pairings
def compute_loss(pred_label_pairs):
    overall_loss = 0.
    for prediction,label in pred_label_pairs:
        if prediction == label:
            overall_loss += 0
        else:
            overall_loss += 1
    avg_loss = overall_loss/len(pred_label_pairs)
    return avg_loss

# This part of the code loops through each item in the test set, gets a
# prediction from the trained decision tree, and stores the prediction and the
# true label in a tuple for computing the loss later
prediction_label_pairs = []
i = 0
for item_label,item_features in test_set:
    print('ITEM: %s'%i)
    prediction = DT_test(trained_decision_tree,item_features)
    print('Prediction: %s\n'%prediction)
    pred_label_tuple = (prediction,item_label)
    prediction_label_pairs.append(pred_label_tuple)
    i += 1

loss = compute_loss(prediction_label_pairs)
print('AVG LOSS: %s'%loss)

# 2) Answer the following question and write a print command so that the answer
# will print when we run the code.
# Look at the contents of the 'compute_loss' function, and compare it to the
# different types of loss functions described in Section 1.4 of 'A Course in
# Machine Learning' Ch 1. Which of those different loss functions is implemented
# in this code?
answer = ("\n \nThe loss function that is implemented is the binary classifier "
          "\nNote that inside the loop in each iteration "
          "compute_loss checks if the prediction matches the label, \nand if it "
          "does then it adds one to the sum, and if not it adds zero (or nothing) to the sum. \n"
          "While the code is not a mathematical function (as is in the textbook definition of binary classification),\nwe can "
          "read the if statement as a piecewise function, which branches according to "
          "y hat= y (prediction == label in this case) \nor otherwise, and once we"
          "make this comparison the loss function behavior matches the binary classification equation. \n"
          "Therefore the loss function used in the script is binary classification "
          "loss defined in section 1.4. \n"
          "Note also that the model makes an average sum of binary classification across each prediction\n"
          "as it divides the overall sum (for each binary classification instance in the dictionary)\n"
          "by the length of the dictionary at the end of the function.\nTherefore the overall"
          "performance returned models the average loss of the expected value in the model"
          "\n\n")
print(answer)