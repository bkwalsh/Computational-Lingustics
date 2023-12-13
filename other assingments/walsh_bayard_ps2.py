import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter
import nltk
from nltk.util import ngrams
nltk.download("punkt")

##############################################
## N-gram LM SECTION
##############################################

## 1) Create a variable called 'corpus' and assign as its value a STRING consisting
## of a mini-corpus with the following bigram probabilities  (based on maximum
## likelihood estimation):
## P(inquisitive | was) = .2
## P(treasure | hidden ) = 1.0
## P(treasure | buried ) = .5
## P(only | the ) = .1
## P(immense | an) = .25
## Your corpus should be a series of complete sentences. Everyone should submit
# a unique corpus, so be as creative as you like with it.
corpus = '''
I was inquisitive.

I was lost.

I was found.

I was beyond the fence.

I was watching the man.

We found the hidden treasure.

We found the buried treasure.

We found the buried body.

There is an immense weight on my chest.

There is an AI for that.

There is an immesurable weight of guilt.

There is an error in xml document.

The rat.

The cat.

The fat dog.

The silly king.

The only one.

'''


## 2) Write a function 'compute_bigram_probabilities' that takes as input a text
## in the form of a string (e.g., your variable 'corpus' above), computes all of
## the bigram probabilities from that corpus using maximum likelihood estimation
## (relative frequency), and returns a dictionary-like object allowing you to
## look up the probability for any bigram
## A couple of things that may come in handy: a) the split() method that you learn
## about in the "Python Strings" tutorial above, which splits a string into a list,
## b) the collections.Counter() container -- this is like a dictionary, but is a
## bit more convenient for counting things (remember to 'import collections' if
## you want to use this) c) the ngrams function from the library nltk, which allows
## you to extract ngrams from a text (also remember to 'import nltk' if you want
## to use this)
def compute_bigram_probabilities(input_string):
    lst = input_string.lower().replace('.', ' ').split() # using .replace . with space for bigram formatting
    counterwords = Counter(lst)
    bigrams = list(ngrams(lst, 2))
    counterbigrams = Counter(bigrams)
    specific_queries = {}

    for item in bigrams:
        specific_queries[item] = counterbigrams[item]/counterwords[item[0]]  # get first word in bigram tuple
    
    return specific_queries


 
## 3) Run 'compute_bigram_probabilities()' with your 'corpus' variable as input,
## get the stored probabilities as returned output from the function, and use
## this output to access and print the five bigram probabilities above to confirm
## that your function produces the correct values
corpus_bigram = compute_bigram_probabilities(corpus)

# using bigrams as keys for corpus dictionary 
print("\n\n")
print("P(inquisitive | was) = " + str(corpus_bigram[('was', 'inquisitive')]))
print("P(treasure | hidden) = " + str(corpus_bigram[('hidden', 'treasure')]))
print("P(treasure | buried) = " + str(corpus_bigram[('buried','treasure')]))
print("P(only | the) = " + str(corpus_bigram[('the','only')]))
print("P(immense | an) = " + str(corpus_bigram[('an','immense')]))




##############################################
## NNLM SECTION
##############################################

# Here we define a class approximating the neural network language model (NNLM)
# introduced in Bengio et al. 2003
class NN_LM(nn.Module):

    def __init__(self,vocab_size,emb_size,hid_size):
        super(NN_LM, self).__init__()
        self.emb_size = emb_size
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.H_lyr = nn.Linear(3*emb_size, hid_size)
        self.U_lyr = nn.Linear(hid_size,vocab_size)

    def forward(self,inputs):
        x = self.emb(inputs).view((-1,3*self.emb_size))
        hid = torch.tanh(self.H_lyr(x))
        out = F.log_softmax(self.U_lyr(hid),dim=1)
        return out

# This function processes training data, establishing number IDs for each vocabulary word,
# converting word sequence into ID sequence (input_as_ids), and providing dict
# to map from word to its ID (word2id), and list to map from ID back to word (id2word)
def process_training_data(corpus_text):
        """Tokenizes a text file."""
        # Create the model's vocabulary and map to unique indices
        word2id = {}
        id2word = []

        for word in corpus_text.split():
            if word not in word2id:
                id2word.append(word)
                word2id[word] = len(id2word) - 1

        # Convert string of text into string of IDs in a tensor for input to model
        input_as_ids = []
        for word in corpus_text.split():
            input_as_ids.append(word2id[word])
        # final_ids = torch.LongTensor(input_as_ids)

        return input_as_ids,word2id,id2word


# This function runs the training of the model.
def run_training(train_data,id2word):

    # Define hidden layer size, embedding size, and number of training epochs
    hidden_size = 15
    emb_size = 5
    num_training_epochs = 50

    ## Initialize NNLM
    nnlm_model = NN_LM(len(id2word),emb_size,hidden_size)
    ## Define the optimizer as Adam
    nnlm_optimizer = optim.Adam(nnlm_model.parameters(), lr=.001)
    ## Define the loss function as negative log likelihood loss
    criterion = nn.NLLLoss()

    # Run training for specified number of epochs
    print('\nTraining NNLM on training corpus ...\n')
    for epoch in range(num_training_epochs):
        # Move through data one word (ID) at a time, extracting a window of three
        # context words, and a target fourth word for the model to predict
        for i in range(len(train_data) - 3):
            input_context = torch.LongTensor(train_data[i:i+3])
            target_word = torch.LongTensor([train_data[i+3]])

            # Run model on input, get loss, update weights
            nnlm_optimizer.zero_grad()
            output = nnlm_model(input_context)
            loss = criterion(output, target_word)
            loss.backward()
            nnlm_optimizer.step()

    return nnlm_model

# Here is a tiny toy training corpus for simulation purposes
train_corpus = """
okay, he thought ; i'm off to work . he reached for the doorknob that opened the way out into the unlit hall ,
then shrank back as he glimpsed the vacuity of the rest of the building . it lay in wait for him, out here ,
the force which he had felt busily penetrating his specific apartment . god , he thought , and reshut the door .
he was not ready for the trip up those clanging stairs to the empty roof where he had no animal . the echo of
himself ascending : the echo of nothing . time to grasp the handles , he said to himself , and crossed the living
room to the black empathy box .
"""

# Here we process the training corpus into a sequence of number IDs, and get the
# mapping from word to ID and ID to word. See code above for details.
train_data,word2id,id2word = process_training_data(train_corpus)

# Here we run the code to initialize and train a model on the processed training
# data, and output the trained model
trained_model = run_training(train_data,id2word)


test_corpus = [
'reached for the',
'the empty roof',
'force which he',
'the black empathy',
'rest of the',
'thought , and',
'work vacuity specific',
'he off to'
]

## 4) Above is a list 'test_corpus' of three-word contexts to feed as input to
## the trained NNLM 'trained_model'. Write code here that identifies the top word
## prediction of the trained model for each of the test input contexts.
## This will mean, for each test context, 1) convert the context to the correct
## input format for the model, 2) run the trained model on the input context
## to get the output, and 3) identify, based on the output, which next word the
## model is assigning the highest probability to.
## Please then have your code print each input context along with the word
## that the model predicts for that context.
## Note that the model's output represents log probabilities for each word
## in the vocabulary. For converting between words and their indices (number IDs),
## you can use the word2id dictionary and the id2word list generated above. Check
## the relevant code to see where exactly those objects come from. You can also
## look at the training code above to see the input format used for the model.
## You can use torch.LongTensor to convert to tensor when needed.
## Another function that may come in handy is torch.argsort, or numpy.argsort if
## you first convert the output torch tensors to numpy arrays.
## Remember that you can print things if you aren't sure what they are/what
## format they have.

def get_next_word(str_lsts):
    outputlst=[]

    for i in range(len(str_lsts)):
        tokenized_input = test_corpus[i].split()
        lst = []

        # get list of wordids
        for i in tokenized_input:
            lst.append(word2id[i])

        input_tensor = torch.tensor(lst)
        input_tensor = input_tensor.unsqueeze(0)

        # train on ids
        with torch.no_grad():
            output = trained_model(input_tensor)


        probabilities = torch.exp(output)
        max_index = torch.argmax(probabilities)

        print("Input of Text Words:", tokenized_input)
        print("Indexes of Input List:", lst)
        print("Index of Maximum Generated Value:", max_index.item())
        print("Maximum Generated Value:", id2word[max_index.item()])
        print("\n\n")

        outputlst.append(id2word[max_index.item()])
    print("Output list:")
    print(outputlst)
    print("\n\n")

get_next_word(test_corpus)    



## 5) Most of the text contexts we used as input are exact sequences from the training
## data. How does the model do at predicting the next word in those sequences? What
## about for the couple of sequences that are not in the training text? What do
## the model's predictions look like, and why do you think this is? Have your code
## print your responses to these questions.


answer_5= '''

In the case of exact sequences, the model predicts the next word correctly every time.
This is interesting because printing the input indexes list shows that the index
corresponds to the first time the word occurs in the text, (such as 'the' being 11), even if it shows
up multiple times. This indicates the model is more sophisticated than simply indexing
the next value from a numbered list. Furthermore, in cases where the 3 words are in sequence,
the model has 100 percent accuracy, showing it has memorized the input text completely.

In the two cases of non-exact sequences, there is a lot more error in the prediction.
While the model is not deterministic (multiple runs have generated 'work', 'had', and 'time' for the
first non-exact test), the first non-exact case seems to be selecting somewhat random outputs, 
either picking a word directly after one of the words in the string, or another random word.
This is likely because of the disconnected nature of the three words in this input string.

In the second non-exact case, during my testing, the model selected either 'work' which 
corresponds to the next word in the 'off',' to' sequence, or 'the' which is the most
the popular word following 'to'. Therefore this case is also non-deterministic but seems
to be either following the partially exact end of the input string or generating based off
the last word.

Note that this analysis is through observations and would be strengthened through
multiple runs and statistical analysis.

'''

print(answer_5)