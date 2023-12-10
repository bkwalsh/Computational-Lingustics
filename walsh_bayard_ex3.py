

# This is a LIST of all hypotheses to be considered
hypothesis_space = ['dalmatian','terrier','poodle','dog','animal']

# This is a DICTIONARY mapping from hypotheses to their "heights" -- these
# heights are estimated directly from the node heights in Xu & Tenenbaum Fig 7
height = {
'dalmatian': .05,'terrier': .05,'poodle': .05, 'dog': .25, 'animal': .5
}

# This is a DICTIONARY mapping from hypotheses to their "parent heights" -- the
# heights of their nearest parent node. These are estimated directly from
# the node heights in Xu & Tenenbaum Fig 7
parent_height = {
'dalmatian': .2,'terrier': .2, 'poodle': .2,'dog': .38, 'animal': .9
}

# This is a DICTIONARY mapping from each observed item type to a list of
# the categories that that item belongs to
category_membership = {
'dalmatian': ['dalmatian','dog','animal'],
'terrier': ['terrier','dog','animal'],
'poodle': ['poodle','dog','animal']
}


# This function takes as input a LIST of observations and a STRING naming the
# current hypothesis, and uses Xu & Tenenbaum Equation 6 to compute the
# likelihood -- P(hypothesis | observations)
def compute_likelihood(observations,hypothesis):
    # check whether the hypothesis that we received as input is a possible
    # candidate given the input
    viable_hypothesis = True
    for obs in observations:
        if hypothesis not in category_membership[obs]:
            viable_hypothesis = False

    # if it's a viable hypothesis, proceed with the formula in Equation 6
    if viable_hypothesis:
        # Assign a value for epsilon
        epsilon = .05

        # Get the height of the current hypothesis by accessing the DICTIONARY
        # 'height' above
        hyp_height = height[hypothesis]

        # 1a) Create a variable 'n' and assign it a value equivalent
        # to the LENGTH of the input LIST 'observations'
        n = len ( observations )


        # 1b) Create a variable 'likelihood' and assign it a value based on the
        # right side of Xu & Tenenbaum Equation 6, using the variables
        # 'hyp_height', 'epsilon', and 'n' created above. Use parentheses to
        # ensure proper order of operations.
        likelihood = (1 / ( hyp_height + epsilon ) ) ** n


    # if not a viable hypothesis, likelihood = 0
    else:
        likelihood = 0.0

    # return the variable 'likelihood' as function output
    return likelihood


# This function takes as input the STRING naming the current hypothesis,
# and computes the prior probability for that hypothesis -- P(hypothesis)
def compute_prior(hypothesis):

    # 2) Create a variable 'prior' and assign it a value based on the
    # right side of Xu & Tenenbaum Equation 7. You will need to access the
    # DICTIONARIES 'parent_height' and 'height', and use them to find the
    # relevant values for the current input hypothesis
    prior = parent_height[ hypothesis ] - height[ hypothesis ]


    # return the variable 'prior' as function output
    return prior



# This function takes as input the list of observations, and computes a
# posterior probability for each hypothesis in the LIST 'hypothesis_space'
def simulation(observations):
    product_dict = {}

    for hypothesis in hypothesis_space:
        # compute P(X | h) -- likelihood of data given this hypothesis
        likelihood = compute_likelihood(observations,hypothesis)
        # compute P(h) -- prior of that hypothesis
        prior = compute_prior(hypothesis)
        # compute product of prior and likelihood
        product = likelihood * prior

        # save product in DICTIONARY for later
        product_dict[hypothesis] = product

    # sum over the products for the relevant hypotheses,
    # to get the normalizing constant (denominator P(X))
    px = sum([product_dict[hyp] for hyp in product_dict])

    # divide product from each hypothesis by denominator px to get
    # final posterior probability for each hypothesis given this data
    for hypothesis in hypothesis_space:
        posterior = product_dict[hypothesis]/px
        # print the posterior
        print('P("blick" = %s | X): %s'%(hypothesis,posterior))


################# SIMULATIONS RUN BELOW ##########################


# These are the three LISTS of observations based on which you want to compute
# posterior probabilities for all potential hypotheses
observations_1 = ['dalmatian']
observations_2 = ['dalmatian','terrier','poodle']
observations_3 = ['dalmatian','dalmatian','dalmatian']


print('\nPOSTERIOR PROBABILITIES AFTER HEARING "blick" WITH X = %s:'%observations_1)
# Here we run the FUNCTION 'simulation' with the input being the LIST
# 'observations_1'. As a reference, your probability of 'dog' should
# end up around .16.
simulation(observations_1)

print('\nPOSTERIOR PROBABILITIES AFTER HEARING "blick" WITH X = %s:'%observations_2)
# 3a) Run the FUNCTION 'simulation' with the LIST 'observations_2' as the
# input argument. Your probability of 'dog' should end up around .66.
simulation(observations_2)

print('\nPOSTERIOR PROBABILITIES AFTER HEARING "blick" WITH X = %s:'%observations_3)
# 3b) Run the FUNCTION 'simulation' with the LIST 'observations_3' as the
# input argument. Your probability of 'dog' should end up around .03.
simulation(observations_3)

# 4a) Create a new LIST of observations named 'observations_4' containing any
# number (zero or more) of instances of 'dalmatian', 'terrier', and/or 'poodle'.
observations_4 = ['poodle','terrier','terrier','terrier','terrier','poodle','poodle','dalmatian']

print('\nPOSTERIOR PROBABILITIES AFTER HEARING "blick" WITH X = %s:'%observations_4)
# 4b) Run the FUNCTION 'simulation' with the LIST 'observations_4' as the
# input argument.
simulation(observations_4)


# 4c) Observe how all of the posterior probabilities look with your new set of
# observations, and briefly explain why the probabilities come out that way. Add
# a print statement such that your answer to this question will print when the code runs.
answer = ("\n The probabilities are only non zero for dog or animal in my example case, " 
          "because there are mixed "
          "instances of dogs. In other words, with a sample size larger than "
          "a single specific dog, we know that a blick cannot "
          "mean a Dalmatian or poodle because both are defined as Blick. This finding follows the "
          "assertion from the Xu/Tenenbaum hypothesis of generalizing the "
          "meaning of a word when given more different labeling instances. However, we see that "
          "the probability heavily falls on dog- (~97%) instead of the broader category of animal "
          "this is because in the larger sample size (relative to this model) of 8 instances, all instances "
          "are specific dogs- therefore it would inform the uninitiated "
          "language learner that a blick must define not a specific dog, but dog type generally. "
          "While it is still possible to be animal type, because all cases are dogs, we assume that the "
          "broader generalization is not necessary, however with some animal types, we would assue "
          "Blick means animal. \n")

print (answer)