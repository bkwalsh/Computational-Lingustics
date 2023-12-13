import random
import matplotlib.pyplot as plt


# This is a function which takes the desired vocabulary size as an input
# argument and returns a list of word difficulties drawn from a normal
# distribution with mean 400 and standard deviation 70

def generate_word_difficulties(vocab_size):

    # Here we create an empty list 'word_difficulties' to store the word
    # difficulties in
    word_difficulties = []

    # Here we loop over the full desired vocab size
    for word in range(vocab_size):
        # Here we draw one word difficulty from a normal distribution
        word_difficulty = random.normalvariate(400,70)
        # Here we append that word difficulty to our list of word difficulties
        word_difficulties.append(word_difficulty)

    # Here we plot a histogram of the word difficulties
    plt.hist(word_difficulties)
    plt.xlabel('Timesteps needed for learning')
    plt.ylabel('Number of words')
    plt.savefig('difficulty_distribution.png')
    plt.clf()

    return word_difficulties

# This is the function that runs the McMurray (2007) simulation

def mcmurray(all_word_difficulties):


    # **
    # 1) Create a VARIABLE called 'total_steps_to_run',
    # and assign it a value of 600
    # **
    total_steps_to_run=600

    # **
    # 2) Create an empty LIST called 'learning_curve'
    # (you will use this to store the number of words that
    # are known at each time step)
    # **
    learning_curve=[]


    # Here we start a loop over the total number of time steps
    for time_step_number in range(total_steps_to_run):

        # Here we set a variable 'current_num_words' and set it to 0.
        # This will track how many words are known at this time step.
        current_num_words = 0

        # Here we loop through the list of word difficulties
        for word_difficulty in all_word_difficulties:

            # **
            # 3) Create an IF STATEMENT which checks whether 'time_step_number'
            # is greater than or equal to 'word_difficulty',
            # and which, if that condition is true, increases
            # 'current_num_words' by 1, using the ASSIGNMENT OPERATOR +=
            # **
            if time_step_number>=word_difficulty:
                current_num_words+=1

        # **
        # 4) Append the latest value of 'current_num_words' to
        # the list 'learning_curve', to record
        # the number of words known at this time step
        # **
        learning_curve.append(current_num_words)


    # Here we plot the number of words known over time
    plt.plot(learning_curve)


    # **
    # 5) Create a variable 'plot_filename' and set the
    # value to a string '<yourlastname_yourfirstname>_curve.png'.
    # This will allow the code to save a plot identified by your name.
    # **
    plot_filename='walsh_bayard_curve.png'


    # Here we save the plot to a file with the specified filename
    plt.savefig(plot_filename)

# Here we run the function 'generate_word_difficulties'
# with input argument 10000, which allows us to create a
# list of 10000 word difficulties, drawn from a normal distribution

all_word_difficulties = generate_word_difficulties(10000)

# Here we run the 'mcmurray' function, which runs the simulation using
# the list of word difficulties

mcmurray(all_word_difficulties)


# **
# 6) You will see that this code creates a file 'difficulty_distribution.png'.
# Please answer the following questions about that plot, and add a print
# statement that will print those answers when we run the code:
# a) Based on the code, briefly explain what the plot in
# 'difficulty_distribution.png' represents, and how it relates to the
# simulation as a whole.
# b) If you run the simulation multiple times, you'll see that
# 'difficulty_distribution.png' changes between runs. Why is that?
# **

a = ("\nA): This plot is a histogram of word difficulties, so it shows the broad "
     "bounds (100) sized buckets where words are distributed. Following the "
     "mcmurray paper, we are assuming learning words can be done in parrallel "
     "so therefore for each instance we iterate through each word that could "
     "be learned at the time stamp and then append it to the list. "
     "Therefore our graph follows the growth gaussian in the paper "
     "however it records overall words learned so it "
     "increases slowly, then quickly, then slowly again "
     "because it assumes word acquistion increase after 300~ time stamp "
     "and then decreases again after 500~ time stamp and this "
     "distribution is based on the mcmurray model \n \n \n")


b = ("B): The reason why you'll see multiple different graphs when you run the "
     "function multiple times is because of the line "
     "word_difficulty = random.normalvariate(400,70). Note that for each call "
      "in the loop this creates "
     "a random value roughly around 400 with a standard deviation, meaning (95%) of all "
     "values fall in 260-540. As we run this 10000 times, it will converge towards "
     "the outline of a similar looking gaussian (this is because of the central limit theorem), "
      "but will be slightly differnt based on "
     "code induced randomness. \n")

print(a+b)