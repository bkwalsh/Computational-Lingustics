
import math

# A function taking as input two vectors (each in the form of a LIST)
# and returning the cosine similarity of those two vectors.
def cosine_similarity(A,B):
    # 1) YOUR TASK: implement the cosine similarity computation here. The
    # contents of this function should assume that A and B are vectors in LIST
    # form, and should calculate the cosine for those vectors. (You can assume that
    # A and B vectors will have the same number of dimensions as each other, but
    # you should account for this number of dimensions varying between runs of
    # the function.) The function should assign this computed cosine value to
    # the VARIABLE 'cosine_value'. I've imported the 'math' module above so that
    # you can use the math.sqrt() function for computing square roots.
    # Note that you should NOT import and use pre-implemented
    # cosine similarity functions.


    # This line returns 'cosine_value' as the function output
    return cosine_value

# FYI this line is a way to ensure that the following code runs only if you are
# using this source file as the main program (rather than importing functions
# from it for use by another program).
if __name__ == "__main__":

    # Below are two tests to allow you to check your function's output

    # Two 3-dimensional vectors v1 and v2
    v1 = [1.,0.,1.]
    v2 = [0.,1.,0.]

    # Pass v1 and v2 as inputs to your function, and assign the function output
    # to the VARIABLE 'cos_value', which will be printed.
    # For these vectors, your function's output value should be 0.0
    cos_value = cosine_similarity(v1,v2)
    print('Vector 1: %s'%v1)
    print('Vector 2: %s'%v2)
    print('COSINE: %s\n'%cos_value)

    # New vectors, now with 6 dimensions
    # For these vectors, your function's output value should be about .57
    v3 = [.5,-.12,.23,.31,-.4,.06]
    v4 = [.22,.48,.11,.29,-.34,.18]

    cos_value = cosine_similarity(v3,v4)
    print('Vector 3: %s'%v3)
    print('Vector 4: %s'%v4)
    print('COSINE: %s\n'%cos_value)

    # 2) Create two new vectors called v5 and v6, such that the cosine similarity
    # between v5 and v6 is greater than .91 and less than .98. Uncomment the four
    # lines of code below to run your 'cosine_similarity' function on these vectors
    # so that you can check whether their cosine value falls in the right range.
    # The vectors can have any number of dimensions (as long as they have the
    # same number of dimensions as each other). As above, the vectors
    # should be in the form of LISTS.


    # cos_value = cosine_similarity(v5,v6)
    # print('Vector 5: %s'%v5)
    # print('Vector 6: %s'%v6)
    # print('COSINE: %s\n'%cos_value)
