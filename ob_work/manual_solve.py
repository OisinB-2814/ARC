#!/usr/bin/python

import os, sys
import json
import numpy as np
import re

# Name: Oisin Brannock
# Student ID: 20235671
# Class: CT5148
# GitHub repo: https://github.com/OisinB-2814/ARC


### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.

def solve_6f8cd79b(x: np.array):
    '''
    
    The goal of this function is to take in a numpy array filled with a single value (or colour) of size n x m.
    The function makes a 'wall' around the array and returns a new array surrounded by 8 (sky blue) on the top, bottom, right and left walls of the matrix
    Parameters:
        x (np.array): A 2D numpy array
    
    >>> solve_6f8cd79b(train[0])
    array([[8, 8, 8],
           [8, 0, 8],
           [8, 8, 8]])
       
    >>> solve_6f8cd79b(np.zeros([10, 10]))
    array([[8., 8., 8., 8., 8., 8., 8., 8., 8., 8.],
           [8., 0., 0., 0., 0., 0., 0., 0., 0., 8.],
           [8., 0., 0., 0., 0., 0., 0., 0., 0., 8.],
           [8., 0., 0., 0., 0., 0., 0., 0., 0., 8.],
           [8., 0., 0., 0., 0., 0., 0., 0., 0., 8.],
           [8., 0., 0., 0., 0., 0., 0., 0., 0., 8.],
           [8., 0., 0., 0., 0., 0., 0., 0., 0., 8.],
           [8., 0., 0., 0., 0., 0., 0., 0., 0., 8.],
           [8., 0., 0., 0., 0., 0., 0., 0., 0., 8.],
           [8., 8., 8., 8., 8., 8., 8., 8., 8., 8.]])
    
    All training and test grids are solved correctly 
    
    '''
    # Make a copy of x so we don't alter the original data
    y = x.copy()
    # Top row
    y[:, 0] = 8
    # Left most column
    y[0, :] = 8 
    # Bottom row
    y[-1, :] = 8 
    # Right most column
    y[:, -1] = 8
    # Pull back new matrix
    return y

def solve_0d3d703e(x: np.array):
    '''

    The goal of this task is to change a column from one colour (or number) to another based on a mapping grid. 
    This is accomplished here by first initialising a dictionary with all of our mappings. 
    The np.vectorize function is then used to create a new array based on our input array by using 
    a lambda function that utilises the mapping dict to change the input. This is returned to us then as a new numpy 2d array.
    
    Parameters:
        x (np.array): A 2D numpy array
    
    >>> solve_0d3d703e(np.array([[9, 7, 2], [9, 7, 2], [9, 7, 2]]))
    array([[8, 7, 6],
           [8, 7, 6],
           [8, 7, 6]])
           
    >>> solve_0d3d703e(np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]))
    array([[5, 6, 4],
           [5, 6, 4],
           [5, 6, 4]])

    All training and test grids are solved correctly 

    '''
    # Create mapping dictionary that contains all colour/numerical transformations.
    mappings = {0: 0, 1: 5, 2: 6, 3: 4, 4: 3, 5: 1, 6: 2, 7: 7, 8: 9, 9: 8}
    # If statement within lambda ensures that function doesn't fall if we get colours beyond 9 e.g 10
    # https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html
    new_x = np.vectorize(lambda y: mappings[y] if y in mappings else y)(x)
    return new_x

def solve_e9afcf9a(x: np.array):
    '''

    The goal of this function is to take an input in the form of a 2 x 2 numpy array and output a 2 x 2 numpy array with every 
    second value swapped between the first and second row.
    
    It takes in ther shape of the input array columns, appends the value of every second index to a new list for each row and
    appends this to a new list of lists where the first list is new_array and the second is simply the reverse! Finally this list of 
    lists is converted back into a numpy array and returned.
    
    Parameters:
        x (np.array): A 2D numpy array of shape 2 x 2
    
    >>> solve_e9afcf9a(np.array([[9, 9, 9, 9],[5, 5, 5, 5]]))
    array([[9, 5, 9, 5],
           [5, 9, 5, 9]])
    
    >>> solve_e9afcf9a(np.array([[7, 7, 7, 7, 7, 7, 7, 7],[2, 2, 2, 2, 2, 2, 2, 2]]))
    array([[7, 2, 7, 2, 7, 2, 7, 2],
           [2, 7, 2, 7, 2, 7, 2, 7]])
           
    All training and test grids are solved correctly 

    '''
    # We want to get the shape of the initial array passed in and take only the second arg for our range loop later
    _, b = x.shape
    # Initialise 2 empty lists to input resultswd
    new_array = []
    final_array = []
    # For n in range of the length of the number of columns in the array:
    for n in range(b):
        # Append the even indexed values to new_array (0, 2, 4 etc.)
        new_array.append(x[n % 2][n])
    # Using .insert() function to insert the new array list as 0th arg
    final_array.insert(0, new_array)
    # Using .insert() again to input the reverse of the list above
    final_array.insert(1, new_array[::-1])
    # Converting to a numpy array to align with the input value
    final_array = np.array(final_array)
    return final_array


'''

Reflection:

I was able to complete all tasks simply just using numpy and basic python properties like slicing, dictionaries, lambda functions and for loops
The tasks all linked through numpy and the manipulation of arrays. It got me thinking about whether all of these tasks could be solved with
a single numpy function with different parameters and statements. I came across a paper online (https://arxiv.org/abs/2011.09860) in fact that claims a model accuracy of 78% on the dataset using one algorithm!
The tasks themselves to me took a lot of careful thought to code even thoughthey could be easily solved by my brain. 
I can see how the ultimate goal of ARC aligns with the assignment in getting us to think about how we code and finding efficient ways of doing things in the simplest way possible! 

'''

def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})" 
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals(): 
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)
    
def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""
    
    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)

        
def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    # if yhat has the right shape, then (y == yhat) is a bool array
    # and we test whether it is True everywhere. if yhat has the wrong
    # shape, then y == yhat is just a single bool.
    print(np.all(y == yhat))

if __name__ == "__main__": main()