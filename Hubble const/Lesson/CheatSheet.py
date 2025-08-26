#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 09:51:33 2021

This code is meant to serve as a sort of "Python Cheat Sheet"
for students on PHYS1201
    
@author: christian
"""
### IMPORTING PACKAGES
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from scipy import optimize

### GLOBALS
###
### Global variables are variables that can be used anywhere
### inside a code -- including in functions -- without explicitly
### passing them as input parameters. In general, they are bad
### news, and you should avoid using them. 
### 
### If you *must* have global variables, define them explicitly
### as such and stick them at the top of your .py file
global c_light 
c_light  = 2.99e10 #the speed of light in cm/s

### FUNCTIONS
###
### Always stick your functions at the top of your .py file.
### Also remember that if function B needs (i.e. calls) function A,
### then function A has to come first -- i.e. be above -- function B
### in the .py file

def sample_function1(input1, input2, input3, input4):
    """
    This silly function adds input1 and input2 and returns this as output1.
    It also subtracts input4 from input3 and returns this as output2.
    
    A function is a piece of code designed to carry out a particular task.
    Almost all functions take one or more *input parameters* (the things in
    parentheses after the function name) and use these to calculate one or more
    *output parameter* (the things after the return statement in the function).

    Every function should have a *docstring* at the top that explains what it does,
    specifies exactly what the input parameters are (including what *types* 
    -- float, int, list, array... -- the function expects them to be) and also
    reports what output parameters it returns.
    
    In *spyder*, as soon as you type three double quotes after a *def* statement,
    the editor will have a little pop-up "generate docstring" that creates a 
    skeleton docstring for your function.
    
    The docstring is available to the user simply by typing "help(function_name)"
    The idea is that the user should be able to *use* the function correctly just 
    by looking at the information in the docstring. They should not need to look 
    at the actual function itself.
    
    One thing to note is that functions do not automatically check or enforce the 
    types we want our inputs to be. I.e. we might want input1 to be a float, but
    a user could abuse our function by passing in an int, or an array, or a boolean.
    In some cases this will cause the code to fall over (e.g. if we try to add 
    a boolean to a float). In other cases the resulting operation may be legal,
    but give unexpected results. If we are worried about this, we should put in
    checks.
    

    Parameters
    ----------
    input1 : float
        Just a real number we want to add to input2
    input2 : float
        Just a real number we want to add to input1
    input3 : int
        Just an integer from which we want to subtract input4
    input4 : int
        Just an integer which we want to subtract from input3

    Returns
    -------
    output1 : float
        The result of adding input1 and input2.
    output2 : int
        The result of subtracting input4 from input3

    """
    
    output1 = input1 + input2
    output2 = input3 - input4
    
    return output1, output2

### We *call* a function by invoking its name and assigning its
### outputs to other variables

in_test1 = 3.1
in_test2 = 4.2
in_test3 = 1
in_test4 = 5

out_test1, out_test2 = sample_function1(in_test1, in_test2, in_test3, in_test4)    
print()
print('FUNCTIONS')
print('=========')
print('out_test1 = ', out_test1)
print('out_test2 = ', out_test2)

### Note that we need to have a 1-to-1 match between the input parameters
### we provide to the function in the call and the input parameters it
### expects, i.e. those listed in the def statement. 
### We also want to have a 1-to-1 match between the variables we
### assign the function to -- i.e. out_test1 and out_test2 here -- and the
### output parameters of the runction, i.e. those appearing after the return
### statement in the function.
###
### The names of the input and output variables should ideally be different
### inside and outside the function. For example, it would not be good practice
### to use the variable name input1 instead of in_test1 here. It's not that 
### this is not allowed by python, but it can more easily lead to confusion
### and make it harder to debug the code.
###
### There is a way to have optional parameters (that can be provided but 
### don't have to be and you'll see those in some function we use like 
### pyplot.plot -- but this is not something to worry about in our own functions
###
### If you call a function without assigning it to output parameters, and
### objects it returns via its return statements are just "lost". You pretty 
### much never want to do this, unless your function's purpose is just to plot
### the inputs, or something like that. 
### 
### If you assign a function that returns multiple things -- has several items
### after the return statement -- to a single variable name, that variable
### becomes a tuple and stores all of them.

out_test3 = sample_function1(in_test1, in_test2, in_test3, in_test4)
print()
print('The type of out_test3 is ', type(out_test3))
print('It contains ', out_test3)
print('The first element is ', out_test3[0])
print('The second element is ', out_test3[1])
   



### VARIABLES
###
### Variables are named containers for things. They can contain
### different types of data. For example, the statement
###     x = 3
### assigns the integer 3 to the variable named x
print()
print('VARIABLES')
print('=========')
x = 3
print('x = ', x)
print('The type of x is ', type(x))
x = 3.1
print()
print('x = ', x)
print('The type of x is ', type(x))
###
### One type of data to be aware of is "Booleans" (aka "Logicals")
### which are things that can only take on the value True or False.
### They are useful for making comparisons.
x = True
print()
print('x = ', x)
print('The type of x is ', type(x))
###
### We can *define* Booleans by making comparisons. For example
### A statement like "x1 > x2" asks "Is x1 greater than x2" and the 
### answer is then a Boolean (True or False).  Similarly, a statement like
### "x1 >= x2" asks "Is x1 greater than or equal to x2" 
x1 = 3
x2 = 1
x = (x1 > x2)
print()
print('x = ', x)
print('The type of x is ', type(x))
x = (x1 <= x2)
print('x = ', x)
print('The type of x is ', type(x))
###
### Note that if we want to check whether two things are equal
### to each other, we can't just use the "=" symbol, since that's
### already used for *assigning* things to variables. So instead, 
### we use the symbol "==". So, for example, the statement 
###     x1 == x2
### asks "Is x1 equal to x2", producing a Boolean (True or False)
x1 = 3
x2 = 3
x = (x1 == x2)
print()
print('x = ', x)
print('The type of x is ', type(x))
x2 = 4
x = (x1 == x2)
print('x = ', x)
print('The type of x is ', type(x))
###
### There is also a way of asking "Is this thing NOT equal to that thing?"
### The operator for that is "!="
x1 = 3
x2 = 5
x = (x1 != x2)
print()
print('x = ', x)
print('The type of x is ', type(x))
###
### We can also combine Booleans, via "and", "or" or "not"
### "a or b" evaluates as True if *either* a or b are True
### "a and b" only evaluates as True if *both* a and b are True
x1 = 3
x2 = 1
x3 = 1
x4 = 3
my_bool = (x1 > x2) and (x3 > x4)
print()
print('my_bool = ', my_bool)
my_bool = (x1 > x2) or (x3 > x4)
print()
print('my_bool = ', my_bool)
### If we combine more than two Booleans, it's helpful to use parenthesis
### to make sure we know what order they're calculated in
my_bool = ((x1 > x2) and (x3 > x4)) or ((x1 > x2) or (x3 > x4))
print()
print('my_bool = ', my_bool)
### There is also a way to "invert" a boolean (switch True to False, and
### vice versa). This is done via the "not" operator 
invert_my_bool = not(my_bool)
print()
print('invert_my_bool = ', invert_my_bool)



### LISTS
###
### Lists are just sequences of things. They do not all have to be
### the same type
my_list = [1, 2.0, True, 'blue']
print()
print('LISTS')
print('=====')
print('The type of my_list is ', type(my_list))
print(my_list)
### We can access individual elements in the list by using
### their indices in square brackets. Counting starts at ZERO! 
### So, for example, the first element of my_list is my_list[0]
first = my_list[0]
second = my_list[1]
third = my_list[2]
fourth = my_list[3]
print()
print()
print(first, type(first))
print(second, type(second))
print(third, type(third))
print(fourth, type(fourth))
### We can modify individual elements of a list
my_list[0] = 2
print()
print(my_list)
### The indices can, of course, themselves be variables (but they
### have to be integers
my_index = 3
print()
print(my_list[my_index])
### We can use slicing to refer to parts of lists by using the "slicing"
### notation -- the index [i:j] means "elements starting at index i, up to
### but NOT including j"
part_list = my_list[1:3]
print()
print(part_list)

### Lists can contain other lists
my_sublist = [10, 20, 30]
my_list = [1, 2, my_sublist, True]
print()
print(my_list)
### If we want to refer to the zeroth element of my_sublist, which is the 
### second element of my_list, we can write
print()
print(my_list[2][0])



### TUPLES
### Tuples are just like lists, except you can't modify individual elements
my_tuple = tuple(my_list)
print()
print('TUPLES')
print('======')
print(my_tuple)
print(type(my_tuple))
print(my_tuple[my_index])
### the following line -- if uncommented -- will give an error
#my_tuple[0] = 25



### ARRAYS
###
### Arrays are similar to lists, but they can only contain 
### items of the same type. In practice, we essentially always
### use them with numbers.
my_array = np.array([1.0, 2.0, 3.0, 4.0])
print()
print('ARRAYS')
print('======')
print(my_array)
print(type(my_array))
### We can still access elements of arrays and partial arrays in the 
### same way as for lists
first = my_array[0]
print()
print(first, type(first))
part_array = my_array[1:3]
print()
print(part_array)
#### Arrays can be conveniently initialized in several ways
####
#### np.arange(start, stop, step) creates and array starting a start
#### stopping just before stop and stepping in steps of step
start = 10.0
stop = 20.0
step = 1.0
array1 = np.arange(start, stop, step)
print(array1)
### np.zeros(npoints) creates an array containin npoints zeros
n_p = 10
all_zeros = np.zeros(n_p)
print()
print(all_zeros)
### Just as there can be lists containing other lists, there can be 
### multi-dimensional arrays. We can initialize a 2-D array with zeros, for example,
### via np.zeros((npoints1, npoints2)), where the two npoints are the number
### of rows and colummns, respectively
nrows = 10
ncols = 3
all_zeros_2d = np.zeros((nrows, ncols))
print()
print(all_zeros_2d)
###
### Just as for lists, we can access -- and change -- individual elements 
##  via the notation: 
### array[row][column] 
### But it is actually preferably to use the alternative notation
### array[row, column]
all_zeros_2d[0,0] = 100.0
print()
print(all_zeros_2d)
### Note that, when printed, "row zero" is at the top
###
###
### We can mathematically manipulate all elements of an array at once
### (this does NOT work for lists!)
all_ones = all_zeros + 1.0
new_array = all_ones + 10.0
new_array2 = 5.0*new_array / np.sin(all_ones)
print()
print(all_zeros)
print(new_array)
print(new_array2)



### CONDITIONALS (IF-STATEMENTS)
###
### if-statements allow us to execute parts of the code only if certain
### conditions are made. They use Booleans (i.e. things that are either
### True or False). For example, the simplest type of if-statement is
print()
print('CONDITIONALS (IF-STATEMENTS)')
print('============================')
x = 1
if (x > 0):
    print('x = ', x, ' which is greater than zero')
###
### If we want something else to happen if the condition is False, we
### can use the "else" statement 
print()
x = -1
if (x > 0):
    print('x = ', x, ' which is greater than zero')
else:
    print('x = ', x, ' which is not greater than zero')
###
### If there are multiple conditions we want to test, each with their
### own consequent actions, we can use the if-elif-else construction
print()
x = 0
if (x > 1):
    print('x = ', x, ' which is greater than one')
elif (x < -1):
    print('x = ', x, ' which is less than minus one')
else:
    print('x = ', x, ' which is between minus one and one')
###
### We can combine multiple elifs in a single statement
print()
x = -1.5
if (x > 1):
    print('x = ', x, ' which is greater than one')
elif (x < -1) and (x >= -2):
    print('x = ', x, ' which is less than minus one but greater than minus two')
elif (x < -2):
    print('x = ', x, ' which is less than minus two')
else:
    print('x = ', x, ' which is between minus one and one')
###
### We have to be a bit careful though, because if our 
### elifs aren't actually mutually exclusive, the code will simply
### execute the first "True" one it finds....
print()
x = -1.5
if (x > 1):
    print('x = ', x, ' which is greater than one')
elif (x < -1):
    print('x = ', x, ' which is less than minus one')
elif (x < -2):
    ###This is True but not executed, because the previous elif was already True
    print('x = ', x, ' which is less than minus two')
else:
    print('x = ', x, ' which is between minus one and one')
### elifs should always be mutually exclusive



### LOOPS
###
### Loops are a convenient way to repeat the same or similar tasks over and
### over again.
### 
### There are two kinds of loops in Python: for-loops and while-loops
### Usually, we use for-loops if we want to do something a known number of 
### times (or for each element in a sequence). We use while-loops when we 
### don't know ahead of time how often we'll need to repeat things.
###
### for-loops
### 
### In python, the statement:
### 
###     for i in x:
###
### means:
###
### Take the sequence x (a list, an array or a tuple). Assign each
### element of the sequence in turn to the variable i. Then execute
### everything that follows the for-statement and is indented. Repeat
### this until i has taken on all possible values.
###
### For example:
print()
print('LOOPS')
print('=====')
print('for-loops')
print()
print('Looping over [1, 2, 3]')
for i in [1, 2, 3]:
    print('i = ', i)

print()
print('Looping over my_list')
for i in my_list:
    print('i = ', i)

print()
print('Looping over new_array2')
for i in new_array2:
    print('i = ', i)

print()
print('Looping over np.arange(10)')
for i in np.arange(10):
    print('i = ',i)

### while-loops
### 
### In python, the statement:
### 
###     while condition:
###
### means:
###
### Examine the condition -- which must be a boolean (i.e. True or False)
### If it is True, execute everything that follows and is indented. Repeat
### this until the condition becomes False.
###
### Note that in order for a while loop to ever finish, the condition has 
### be updated inside the loop. Otherwise it will always be True if it #
### starts as True
x1 = 1.0
print()
print('While-loop -- example 1')
while (x1 < 10.0):
    print('x1 = ', x1)
    x1 = x1 + 1
    
### Here is a skeleton code to illustrate how this might actually
### be used, e.g. in some sort of game where a separate function
### is called from inside the while loop and decides if the game 
### is over or not. The checking function here is obviously just
### a simple toy code, and it should really go to the top of this
### file. I'm including it here only so that it's easier to see what's
### happening at a single glance.
print()
def update_and_check_for_win(x_in, game_not_over_in):
    game_not_over_out = game_not_over_in    
    if (x_in > 10):
        game_not_over_out = False
    x_out = x_in + 2
    return x_out, game_not_over_out

x = 1
game_not_over = True
print()
print('While-loop -- example 2')
while (game_not_over): 
    print('The game is not over')
    x, game_not_over = update_and_check_for_win(x, game_not_over)
print('The game is over')



## INPUT / OUTPUT (interactive)
## 
## Interactive Input
##
## We can ask for user-input via the keyboard via the "input" function
## Whatever the user enters is then returned as a string
##
print()
x = input("Please enter something: ")
print("You entered ", x, " which will be stored as the variable x")
print("The type of x is ", type(x))
## The easiest way to have Python try to interpret the input
## as a float or an integer, if it can, is to use eval()
x2 = eval(x)
print()
print('x2 = eval(x) produces: ', x2)
print('The type of x2 is ', type(x2))
##
## Interactive Output
##
## Interactive output is just done via the print() statement we've
## used numerous times already. We can make this look nicer by using
## formatting codes like this: 
##
x = np.pi
print()
print('Printing Pi = -%10.7f- to 7 digits in 10 spaces ' % x)
print('              -1234567890-')
print('Printing Pi = -%5.2f- to 2 digits in 5 spaces ' % x)
print('              -12345')
print('Printing Pi = -%4.2f- to 2 digits in 4 spaces ' % x)
print('              -1234-')
##
## For integers, we can use
x = 100
print()
print('Printing one hundred = -%10i- in 10 spaces ' % x)
print('                       -1234567890-')
##
## For strings, we can use
x = 'zakalwe'
print()
print('Printing a string = -%10s- in 10 spaces ' % x)
print('                    -1234567890-')
##
## If the number of spaces we set aside is not enough, the required number
## will automatically be provided
x = 'zakalwe'
print()
print('Trying to printing a 7-letter string = -%3s- in 3 spaces ' % x)
print('                                       -1234567890-')
##
## If we want to have multiple things output on one line, we can use
## tuples
x1 = np.pi
x2 = np.exp(1)
print()
print('Printing Pi = %5.3f and e = %5.3f ' % (x1, x2))

## INPUT / OUTPUT (to/from files)
## 
## File-based Output
## 
## We can store 1-D arrays as column-based output files 
## (e.g. csv files) by using np.savetxt(). By default,
## the numerical values we output this way will be saved
## in scientific notation format, e.g. 1.000000000000000000e+00
##
## Note that we need to use "np.columnstack((x,y,z)) to make 
## sure the output is actually column-based, rather than row-based.
## Note the double parentheses.
##
## When running code in spyder, the folder spyder will save
## this file in is the "Working Directory". This can be set under
## Tools/Preferences/Current Working Directory, or it can be 
## changed directly in the address bar at the top right of the
## spyder window.
## 
x = np.arange(10.0)
y = np.zeros(10)
z = np.ones(10)
print()
print('Writing to a column based file')
np.savetxt('test.dat', np.column_stack((x, y, z)))
##
## By default, savetxt just puts a space between the columns.
## If we want the file to be a comma-separated ("csv") file
## (which can be understood by Excel, for example), we can
## use the optional "delimiter" parameter. This needs to be placed
## at the end, just before the closing parenthesis.
print()
print('Writing to a column based csv file')
np.savetxt('test.csv', np.column_stack((x, y)), delimiter=',')
##
## Note that it is not trivial to save a mix of strings
## and numbers with savetxt (possible, but not  trivial)
## I suggest you avoid doing this for now.

## Saving columns of different *types* with savetxt()
## is... a bit odd. Essentially, we have to use one of the
## formatting statements we used for the formatted print() 
## statements. But the odd thing is that we have to basically
## convert everything to a *string*, i.e. use string formatting
## like %20s 
x = np.arange(10.0)
s = np.array(['zero', 'one', 'two', 'three', 'four', \
     'five', 'six', 'seven', 'eight', 'nine'])
z = np.ones(10)
print()
print('Writing mixed data types to a column based file')
np.savetxt('test2.dat', np.column_stack((x, s, y)), \
           fmt='%20s')


## File-based Output
## 
## We can read in column-based data from files via np.loadtxt()
## We need to remember to the "unpack=True" parameter
## to make sure Python understands that the file is column-based, 
## not row-based. 
## If our file contains two columns, we will usually want to read
## these in into two 1-D arrays, which we do as follows
x2, y2, z2 = np.loadtxt('test.dat', unpack=True)
print()
print('Reading arrays in from a column based file')
print(x2)
print(y2)
print(z2)
## 
## If we want to read in only some of the columns, we can use
## the "usecols" parameter. Note that the first column is "column 0"
## as usual in Python
x3, z3 = np.loadtxt('test.dat', unpack=True, usecols=(0, 2))
print()
print('Reading in the first and third column')
print('-- i.e. columns 0 and 2 -- ')
print('from a column based file')
print(x3)
print(z3)

## the easiest way to use loadtxt to read in different *types*
## of data is by using multiple reads.
x3, z3 = np.loadtxt('test2.dat', unpack=True, usecols=(0, 2))
## to read in *strings*, we also have to explicitly say
## that that's what we want, by using the dtype parameter 
s3 = np.loadtxt('test2.dat', unpack=True, usecols=(1), dtype=str)
print()
print('Reading in mixed data types')
print('from a column based file')
print(x3)
print(s3)
print(z3)

## PLOTTING
## 
## The easiest way to make high-quality plots with Python is
## via the pyplot module in the matplotlib package.
##
## A simple line plot:
print()
print('Plotting lines')
x = np.arange(-10, 10.0,1)
y = 20.0 + x + x**2.0
plt.plot(x, y, label='a quadratic')
## There are various optional parameters to control the appearance
## of the plot, most importantly
y = y + 10.0
plt.plot(x, y, color='red', linestyle='dashed', label='another quadratic')
##
## We can also force the plot to cover particular ranges in x and y
xmin = -12
xmax = 15
ymin = 0
ymax = 1.0e2
plt.xlim((xmin, xmax))
plt.ylim((ymin, ymax))
## The plt.legend function creates a legend on the plot
## that shows the things we put into the "label" parameters
plt.legend()
##
## We can create x and y labels easily
plt.xlabel('X [cm]')
plt.ylabel('Y [V]')
##
## We can add a title to our plot if we want
plt.title('This is the title')
##
## We typically should type plt.show() once we're ready
## for the plot to actually be produced. Up until we say plt.show()
## we can still add stuff to the plot.
plt.show()

## A simple scatter plot:
print()
print('Plotting scatter plots')
plt.scatter(x, y)
plt.show()
## Once again, we can control the appearance
## Note that the size parameter here is "s"
plt.scatter(x, y, color='red', marker='x', s=100)
y = y + 10
plt.scatter(x, y, color='blue', marker='x', s=30)
y = y + 10
plt.scatter(x, y, color='black', marker='o', s=10)
plt.show()

## An alternative scatter plot, using plt.plot
## Note that the size parameter here is "markersize" rather than "s"
## (and the values are not interpreted quite the same way -- just use
## trial and error to find nice sizes)
plt.plot(x, y, color='red', marker='x', markersize=10, linestyle='None')
y = y + 10
plt.plot(x, y, color='blue', marker='x', markersize=5, linestyle='None')
y = y + 10
plt.plot(x, y, color='black', marker='o', markersize=10, linestyle='None')
plt.show()

## Logarithmic axes
## 
## We can generate plot with logarithmic axes by using 
## the loglog or semilogx or semilogy functions
x = np.arange(1.0, 1000.0, 1.0)
y1 = x**1.5
y2 = np.exp(x/100.0)
#
print()
print('linear-linear plots')
plt.plot(x, y1, label='a power law')
plt.plot(x, y2, label='an exponential')
plt.legend()
plt.title('Linear-Linear')
plt.show()
#
print()
print('log-log plots')
plt.loglog(x, y1, label='a power law')
plt.loglog(x, y2, label='an exponential')
plt.legend()
plt.title('Log-Log')
plt.show()
#
print()
print('semi-log plots')
plt.semilogx(x, y1, label='a power law')
plt.semilogx(x, y2, label='an exponential')
plt.legend()
plt.title('Semi-log (x is logarithmic)')
plt.show()
#
plt.semilogy(x, y1, label='a power law')
plt.semilogy(x, y2, label='an exponential')
plt.legend()
plt.title('Semi-log (y is logarithmic)')
plt.show()

## Plotting points with error bars
## We can add error bars to points by using the 
## plt.errorbar function
print()
print('Plotting points with error bars')
x = np.arange(-10, 10.0,1)
y = 20.0 + x + x**2.0
err_x = 0.5
err_y = 0.1 * y
plt.errorbar(x, y, yerr=err_y, \
             markersize=4, marker='o', color='black', linestyle='None')
#
y = y + 20
plt.errorbar(x, y, xerr=err_x, \
             markersize=4, marker='o', color='blue', linestyle='None')
y = y + 20
plt.errorbar(x, y, xerr=err_x, yerr=err_y, \
             markersize=4, marker='o', color='red', linestyle='None')
plt.show()

## Plotting histograms
##
## If we have a bunch of values for which we want to plot a histogram
## we can do this via pyplot.hist
##
## Let's generate 10,000 Gaussian random numbers (see below)
mean = 10.0
stdev = 2.0
ndat = 10000
x = np.random.normal(mean, stdev, size=ndat)
##
## now make a histogram for these, putting them into 40 bins
print()
print('plotting histograms')
nbins = 40
plt.hist(x, bins=nbins, color='red')
plt.title('A filled histogram')
plt.show()
## if we want the histogram to be unfilled we can do that
plt.title('An unfilled histogram')
plt.hist(x, bins=nbins, histtype='step')
plt.show()
## If we want to fairly compare distributions that have
## different number of points in them, we can force
## hist() to normalize them for us -- this effectively creates
## probability density functions that we can also directly
## compare to analytical PDFs (like the Gaussian distribution) 
plt.title('A normalized Histogram')
plt.hist(x, bins=nbins, histtype='step', density=True)
plt.show()


## RANDOM NUMBERS AND DISTRIBUTIONS
##
## We can draw random numbers that are uniformly
## distributed between a and b via np.random.uniform()
a = 10.0
b = 20.0
ndat = 10000
print()
print('Drawing uniform random numbers')
x = np.random.uniform(a, b, size=ndat)
print(x)
nbins = 100
## Note that if we allow Python to set the range of the histogram
## automatically, the bin edges may not align nicely with the
## edges of our interval, so we can get some artefacts in the first
## and last bin.
plt.hist(x, bins=nbins, histtype='step')
plt.show()

## We can draw random numbers from a Gaussian distribution
## with some mean and standard deviation via np.random.normal()
mean = 10.0
stdev = 1.0
ndat = 10000
print()
print('Drawing Gaussian random numbers')
x = np.random.normal(mean, stdev, size=ndat)
print(x)
nbins = 100
plt.hist(x, bins=nbins, histtype='step')
plt.show()


## We can generate the actual shape of the Gaussian distribution
## by using scipy.stats.norm.pdf.
## We should first generate a suitable grid of x values though
## so that the function can then give us the y-values for each x
print()
print('Generating and plotting the analytical Gaussian distribution')
xlo = mean - 5*stdev
xhi = mean + 5*stdev
xstep = 0.1 * stdev
x = np.arange(xlo, xhi, xstep)
y = stats.norm.pdf(x, mean, stdev)
plt.plot(x, y)
plt.title('A Gaussian Distribution')
plt.show()


## As noted above, if we want to plot a histogram and 
## the corresponding distribution on the same plot
## it's convenient to use pyplot.hist with the density=True
## parameter
mean = 10.0
stdev = 2.0
ndat = 10000
ran = np.random.normal(mean, stdev, size=ndat)
plt.hist(ran, bins=nbins, histtype='step', density=True)
#
xlo = mean - 5*stdev
xhi = mean + 5*stdev
xstep = 0.1 * stdev
x = np.arange(xlo, xhi, xstep)
y = stats.norm.pdf(x, mean, stdev)
plt.plot(x, y)
plt.show()


## We can draw random integers from a Poisson distribution
## via np.random.poisson
lamda = 5.3
ndat = 10000
print()
print('Drawing Poisson random numbers')
ran = np.random.poisson(lamda, size=ndat)
print(ran)
## Note that if we allow Python to set the range of the histogram
## automatically, the bin edges may not align nicely with the
## integers, so we can get artefacts. In this case it's 
## best to choose the bin edges ourselves. If the "bins" parameter
## is not an integer, but a sequence, it's interpreted as the location
## of the bin edges
bin_edges = np.arange(0, 20, 1)
plt.hist(ran, bins=bin_edges, histtype='step')
plt.title('Poisson Random Numbers')
plt.show()


## FITTING MODELS TO DATA
##
## Suppose we have a data set consisting of 
## N pairs of (x,y) values, and the y-values (only)
## have associated errors that can be different
## for each point. Let's simulate something like this

## let's create some fake, simulated data
print()
print('Model Fitting -- Generating Simulated Data')
xdat = np.arange(-5.0, 6.0, 1.0)
true_slope = 2.0
true_intercept = 10.0
ytrue = true_slope * xdat + true_intercept
## the error on each y-value is drawn from a Gaussian
## distribution, but the standard deviation can be 
## different for each point. We'll just do this 
## by hand for now.
sigma_y = np.array([1.0, 2.0, 0.5, 2.3, 3.0, 0.7, 1.0, 1.1, 0.4, 2.1, 0.8])
## the actual offset of each observed data point from the true
## value is then just a draw from a Gaussian distribution
## Note that we've usually used np.random.normal by giving it a *single*
## mean and a *single* standard deviation, and then explicitly telling it
## how many random points to generate. But we can also give me *arrays*
## of means and/or standard deviations, in which case it generates one
## random draw for each of the array elements.
offset = np.random.normal(0.0, sigma_y)
ydat = ytrue + offset
##
##let's plot this and make sure it looks ok
plt.scatter(xdat, ydat, linestyle='None')
plt.errorbar(xdat, ydat, yerr=sigma_y,linestyle='None')
plt.plot(xdat, ytrue)
plt.show()

## Now let's use the brute-force method to find the best-fitting
## straight line -- i.e. the optimal slope and intercept -- by
## minimizing chisq. 
## We need to know a sensible range bracketing the true value
## which we can get just from inspecting the plot and getting 
## a rough estimate of the slope and intercept
slope_min = 1.0
slope_max = 3.0
slope_step = 0.02
slopes = np.arange(slope_min, slope_max, slope_step)
#
intercept_min = 5.0
intercept_max = 15.0
intercept_step = 0.1
intercepts = np.arange(intercept_min, intercept_max, intercept_step)

## Now we loop over all slopes and intercepts,
## calculating the chisq for each pair, and storing
## the lowest chisq (and the slope and intercept 
## that produced it).

## we initialize the best chisq to a large number
## so that it definitely gets changed immediately
chi2_best = 1.e50
slope_best = 1.e50
intercept_best = 1.e50

for slope in slopes:
    for intercept in intercepts:
        ymod = slope * xdat + intercept
        chi2 = np.sum((ydat - ymod)**2.0/(sigma_y**2.0))
        if (chi2 < chi2_best):
            chi2_best = chi2
            slope_best = slope
            intercept_best = intercept

print()            
print('Best slope (brute-force) = %8.3f' % slope_best)
print('Best intercept (brute-force) = %8.3f' % intercept_best) 
print('Lowest Chi-squared (brute force) = %8.3f' % chi2_best)

## Now we need to check if this model is good enough.
## We can do this approximately by checking whether
## chi2_best is within 2-3 * sqrt(2*(N-M)) of N-M.
## where N = number of data points and M = number of fit parameters.
##
## Or we can do it numerically, by using the CDF

N = len(xdat)
M = 2 #slope and intercept
p = 1.0 - stats.chi2.cdf(chi2_best, N-M)
print('The likelihood of finding a value of Chi-squared this high')
print('by chance (if the model is correct) is p = %8.3f' %p)

##let's plot this and make sure it looks ok
ymod = slope_best * xdat + intercept_best
plt.scatter(xdat, ydat, linestyle='None')
plt.errorbar(xdat, ydat, yerr=sigma_y,linestyle='None')
plt.plot(xdat, ymod, color='red')
plt.show()

##to get the uncertainties on the slope and the intercept
##we need to find the max and min slopes for which chi2 <= chi2_best + 1
## and similarly for the intercepts. We do this by looping once more.

slope_min = 1.e50
slope_max = -1.e50
intercept_min = 1.e50
intercept_max = -1.e50

for slope in slopes:
    for intercept in intercepts:
        ymod = slope * xdat + intercept
        chi2 = np.sum((ydat - ymod)**2.0/(sigma_y**2.0))

        if (chi2 <= chi2_best+1) and (slope < slope_min):
            slope_min = slope

        if (chi2 <= chi2_best+1) and (slope > slope_max):
            slope_max = slope

        if (chi2 <= chi2_best+1) and (intercept < intercept_min):
            intercept_min = intercept

        if (chi2 <= chi2_best+1) and (intercept > intercept_max):
            intercept_max = intercept

slope_err = (slope_max - slope_min)/2.0
intercept_err = (intercept_max - intercept_min)/2.0
print('The error on the best slope (brute-force) is %8.3f' % slope_err)
print('The error on the best intercept (brute-force) is %8.3f' % intercept_err)
## We should always sanity check that these errors are *larger*
## than the steps we used in slope and intercept. If they are
## not, we need to reduce these steps and repeat.

## We can also do the same thing with curve_fit
## 
## we then need to first define the fitting function
def straight_line(x, m, b):
    y = m*x + b
    return y

pars_start = np.zeros(2) #set up an array holding the starting points for the parameters
pars_start[0] = 1.5 #starting point for slope
pars_start[1] = 20.0 #starting point for intercept
pars_opt, pars_cov = optimize.curve_fit(straight_line, xdat, ydat, \
                                        p0=pars_start, sigma=sigma_y,\
                                        absolute_sigma=True)

slope_best2 = pars_opt[0]
intercept_best2 = pars_opt[1]
print()            
print('Best slope (curve_fit) = %8.3f' % slope_best2)
print('Best intercept (curve_fit) = %8.3f' % intercept_best2) 

    
#curve_fit does not automatically spit out the chi-squared value of the fit
#so we have to calculate this by hand.
ymod2 = straight_line(xdat, slope_best2, intercept_best2)
chi2_best2 = np.sum((ydat - ymod2)**2.0/(sigma_y**2.0))
print('Lowest Chi-squared (curve_fit) = %8.3f' % chi2_best2)

##finally, we get the errors on slope and intercept from the covariance matrix
slope_err2 = pars_cov[0,0]**0.5
intercept_err2 = pars_cov[1,1]**0.5

print('The error on the best slope (curve_fit) is %8.3f' % slope_err2)
print('The error on the best intercept (curve_fit) is %8.3f' % intercept_err2)


## Now suppose that our model is *not* strictly an acceptable
## match to the data. We'll simulate this by adding another
## uncertainy to the y-values, and we'll pretend that we don't
## know about this when we analyse them

print()
print('ALLOWING FOR INTRINSIC DISPERSION')
sigma_y2 = 1.5
offset2 = np.random.normal(0.0, sigma_y2, N)
ydat2 = ydat + offset2

## if we analyze this now in the usual way, but assuming
## the errors are still just the old sigma, we'll get a bad chisq

pars_opt, pars_cov = optimize.curve_fit(straight_line, xdat, ydat2, \
                                        p0=pars_start, sigma=sigma_y,\
                                        absolute_sigma=True)

slope_best2 = pars_opt[0]
intercept_best2 = pars_opt[1]
print()            
print('Best slope (curve_fit; no dispersion) = %8.3f' % slope_best2)
print('Best intercept (curve_fit; no dispersion) = %8.3f' % intercept_best2) 
##finally, we get the errors on slope and intercept from the covariance matrix
slope_err2 = pars_cov[0,0]**0.5
intercept_err2 = pars_cov[1,1]**0.5
print('The error on the best slope (curve_fit, no dispersion) is %8.3f' % slope_err2)
print('The error on the best intercept (curve_fit, no dispersion) is %8.3f' % intercept_err2)

    
# curve_fit does not automatically spit out the chi-squared value of the fit
# so we have to calculate this by hand.
ymod2 = straight_line(xdat, slope_best2, intercept_best2)
chi2_best2 = np.sum((ydat2 - ymod2)**2.0/(sigma_y**2.0))
print('Lowest Chi-squared (curve_fit, no dispersion) = %8.3f' % chi2_best2)
p2 = 1.0 - stats.chi2.cdf(chi2_best2, N-M)
print('The likelihood of finding a value of Chi-squared this high')
print('by chance (if the model is correct) is p = %8.3f' %p2)

## if we decide we nevertheless want to accept this model and 
## calculate its optimal slope and intercept (with errors),
## we need to add an intrinsic dispersion term that results in 
## the minimum chisq become N-M. This is most easily done by trial
## and error, or it could be done in a loop.

## here, we'll just show that for a sigma_intrinsic not too far
## from the sigma_y2 we added, we recover a good chisq. 
## it's important to do this because it affects both the value
## of the parameters and their errors.

sigma_intrinsic = 1.4
sigma_net = np.sqrt(sigma_y**2.0 + sigma_intrinsic**2.0)

pars_opt, pars_cov = optimize.curve_fit(straight_line, xdat, ydat2, \
                                        p0=pars_start, sigma=sigma_net,\
                                        absolute_sigma=True)

slope_best2 = pars_opt[0]
intercept_best2 = pars_opt[1]
print()            
print('Best slope (curve_fit, with dispersion) = %8.3f' % slope_best2)
print('Best intercept (curve_fit, with dispersion) = %8.3f' % intercept_best2) 
slope_err2 = pars_cov[0,0]**0.5
intercept_err2 = pars_cov[1,1]**0.5
print('The error on the best slope (curve_fit, with dispersion) is %8.3f' % slope_err2)
print('The error on the best intercept (curve_fit, with dispersion) is %8.3f' % intercept_err2)
    
# curve_fit does not automatically spit out the chi-squared value of the fit
# so we have to calculate this by hand.
ymod2 = straight_line(xdat, slope_best2, intercept_best2)
chi2_best2 = np.sum((ydat2 - ymod2)**2.0/(sigma_net**2.0))
print('Lowest Chi-squared (curve_fit, with dispersion) = %8.3f -- TARGET = %8.3f' %\
                                                        (chi2_best2,N-M))
p2 = 1.0 - stats.chi2.cdf(chi2_best2, N-M)
print('The likelihood of finding a value of Chi-squared this high')
print('by chance (if the model is correct) is p = %8.3f' %p2)













