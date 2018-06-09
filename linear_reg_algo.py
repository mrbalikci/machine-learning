# look for + or - corrolation 
# data set with no corrolation: no best fit line. 
# is there any relationship between X and Y 
# y = mx + b 
# 2d data
# m = (mean(x)mean(y)-mean(xy))/(sqr(mean(x)-mean(sqr(x)))
# b = mean(y) - m.mean(x)

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

# regression line -- best fit line or y hat line 

xs = np.array([1, 2 ,3, 4, 5, 6], dtype = np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype = np.float64)


# best fit slope and y int
def best_fit_slope_intercept(xs, ys):
    
    m = ( (( mean(xs)*mean(ys)) - mean(xs*ys)) /
           ((mean(xs)**2) - mean(xs**2) ))
    
    b = mean(ys) - m*mean(xs)
    
    return m, b

m, b= best_fit_slope_intercept(xs, ys)

# calculate y intercept: b 
# def best_fit_b(xs, ys):
#     b = mean(ys) - m*mean(xs)
#     return b

# b = best_fit_b(xs, ys)

# the model for the data 
print(m, b)


### 
# The Line of Fit 
### 


regression_line = [(m*x)+b for x in xs]

# or 

# for x in xs:
#     regression_line.append((m*x) + b)
###
# How good/accurate is the best fit line?
### 
# squared error 
# r squared = 1 - (squ(error)y^)/(squ(error)mean(y))

def squared_error(ys_orig, ys_line):
    sq = sum((ys_line - ys_orig)**2)
    return sq

def coefficient_of_determination(ys_orig, ys_line):
    
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)

    return 1 - (squared_error_regr / squared_error_y_mean)

# predict 
predict_x = 8
predict_y = (m*predict_x) + b 
print(predict_y)

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

# plot the data and predicted data
plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='r')
plt.plot(xs, regression_line)
plt.show()


