import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

'''
basically what you want in linear regression (y=mx+b)
is m and b in above eqation:(M[x] is the mean of x's)

	m = (M[x]*M[y] - M[x*y])/(M[x]^2 - M[x^2])
	b = M[y] - m*M[x]

for error estimation you may use sqaured error method,
for which r^2 is a good estimate:
	
	r^2 = 1 - SE(y_hat)/SE(y_mean)

y_hat is regression line and sqaured error is calulated by
subtracting each data point from corresponding point on the
regression line and sqauring them and summing them up

'''

def seed_data(size, variance, corr_step=2, correlation=False):
	'''random data for testing the regression line
		size: size of list of random numbers
		variance: variance of random numbers
		corr_step: if correlation is set to true, the step of 
					correlation, default is 2
		correlation: pos, neg correaltion or False for no correlation
	'''

	val = 1
	Y = []

	for i in range(size):
		d = val + np.random.uniform(-variance, +variance)
		Y.append(d)
		if correlation and correlation == 'pos':
			val += corr_step
		elif correlation and correlation == 'neg':
			val -= corr_step

	X = [i for i in range(len(Y))]

	X = np.array(X)
	Y = np.array(Y)

	return X, Y

def M(li):
	'''returns:
	the mean of given list
	'''
	return np.mean(li)

def regr_line_params(X, Y):
	'''returns:
	m and b from linear line eqaution
	'''
	X = np.array(X)
	Y = np.array(Y)

	m = (M(X)*M(Y) - M(X*Y)) / (M(X)**2 - M(X**2))
	b = M(Y) - m*M(X)

	return m, b

def regr_line_data(X, m, b):
	'''args:
		X: the x axis data

		returns:
		data for y's of regression line
		corresponding to x's
	'''

	Y_regr = [m*x+b for x in X]
	Y_regr = np.array(Y_regr)

	return Y_regr

def sqaured_err(Y_orig, Y_refr):
	'''args:
	original data point, modeled data point
	
	returns:
	sqaured error for modeled line
	'''
	Y_orig = np.array(Y_orig)
	Y_refr = np.array(Y_refr)

	SE = np.sum((Y_orig - Y_refr)**2)
	return SE

def deter_coeff(Y_orig, Y_regr):
	'''args:
	original data point, data points from regression line

	returns:
	r sqaured error estimate
	'''
	# a list of one value with length of Y_orig
	Y_mean = [M(Y_orig) for i in Y_orig]

	# converting to numpy array is unnecessary here
	# just for completeness, and may use later time
	Y_orig = np.array(Y_orig)
	Y_regr = np.array(Y_regr)
	Y_mean = np.array(Y_mean)

	# sqaured error of mean of Y's line
	SE_Y_mean = sqaured_err(Y_orig, Y_mean)
	# squared error of mean of regression line
	SE_Y_reg = sqaured_err(Y_orig, Y_regr)

	R_squared = 1 - SE_Y_reg/SE_Y_mean

	return R_squared


# first create some seed data
X, Y = seed_data(size=50, variance=50, corr_step=3, correlation='pos')
# obtain m and b of regression line
m, b = regr_line_params(X, Y)
# y's for regression line
Y_regr = regr_line_data(X, m, b)
# error estimate for regresion line
err = deter_coeff(Y, Y_regr)

print("\nCoefficient of Determination: {}\n".format(err))

# plot the data points and regression line
plt.plot(X, Y_regr, label='Regression Line', color='#34495e')
plt.scatter(X, Y, label='Data Points', color='#3498db')
plt.legend()
plt.show()
