#this script used to generate demo for understanding reject sampling
# explanation: some times we cannnot get sample following our demands distribution.
# we can make sample from a easier distribution and apply some rules to make the selected 
# sample follows the one we want

import matplotlib.pyplot as plt
import numpy as np

def normfun(x,mu,sigma):
    pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf
# target distribution: overlap of two gaussian distribution
# easier one: one gaussian distribution

mu = 0
sigma = 10
Num = 10000
resolution  =0.001
upper  = 30
lower = -30
x = np.arange(lower,upper,resolution)
y = 3*normfun(x, -8, 3) + 4*normfun(x, 7, 5)
ye = 15*normfun(x, mu, sigma)
plt.plot(x,y)
plt.plot(x,ye)
plt.show()

# bar for the selection
rule= y/ye
plt.plot(rule)
plt.show()

# generate samples from normal distribution
s = np.random.normal(mu, sigma, Num)
plt.hist(s,20)


line = np.random.random(Num)

res = [];
# key of reject sampling
for i in range(len(s)):
	if(s[i] > lower and s[i]  < upper):
		idx = int((round(s[i]/resolution)*resolution - lower)/resolution)
		if(line[i]<rule[idx]):
			res.append(s[i])
print(len(res))
plt.hist(res,20)
plt.show()




