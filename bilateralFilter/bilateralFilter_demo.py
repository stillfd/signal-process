import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def wgn(r,c,snr):
	snr = 10**(snr/10.0)
	xp = np.sum((r*c)**2)/(r*c)
	np2 = xp/snr
	return np.random.randn(r,c)*np.sqrt(np2)


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def GuassianFilter(p,sigma, ra):
	
	K = np.zeros((2*ra+1,2*ra+1))
	res = np.zeros((p.shape))
	for i in range(-ra,ra+1):
		for j in range(-ra, ra+1):
			K[i+ra,j+ra] = np.exp(-(i**2+j**2)/(2*(sigma**2)))/(6.28*(sigma**2))
	r,c = p.shape

	pad = np.zeros((r+2*ra, c+2*ra))
	pad[ra:ra+r,ra:ra+c] = p

	for i in range(r):
		for j in range(c):
			cnt = 0
			s 	= 0.0
			res[i,j]= sum(sum(pad[i:ra*2+1+i,j:ra*2+1+j]*K))*1.0/sum(sum(K))
	return res


# sigmb should be small
def BiLateralFilter(p,sigma, sigmb, ra):
	K = np.zeros((2*ra+1,2*ra+1))
	res = np.zeros((p.shape))
	for i in range(-ra,ra+1):
		for j in range(-ra, ra+1):
			K[i+ra,j+ra] = np.exp(-(i**2+j**2)/(2*(sigma**2)))/(6.28*(sigma**2))
	r,c = p.shape

	pad = np.zeros((r+2*ra, c+2*ra))
	pad[ra:ra+r,ra:ra+c] = p

	for i in range(r):
		for j in range(c):
			Roi = pad[i:ra*2+1+i,j:ra*2+1+j]
			H = np.exp(-(Roi - p[i,j])**2/(2*(sigmb**2))) 
			# print(H)
			# break
			
			res[i,j]= sum(sum(pad[i:ra*2+1+i,j:ra*2+1+j]*K*H))*1.0/sum(sum(K*H))
		# break
	return res


img = mpimg.imread('ML.jpg')
print(img.shape)
gray = rgb2gray(img)
gray = gray/np.max(gray)
plt.figure
plt.subplot(221) 
plt.imshow(gray)
plt.subplot(222) 
r,c = gray.shape
noise = wgn(r,c,0.3)
gray2 = gray
# gray2 = gray+0.5*(noise/np.max(noise))
plt.imshow(gray2)
res = GuassianFilter(gray2,2,3)
plt.subplot(223) 
plt.imshow(res)
res = BiLateralFilter(gray2,2,0.1,3)
plt.subplot(224) 
plt.imshow(res)
plt.show()

