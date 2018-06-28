# utf-8
import math

def gaussian_filter(sigma=2, derivative_order=0):
	'''
	brief
	-----
	Generate Gaussian filter or its derivative with the input sigma

	g(x) = 1/ (sqrt(2*pi)*sigma) * e^(-1/2*x^2/sigma^2)  

	g_n_(x) = 1/sigma^2 *( g_(n-1)_(x)*x + (n-1)*g_(n-2)*(x)
	'''  
	fltG = []
	hlFltSz = max(3, math.ceil(3*sigma + 1) )
	if derivative_order == 0:
		a = 1. / math.sqrt(2*math.pi)
		for x in range(0, 2*hlFltSz+1):
			x = 1.0*x
			fltG.append( a*math.exp((x - hlFltSz) * (x - hlFltSz)/ (2.*sigma*sigma) ) )
			return fltG
	elif derivative_order == 1:
		return map(lambda x, y: (x - hlFltSz)*y/(2.*sigma*sigma), enumerate(gaussian_filter(sigma, 0)) )
	elif derivative_order >=2:
		filtG0 = gaussian_filter(sigma, derivative_order-2)
		filtG1 = gaussian_filter(sigma, derivative_order-1)
		for i in range(len(filtG0)):
			fltG.append((derivative_order - 1)*filtG0 + filtG1*(i- hlFltSz))/(1.*sigma*sigma)
		return fltG

if __name__ == '__main__':
	
	for i in range(4):
		print(gaussian_filter(2, i))
