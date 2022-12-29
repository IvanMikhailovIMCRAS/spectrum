import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import minimize

wavenums = np.arange(4000., 600., -0.9367)

def gauss(x, mu, sigma):
    return 1./sigma/np.sqrt(2.*np.pi) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def lorentz(x, x0, gamma):
    return 1./np.pi/gamma * (gamma**2 / (gamma**2 + (x - x0)**2))

data = 1.5 * gauss(wavenums, 2567, 100) + gauss(wavenums, 1230, 200) * 2 + gauss(wavenums, 840, 29)
# plt.plot(wavenums, data)
# plt.show()

def get_derivative(data, n=1, win_wight=13, order=5):
	return savgol_filter(
		data, win_wight, polyorder=order, deriv=n)
  
der2 = get_derivative(data, n=2)

minima = []
for i in range(1, len(der2) - 1):
    if der2[i-1] > der2[i] < der2[i+1]:
        minima.append(i)
print(minima)
threshhold = 0.01 * np.std(data)
minima_wn = []
minima_data = []
for i in minima:
    if data[i] >= threshhold:
        minima_wn.append(wavenums[i])
        minima_data.append(data[i])
        
print(minima_wn)

amplitudes = np.ones(len(minima_wn))
mus = np.array(minima_wn)
sigmas = 1./np.array(minima_data)/np.sqrt(2.*np.pi)

approx_data = np.zeros(len(data))
for a, m, s in zip(amplitudes, mus, sigmas):
    approx_data += a * gauss(wavenums, m, s)

# plt.figure()
# plt.plot(wavenums, data, 'r', wavenums, approx_data,  'b')
# plt.show()

def loss(v, data):
    assert len(v) % 3 == 0 and len(v) != 0
    third = len(v)//3
    ams, mus, sigmas = v[:third], v[third: 2* third], v[2*third:] 
    approx_data = np.zeros(len(data))
    for a, m, s in zip(ams, mus, sigmas):
        approx_data += a * gauss(wavenums, m, s)
    return np.sum(np.square(data - approx_data))

initial_params = np.array(list(amplitudes) + list(mus) + list(sigmas))
res = minimize(lambda x: loss(x, data=data), initial_params, method='Nelder-Mead', tol=1e-6).x
third = len(res) // 3
ams, mus, sigmas = res[:third], res[third: 2* third], res[2*third:] 
approx_data = np.zeros(len(data))
for a, m, s in zip(ams, mus, sigmas):
    approx_data += a * gauss(wavenums, m, s)

plt.plot(wavenums, data, 'b', wavenums, approx_data, 'r')
plt.show()
    
    
    
    
# x0 = [1.3, 0.7, 0.8, 1.9, 1.2]

# res = minimize(rosen, x0, method='Nelder-Mead', tol=1e-6)

# res.x
# array([ 1.,  1.,  1.,  1.,  1.])