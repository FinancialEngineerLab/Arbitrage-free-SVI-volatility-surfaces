import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from scipy.optimize import Bounds
from joblib import Parallel, delayed
import multiprocessing
import sympy
from sympy.solvers import solve
from sympy import Symbol
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt

# otmoptionprice is a DataFrame which contains

# ['Time'], ['Maturity'], ['FuturesPrice'], ['Strike'], ['ImpliedVolatility']

# for out-of-the-money (OTM) options of a SINGLE underlying asset

otmoptionprice_grouped = otmoptionprice.groupby('Time')

# pass no keyword arguments -- using default n = 100

# initial calibration

rslt = applyParallel(otmoptionprice_grouped,sviInit)

otmoptionprice['Moneyness'] = np.log( otmoptionprice['Strike']/otmoptionprice['FuturesPrice'] )

dfk = otmoptionprice[['Time','Maturity','Moneyness']].groupby(['Time','Maturity'])['Moneyness'].agg([np.amin,np.amax]).rename(columns={'amin':'k_min','amax':'k_max'})

# dfsvipInit is initial calibration results parameter DataFrame

dfsvipInit = rslt.join(dfk,how='inner')

# recalibrate each slice

rslt = applyParallel(otmoptionprice_grouped,sviSec,dfsvip=dfsvipInit)

dfsvipSec = rslt.join(dfk,how='inner')

d = 'yyyy-mm-dd'

# depict figures

sviSurface(d,otmoptionprice,dfsvipInit)

sviSurface(d,otmoptionprice,dfsvipSec)
