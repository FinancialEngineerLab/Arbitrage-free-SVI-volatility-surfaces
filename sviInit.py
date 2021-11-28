def sviInit(dfi,df,n=100):

  # "An example SVI calibration recipe"

  # w(k,θt) = θt/2 * {1 + ρ φ(θt) k + sqrt[ ( φ(θt) k + ρ )^2 + ( 1 - ρ^2 ) ] }

  # Power-law parameterisation which is consistent with the presence of jumps *

  # φ(θt) = η / ( θ^γ (1+θ)^(1-γ) ), no static arbitrage if η ( 1 + |ρ| ) <= 2

  #-------------------------------------------------------------------------#
  
  # n is the number of initial values used for surface SVI initial calibration

  # input is a dataframe which contains futures price, strike, maturity, implied volatility for a single underlying and for a single day

  # an initial calibration to all tenors (with randomly generated initial values) is applied for the initial values to calibrate each slice

  # forward ATM total variance is based on a linear fit at this stage

  #--------------------------------------------------------------------------#

  df['TotalVariance'] = df['ImpliedVolatility']**2*df['Maturity']/365

  df['Moneyness'] = np.log( df['Strike']/df['FuturesPrice'] )

  df_grouped = df.groupby(by=['Expiration'])

  # theta is based on cubic spline interpolant of total variance *  --> changed to 'linear'

  interp0 = lambda df : pd.Series({'Theta': interpolate.interp1d(x=df['Moneyness'].to_numpy(),y=df['TotalVariance'].to_numpy(),kind='linear',fill_value='extrapolate')(0).item()})

  atmtv = df_grouped.apply(interp0)

  df = df.set_index('Expiration')

  # dfsvi is input for svi calibration which adds theta

  dfsvi = df.join(atmtv,how='inner')

  dfsvi.reset_index(inplace=True)

  # chi is of shape 4 -- [theta,eta,gamma,rho] with chi[1] * ( 1 + abs(rho) ) <= 2

  # the following sets forth the objective function to minimise for initial guess for a given day

  # scipy.optimize.minimize(fun, x0, args)

  # x0 is initiated recursively [theta,eta,gamma,rho,...]

  # eta in (0,2)

  # gamma in (0,1)

  # rho in (-1,1)

  # args is dfsvi

  NT = dfsvi['Maturity'].unique().shape[0]

  bnds = (((0+1e-6,2-1e-6),(0+1e-6,1-1e-6),(-1+1e-6,1-1e-6),)*NT)  # searching space for eta, gamma, rho

  const = NonlinearConstraint(nlc, 0, 2)  # η ( 1 + |ρ| ) <= 2 constraint

  res_store = []

  # multiprocessing

  for i in range(n):

    # generate random starts and minimise

    eta = np.random.uniform(1,2,(NT,1))

    gamma = np.random.uniform(0,1,(NT,1))

    rho = np.random.uniform(-1,1,(NT,1))

    chi_svi = np.concatenate((eta,gamma,rho),axis=1)

    chi_svi_shape = chi_svi.shape

    chi_svi = chi_svi.flatten()  # x0 is (n,) for minimize

    res = minimize(sviInitObj, x0=chi_svi, args=(dfsvi,chi_svi_shape), bounds=bnds, constraints=const)

    theta = dfsvi[['Maturity','Theta']].groupby(['Maturity']).first().to_numpy().flatten()

    res_pro = pd.DataFrame( data = {'obj': res.fun, 'status': res.message, 
                                    'theta': theta, 'eta': res.x[0::3], 
                                    'gamma': res.x[1::3], 'rho': res.x[2::3], 
                                    'eta`': res.jac[0::3], 'gamma`': res.jac[1::3], 
                                    'rho`': res.jac[2::3]} )

    res_store.append(res_pro)

    res_out = pd.concat(res_store)

    res_final = res_out.loc[res_out['obj']==res_out['obj'].min()]

    res_final_index = pd.MultiIndex.from_product([[dfi],dfsvi['Maturity'].unique()],names=['Time','Maturity'])

    res_final.set_index(res_final_index,inplace=True)

  return res_final
