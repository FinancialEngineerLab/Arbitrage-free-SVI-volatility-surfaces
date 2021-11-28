def sviSec(dfi,df,dfsvip):   # -- for 1 day

  # input is sviInit & otm option price market data *

  # recalibrate all 4 parameters for each slice

  # the purpose is to get rid of unlikely maturity arbitrage for a fine grid of k

  # expect to micro-adjust parameters without ruining the initial estimates *

  df['TotalVariance'] = df['ImpliedVolatility']**2*df['Maturity']/365

  # chi is of shape 4 -- [theta,eta,gamma,rho] with chi[1] * ( 1 + abs(rho) ) <= 2

  # theta in (0,inf) *

  phi = lambda chi: chi[1] / ( chi[0]**chi[2] * (1+chi[0])**(1-chi[2]) )

  # for one slice

  w_svi = lambda chi, k : chi[0]/2 * ( 1 + chi[3]*phi(chi)*k + np.sqrt( (phi(chi)*k + chi[3])**2 + (1 - chi[3]**2) ) )

  # consider using option prices

  svi_obj = lambda chi, k, w : np.sum( ( w_svi(chi,k) - w )**2 ) * (10**4)

  # pick three maturities & use one for penalty design each time

  m = df['Maturity'].unique()

  NT = df['Maturity'].unique().shape[0]

  # η ( 1 + |ρ| ) <= 2

  nlcSec = lambda chi : chi[1] * ( 1 + np.abs(chi[3]) )

  const = NonlinearConstraint(nlcSec, 0, 2)  # η ( 1 + |ρ| ) <= 2 constraint

  # implicitly assume at least two tenors

  res_store = []

  for i in range(NT):

    if i == 0:
      
      # no m_before

      dfsvi_1 = df.loc[df['Maturity']==m[i]]

      dfsvi_2 = df.loc[df['Maturity']==m[i+1]]

      # objective function definition -- input & parameters

      k = dfsvi_1['Moneyness'].to_numpy()

      w = dfsvi_1['TotalVariance'].to_numpy()

      chi_1 = dfsvip.loc[(dfi,m[i])][['theta','eta','gamma','rho']].to_numpy()  # this is x0

      chi_2 = dfsvip.loc[(dfi,m[i+1])][['theta','eta','gamma','rho']].to_numpy()

      svi_obj_p = lambda chi : svi_obj(chi,k,w) + crossedness(chi,chi_2) * (10**5)

      # np.inf versus maximum *

      bnds = ((0+1e-6,np.amax(w)),(0+1e-6,2-1e-6),(0+1e-6,1-1e-6),(-1+1e-6,1-1e-6))  # searching space for theta, eta, gamma, rho
      
      res = minimize(svi_obj_p, x0=chi_1, bounds=bnds, constraints=const)

    elif i == (NT-1):
      
      # no m_after

      dfsvi_0 = df.loc[df['Maturity']==m[i-1]]

      dfsvi_1 = df.loc[df['Maturity']==m[i]]

      # objective function definition -- input & parameters

      k = dfsvi_1['Moneyness'].to_numpy()

      w = dfsvi_1['TotalVariance'].to_numpy()

      chi_1 = dfsvip.loc[(dfi,m[i])][['theta','eta','gamma','rho']].to_numpy()  # this is x0

      chi_0 = dfsvip.loc[(dfi,m[i-1])][['theta','eta','gamma','rho']].to_numpy()

      svi_obj_p = lambda chi : svi_obj(chi,k,w) + crossedness(chi,chi_0) * (10**5)

      # np.inf versus maximum *

      bnds = ((0+1e-6,np.amax(w)),(0+1e-6,2-1e-6),(0+1e-6,1-1e-6),(-1+1e-6,1-1e-6))  # searching space for theta, eta, gamma, rho
      
      res = minimize(svi_obj_p, x0=chi_1, bounds=bnds, constraints=const)

    else:

      # both m_before & m_after

      dfsvi_0 = df.loc[df['Maturity']==m[i-1]]

      dfsvi_1 = df.loc[df['Maturity']==m[i]]

      dfsvi_2 = df.loc[df['Maturity']==m[i+1]]

      # objective function definition -- input & parameters

      k = dfsvi_1['Moneyness'].to_numpy()

      w = dfsvi_1['TotalVariance'].to_numpy()

      chi_0 = dfsvip.loc[(dfi,m[i-1])][['theta','eta','gamma','rho']].to_numpy()

      chi_1 = dfsvip.loc[(dfi,m[i])][['theta','eta','gamma','rho']].to_numpy()   # this is x0

      chi_2 = dfsvip.loc[(dfi,m[i+1])][['theta','eta','gamma','rho']].to_numpy()

      svi_obj_p = lambda chi : svi_obj(chi,k,w) + crossedness(chi,chi_0) * (10**5) + crossedness(chi,chi_2) * (10**5)

      # np.inf versus maximum *

      bnds = ((0+1e-6,np.amax(w)),(0+1e-6,2-1e-6),(0+1e-6,1-1e-6),(-1+1e-6,1-1e-6))  # searching space for theta, eta, gamma, rho
      
      res = minimize(svi_obj_p, x0=chi_1, bounds=bnds, constraints=const)

    res_pro = pd.DataFrame( data = {'obj': res.fun, 'status': res.message, 
                                    'theta': res.x[0], 'eta': res.x[1], 
                                    'gamma': res.x[2], 'rho': res.x[3], 
                                    'theta`': res.jac[0], 'eta`': res.jac[1], 
                                    'gamma`': res.jac[2], 'rho`': res.jac[3]}, 
                            index = [i] )

    res_store.append(res_pro)

  res_out = pd.concat(res_store)

  res_final_index = pd.MultiIndex.from_product([[dfi],m],names=['Time','Maturity'])

  res_final = res_out.set_index(res_final_index)

  return res_final
