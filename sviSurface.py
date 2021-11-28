def sviSurface(d,df,dfsvip,n=10**6):   # -- for 1 day

  # d is the date of interest -- 'yyyy-mm-dd'

  # df is otm option price market data

  # dfsvip is ssvi calibration result

  # n is the # of grid points

  # surface svi with power-law formulation

  # w(k,θt) = θt/2 * {1 + ρ φ(θt) k + sqrt[ ( φ(θt) k + ρ )^2 + ( 1 - ρ^2 ) ] }

  # φ(θt) = η / ( θ^γ (1+θ)^(1-γ) )

  # chi is [theta,eta,gamma,rho]

  phi = lambda chi: chi[1] / ( chi[0]**chi[2] * (1+chi[0])**(1-chi[2]) )

  # for one slice

  w_svi = lambda chi, k : chi[0]/2 * ( 1 + chi[3]*phi(chi)*k + np.sqrt( (phi(chi)*k + chi[3])**2 + (1 - chi[3]**2) ) )

  dfsvip_d = dfsvip.loc[d].reset_index(level='Maturity')

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4), dpi=100)

  # ssvi

  for m_svi in dfsvip_d['Maturity']:

    kgrid = np.linspace(dfsvip_d[dfsvip_d['Maturity']==m_svi]['k_min'],dfsvip_d[dfsvip_d['Maturity']==m_svi]['k_max'],n)

    chi = dfsvip_d[dfsvip_d['Maturity']==m_svi][['theta','eta','gamma','rho']].to_numpy().flatten()

    wgrid = w_svi(chi,kgrid)

    IVgrid = np.sqrt(wgrid/(m_svi/365))

    ax1.plot(kgrid,wgrid,label=m_svi)

    ax2.plot(kgrid,IVgrid,label=m_svi)

  ax2.set_prop_cycle(None)

  # market price

  df_d = df.loc[df['Time']==d]

  for m_otm in df_d['Maturity'].unique():

    k = df_d[df_d['Maturity']==m_otm]['Moneyness'].to_numpy()

    IV = df_d[df_d['Maturity']==m_otm]['ImpliedVolatility'].to_numpy()

    w = df_d[df_d['Maturity']==m_otm]['ImpliedVolatility'].to_numpy()**2*m_otm/365

    ax1.plot(k,w,'k^')

    ax2.plot(k,IV,'o')

  ax1.set(xlabel='k',ylabel='w')
  
  ax2.set(xlabel='k',ylabel='IV')
  
  fig.suptitle('Surface SVI on '+d)

  ax1.legend(loc=6,bbox_to_anchor=(1,.5))

  ax2.legend(loc=6,bbox_to_anchor=(1,.5))

  fig.tight_layout(pad=3.5)
