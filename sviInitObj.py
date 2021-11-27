def sviInitObj(x,dfsvi,x_shape):   # objective function -- initial fit

  # consider using option prices instead *

  x = x.reshape(x_shape)

  # objective function for ssvi at a given day

  # c1 can be passed as an argument *

  c1 = dfsvi[['Maturity','Theta']].groupby(['Maturity']).first().to_numpy() # column 1 -- size NT

  chi = np.concatenate((c1,x),axis=1)

  m = dfsvi['Maturity'].unique()

  NT = m.shape[0]

  phi = lambda chi: chi[1] / ( chi[0]**chi[2] * (1+chi[0])**(1-chi[2]) )

  # for one slice

  w_svi = lambda chi, k : chi[0]/2 * ( 1 + chi[3]*phi(chi)*k + np.sqrt( (phi(chi)*k + chi[3])**2 + (1 - chi[3]**2) ) )

  e_svi = 0

  for i in range(NT):

    k = dfsvi.loc[dfsvi['Maturity']==m[i],'Moneyness'].to_numpy()

    tv = dfsvi.loc[dfsvi['Maturity']==m[i],'TotalVariance'].to_numpy()

    e_svi += sum( ( tv - w_svi(chi[i,:],k) )**2 )

  e_svi = e_svi * (10**4)   # scaling versus precision *

  return e_svi
