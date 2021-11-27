def nlc(x):   # no-arbitrage constraints -- η ( 1 + |ρ| ) <= 2

  NT = x.shape[0] // 3     # always 3 parameters -- eta, gamma, rho

  idx = np.array([0,2])

  con = np.array([])

  for i in range(NT):

    # e.g. [0,2],[3,5],[6,8],[9,11]

    coni = x[idx[0]] * ( 1 + np.abs(x[idx[1]]) )

    idx += 3

    con = np.concatenate((con,coni),axis=None)

  return con
