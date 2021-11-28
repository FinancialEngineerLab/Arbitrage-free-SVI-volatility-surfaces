def crossedness(chi_0,chi_1):

  # chi contains 4 parameters that defines an svi slice

  # theta, eta, gamma, rho

  k = Symbol('k', real = True)

  phi = lambda chi: chi[1] / ( chi[0]**chi[2] * (1+chi[0])**(1-chi[2]) )

  # for one slice

  w_svi = lambda chi : chi[0]/2 * ( 1 + chi[3]*phi(chi)*k + sympy.sqrt( (phi(chi)*k + chi[3])**2 + (1 - chi[3]**2) ) )

  k0 = solve( w_svi(chi_0) - w_svi(chi_1), k )

  k0 = list(map(float,k0))

  # k0 has a maximum length of 4

  if len(k0) == 0:   # no crossing

    ret = 0

  elif len(k0) == 1:   # 1 crossing

    # "1" means the tolerance on tail crossing is especially low

    k0_c1 = k0[0] - 1

    k0_cn = k0[-1] + 1

    c1 = max( [0, w_svi(chi_0).subs(k,k0_c1)-w_svi(chi_1).subs(k,k0_c1)] )

    cn = max( [0, w_svi(chi_0).subs(k,k0_cn)-w_svi(chi_1).subs(k,k0_cn)] )

    ret = max( [c1,cn] )

  else:

    k0_c1 = k0[0] - 1

    k0_cn = k0[-1] + 1

    c1 = max( [0, w_svi(chi_0).subs(k,k0_c1)-w_svi(chi_1).subs(k,k0_c1)] )

    cn = max( [0, w_svi(chi_0).subs(k,k0_cn)-w_svi(chi_1).subs(k,k0_cn)] )

    c = []

    for i, j in zip( k0[0:-1], k0[1:] ):

      k0_ci = (1/2) * ( i + j )

      ci = max( [0, w_svi(chi_0).subs(k,k0_ci)-w_svi(chi_1).subs(k,k0_ci)] )

      c.append(ci)

    c.append(c1)

    c.append(cn)

    ret = max( c )

  return ret
