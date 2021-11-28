otmoptionprice_grouped = otmoptionprice.groupby('Time')

# pass no keyword arguments -- using default n = 100

rslt = applyParallel(otmoptionprice_grouped,sviInit)

otmoptionprice['Moneyness'] = np.log( otmoptionprice['Strike']/otmoptionprice['FuturesPrice'] )

dfk = otmoptionprice[['Time','Maturity','Moneyness']].groupby(['Time','Maturity'])['Moneyness'].agg([np.amin,np.amax]).rename(columns={'amin':'k_min','amax':'k_max'})

dfsvipInit = rslt.join(dfk,how='inner')
