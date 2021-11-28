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

sviSurface(d,otmoptionprice,dfsvipInit)

sviSurface(d,otmoptionprice,dfsvipSec)
