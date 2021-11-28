# Parallel Computing groupby svi

def applyParallel(dfGrouped,func,**kwargs):

  # unpack kwargs which is a storage unit

  rslt = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(name,group,**kwargs) for name, group in dfGrouped)

  return pd.concat(rslt)
