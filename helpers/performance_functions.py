import pandas as pd
import numpy as np
import scipy.stats


def drawdown(return_series: pd.Series):
    '''
    Takes a pandas time series of asset returns
    Computes and returns a DataFrame that Contains:
        the wealth index
        the previous peaks
        percent drawdowns
    '''
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({'Wealth': wealth_index,
                        'Peaks' : previous_peaks,
                        'Drawdown' : drawdowns})


def skewness(r):
    '''
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or Dataframe
    Returns a Float or a Series
    '''
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3


def kurtosis(r):
    '''
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or Dataframe
    Returns a Float or a Series
    '''
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4


def is_normal(r, level=0.01):
    '''
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    '''
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level


def var_historic(r, level=5):
    '''
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns 
    fall below that number, and the 100-"level" percent are above
    '''
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else: 
        raise TypeError('Expected r to be Series or DataFrame')


def var_gaussian(r, level=5, modified=False):
    '''
    Returns the parametric Gaussian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    '''
    # compute the Z score assuming it was Gaussian
    z = scipy.stats.norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6+
                (z**3 - 3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )    
    return -(r.mean() + z*r.std(ddof=0))


def cvar_historic(r, level=5):
    '''
    Computes the Conditional VaR of Series or DataFrame
    '''
    if isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    elif isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    else: 
        raise TypeError('Expected r to be Series or DataFrame')


def annualize_rets(r, periods_per_year):
    '''
    Annualizes a set of returns
    We should infer the periods per year
    but that is currently left as an excersise to the reader
    '''
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1


def annualize_vol(r, periods_per_year):
    '''
    Annualizes the vol of a set of returns
    We should infer the periods per year
    but that is currently left as an excersise to the reader
    '''
    return r.std()*(periods_per_year**0.5)


def sharpe_ratio(r, risk_free_rate, periods_per_year):
    '''
    Computes the annualized sharpe ratio of a set of returns
    '''
    #convert the annual riskfree rate to per period
    rf_per_period = (1+risk_free_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol


def beta(r, benchmark_r):
    '''
    Computes the beta of a set of returns against a benchmark
    '''
    if isinstance(r, pd.DataFrame):
        return r.aggregate(beta, benchmark_r=benchmark_r)
    elif isinstance(r, pd.Series):
        # compute the covariance of the returns with the benchmark
        cov = r.cov(benchmark_r)
        # compute the variance of the benchmark
        var = benchmark_r.var(ddof=0)
        return cov/var
    else:
        raise TypeError('Expected r to be Series or DataFrame')
    

def corr(r, benchmark_r):
    '''
    Computes the correlation of a set of returns against a benchmark
    '''
    if isinstance(r, pd.DataFrame):
        return r.aggregate(corr, benchmark_r=benchmark_r)
    elif isinstance(r, pd.Series):
        return r.corr(benchmark_r) # compute the correlation of the returns with the benchmark
    else:
        raise TypeError('Expected r to be Series or DataFrame')


def performance_summary(r, periods_per_year, risk_free_rate, benchmark_r):
    '''
    Returns DataFrame with CAIA Standardized Returns Analysis Framework
    '''
    tbl =pd.DataFrame({'Annualized Return':annualize_rets(r,periods_per_year),
              'Annualized Volatility':annualize_vol(r, periods_per_year),
              'Skewness':skewness(r),
              'Kurtosis':kurtosis(r),
              'Is Normal':r.aggregate(is_normal),
              'Sharpe Ratio':sharpe_ratio(r, risk_free_rate, periods_per_year),
              'Beta to Benchmark':beta(r, benchmark_r),    
              'Correlation to Benchmark':corr(r, benchmark_r),    
              'Monthly Historic VaR (95%)':var_historic(r),
              'Monthly Historic CVaR (95%)':cvar_historic(r),
              'Max Drawdown':r.aggregate(lambda col: drawdown(col)['Drawdown'].min())
             }).T
        
    return tbl