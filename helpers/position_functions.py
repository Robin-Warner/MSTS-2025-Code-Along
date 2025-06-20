# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_flavor as pf

@pf.register_dataframe_method
def get_portfolio_returns(position, returns):
    return position.mul(returns, axis=0, fill_value=0.0).sum(axis=1)


@pf.register_series_method
def get_portfolio_rolling_volatility(pf_returns, window=21):
    return pf_returns.rolling(window=window, min_periods=window).std().mul(np.sqrt(252))


@pf.register_series_method
def get_portfolio_rolling_beta(pf_returns, mkt_returns, window=21):
    return pf_returns.rolling(window).cov(mkt_returns).div(mkt_returns.rolling(window).var())


def calc_dollar_neutral(position):
    long_total = position[position > 0].sum(axis=1)
    short_total = position[position < 0].sum(axis=1).abs()
    position[position > 0] = position[position > 0].div(long_total, axis=0)
    position[position < 0] = position[position < 0].div(short_total, axis=0)
    return position


def calc_beta_neutral_hedge(pf_returns, mkt_returns, window=21):
    pf_beta = get_portfolio_rolling_beta(pf_returns, mkt_returns, window)
    return pf_beta.shift(1).mul(-1).bfill().rename(mkt_returns.name) 


def calc_gearing_factor(pf_vols, target_vol=0.1, max_gearing=None):
    gearing_factor = pf_vols.pow(-1).mul(target_vol).replace(np.inf, np.nan).bfill().rename('pf_gearing_factor')
    if max_gearing is not None:
        gearing_factor = gearing_factor.clip(lower=-max_gearing, upper=max_gearing)
    return gearing_factor.shift(1).fillna(0.0) 


def position_function(signal_df, returns_df, market_returns_df, target_vol=0.1, dollar_neutral=False, beta_neutral=False, max_gearing=None, max_posn=None, window=21):
    '''
    Transforms a signal dataframe into position sizes with various portfolio construction constraints.
    This function takes signal values and applies portfolio construction techniques including 
    dollar neutrality, beta neutrality, volatility targeting, and position size limits to 
    generate appropriate position sizes.
    Parameters
    ----------
    signal_df : pandas.DataFrame
        DataFrame containing signal values for each asset. Index should be dates.
    returns_df : pandas.DataFrame
        DataFrame containing asset returns. Index should align with signal_df.
    market_returns_df : pandas.DataFrame
        DataFrame containing market returns. Used for beta neutrality.
    target_vol : float, optional
        Target portfolio volatility, default 0.1 (10%).
    dollar_neutral : bool, optional
        If True, ensures the sum of positions equals zero (dollar neutral), default False.
    beta_neutral : bool, optional
        If True, adds a market hedge to achieve beta neutrality, default False.
    max_gearing : float, optional
        Maximum allowed portfolio leverage. If None, no leverage limit is applied.
    max_posn : float, optional
        Maximum allowed position size for any single asset. If None, no position size limit is applied.
    window : int, optional
        Rolling window length for volatility calculations, default 21 days.
    Returns
    -------
    pandas.DataFrame
        DataFrame with position sizes for each asset, respecting all specified constraints.
        Index aligns with the original signal_df.
    Notes
    -----
    This function assumes the existence of helper functions:
    - get_portfolio_returns
    - calc_dollar_neutral
    - calc_beta_neutral_hedge
    - get_portfolio_rolling_volatility
    - calc_gearing_factor
    '''
    position_df = signal_df.copy()
    rets_df = returns_df.copy()
    mkt_rets_df = market_returns_df.copy()
    pf_rets = get_portfolio_returns(position_df, rets_df)

    if dollar_neutral:
        position_df = calc_dollar_neutral(position_df)
        pf_rets = get_portfolio_returns(position_df, rets_df)
   
    if beta_neutral:
        beta_hedge = calc_beta_neutral_hedge(pf_rets, mkt_rets_df)
        position_df = pd.concat([position_df, beta_hedge], axis=1).fillna(0.0)
        rets_df = pd.concat([rets_df, mkt_rets_df], axis=1).fillna(0.0)
        pf_rets = get_portfolio_returns(position_df, rets_df)

    pf_vols = get_portfolio_rolling_volatility(pf_rets, window=window)
    gearing_factor = calc_gearing_factor(pf_vols, target_vol=target_vol, max_gearing=max_gearing)
    position_df = position_df.mul(gearing_factor, axis=0).fillna(0.0)  # apply gearing factor to position

    if max_posn is not None:
        if beta_neutral:
            beta_hedge = position_df[beta_hedge.name]
            position_df = position_df.drop(beta_hedge.name, axis=1)
        
        position_df = position_df.clip(lower=-max_posn, upper=max_posn)  # apply max position limits
        
        if beta_neutral:
            position_df = pd.concat([position_df, beta_hedge], axis=1).fillna(0.0)

    return position_df.fillna(0.0)  # ensure no NaN values in the final position DataFrame


def risk_management_demo(df_positions, df_returns, df_market_returns, start_date=None):
    """
    Visualize key risk metrics for different portfolios.
    
    Parameters:
    -----------
    df_positions : pandas.DataFrame
        DataFrame containing positions with portfolio names in MultiIndex columns
    df_returns : pandas.DataFrame
        DataFrame containing security returns
    df_market_returns : pandas.Series
        Series containing market index returns
    """
    df_pf_positions = df_positions
    
    # Calculate returns for each portfolio
    df_pf_returns = pd.DataFrame()
    for pf in df_pf_positions.columns.get_level_values('portfolio').unique():
        df_pf_returns[pf] = df_pf_positions[pf].get_portfolio_returns(df_returns)

    # Total net notional and gross notional for each portfolio
    df_net_notional = df_pf_positions.T.groupby('portfolio').sum().T.sort_index(axis=1)
    df_gross_notional = df_pf_positions.abs().T.groupby('portfolio').sum().T.sort_index(axis=1)

    # Total gearing and max position for each portfolio
    df_gearing = df_pf_positions.drop(df_market_returns.name, axis=1, errors='ignore', level=1).clip(lower=0).T.groupby('portfolio').sum().T
    df_max_posn = df_pf_positions.drop(df_market_returns.name, axis=1, errors='ignore', level=1).abs().T.groupby('portfolio').max().T

    # Calculate portfolio volatility and beta
    df_volatility = df_pf_returns.apply(lambda x: x.get_portfolio_rolling_volatility(window=40)).dropna().sort_index(axis=1)
    df_beta = df_pf_returns.apply(lambda x: x.get_portfolio_rolling_beta(mkt_returns=df_market_returns, window=40)).dropna().sort_index(axis=1)

    # Create plots
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    # Plot 1: Net Notional (top-left)
    df_net_notional.plot.box(
        ax=axes[0, 0],
        title='Net Notional Positions by Portfolio',
        ylabel='Net Notional',
        grid=True
    )
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45)

    # Plot 2: Gross Notional (top-right)
    df_gross_notional.plot.box(
        ax=axes[0, 1],
        title='Gross Notional Positions by Portfolio',
        ylabel='Gross Notional',
        grid=True
    )
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)

    # Plot 3: Gearing (mid-left)
    df_gearing.plot.box(
        ax=axes[1, 0],
        title='Total Gearing by Portfolio',
        ylabel='Portfolio Gearing',
        grid=True
    )
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45)

    # Plot 4: Max Position (mid-right)
    df_max_posn.plot.box(
        ax=axes[1, 1],
        title='Max Positions by Portfolio',
        ylabel='Max Position',
        grid=True
    )
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45)

    # Plot 5: Volatility (bottom-left)
    df_volatility.plot.box(
        ax=axes[2, 0],
        title='Rolling Volatility of Portfolios',
        ylabel='Rolling Volatility',
        grid=True
    )
    axes[2, 0].set_xticklabels(axes[2, 0].get_xticklabels(), rotation=45)

    # Plot 6: Beta (bottom-right)
    df_beta.plot.box(
        ax=axes[2, 1],
        title='Rolling Beta of Portfolios',
        ylabel='Rolling Beta',
        grid=True
    )
    axes[2, 1].set_xticklabels(axes[2, 1].get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.show()