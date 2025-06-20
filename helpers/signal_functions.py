# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_flavor as pf



@pf.register_dataframe_method
def quantile_signals(score_df, long_pct=0.2, short_pct=0.2):
    """
    Long top quantile, short bottom quantile signals
    
    Parameters:
    -----------
    score_df: pandas.DataFrame
        DataFrame with dates as index and stocks as columns
    long_pct: float
        Percentage of stocks to go long (default: top 20%)
    short_pct: float
        Percentage of stocks to go short (default: bottom 20%)
    
    Returns:
    --------
    pandas.DataFrame with same shape as score_df containing signal sizes
    """
    signals = pd.DataFrame(index=score_df.index, columns=score_df.columns, dtype=float)
    
    # Process each date cross-sectionally
    for date in score_df.index:
        row = score_df.loc[date].dropna()
        if len(row) == 0:
            continue
            
        # Calculate cutoffs
        long_cutoff = row.quantile(1 - long_pct)
        short_cutoff = row.quantile(short_pct)
        
        # Assign signals
        long_stocks = row[row >= long_cutoff].index
        short_stocks = row[row <= short_cutoff].index
        flat_stocks = row[(row < long_cutoff) & (row > short_cutoff)].index
        
        signals.loc[date, long_stocks] = 1.0
        signals.loc[date, flat_stocks] = 0.0
        signals.loc[date, short_stocks] = -1.0
    
    return signals


@pf.register_dataframe_method
def quantile_step_signals(score_df, steps=10):
    """
    Create stepped signals based on quantiles
    
    Parameters:
    -----------
    score_df: pandas.DataFrame
        DataFrame with dates as index and stocks as columns
    steps: int
        Number of steps (quantiles)
    
    Returns:
    --------
    pandas.DataFrame with same shape as score_df containing signal sizes
    """
    signals = pd.DataFrame(index=score_df.index, columns=score_df.columns, dtype=float)
    
    for date in score_df.index:
        row = score_df.loc[date].dropna()
        if len(row) == 0:
            continue
            
        if len(row) < steps:
            # Not enough stocks for requested steps
            continue
            
        # Create quantile bins and map to signal sizes
        try:
            quantiles = pd.qcut(row, steps, labels=False)
            signals.loc[date, row.index] = -1 + 2 * (quantiles / (steps - 1))
        except ValueError:
            # Handle case with too many duplicate values
            # Fall back to rank method
            signals.loc[date, row.index] = row.rank(pct=True) * 2 - 1
    
    return signals


@pf.register_dataframe_method
def linear_capped_signals(score_df, upper_pct=0.95, lower_pct=0.05):
    """
    Linear signal sizing with caps at specified percentiles
    
    Parameters:
    -----------
    score_df: pandas.DataFrame
        DataFrame with dates as index and stocks as columns
    upper_pct: float
        Upper percentile for capping positive signals
    lower_pct: float
        Lower percentile for capping negative signals
    
    Returns:
    --------
    pandas.DataFrame with same shape as score_df containing signal sizes
    """
    signals = pd.DataFrame(index=score_df.index, columns=score_df.columns, dtype=float)
    
    for date in score_df.index:
        row = score_df.loc[date].dropna()
        if len(row) == 0:
            continue
            
        # Cross-sectional percentile rank transformed to [-1, 1]
        ranks = row.rank(pct=True) * 2 - 1
        
        # Apply caps using cross-sectional percentiles
        upper_cap = row.quantile(upper_pct)
        lower_cap = row.quantile(lower_pct)
        
        # Scale scores between caps to range [-1, 1]
        for stock in row.index:
            if row[stock] >= upper_cap:
                signals.loc[date, stock] = 1.0
            elif row[stock] <= lower_cap:
                signals.loc[date, stock] = -1.0
            else:
                # Linear scaling for scores between the caps
                scale = 2.0 / (upper_cap - lower_cap)
                mid = (upper_cap + lower_cap) / 2
                signals.loc[date, stock] = (row[stock] - mid) * scale
    
    return signals


@pf.register_dataframe_method
def sigmoid_signals(score_df, steepness=5):
    """
    Transform scores to signals using sigmoid function
    
    Parameters:
    -----------
    score_df: pandas.DataFrame
        DataFrame with dates as index and stocks as columns
    steepness: float
        Controls how quickly signals approach ±1
    
    Returns:
    --------
    pandas.DataFrame with same shape as score_df containing signal sizes
    """
    signals = pd.DataFrame(index=score_df.index, columns=score_df.columns, dtype=float)
    
    for date in score_df.index:
        row = score_df.loc[date].dropna()
        if len(row) == 0:
            continue
            
        # Cross-sectionally center and standardize
        z_scores = (row - row.mean()) / row.std()
        
        # Apply sigmoid transformation
        signals.loc[date, row.index] = 2 / (1 + np.exp(-steepness * z_scores)) - 1
    
    return signals


@pf.register_dataframe_method
def rank_signals(score_df):
    """
    Convert scores to signals based on cross-sectional percentile ranks
    
    Parameters:
    -----------
    score_df: pandas.DataFrame
        DataFrame with dates as index and stocks as columns
    
    Returns:
    --------
    pandas.DataFrame with same shape as score_df containing signal sizes
    """
    signals = pd.DataFrame(index=score_df.index, columns=score_df.columns, dtype=float)
    
    for date in score_df.index:
        row = score_df.loc[date].dropna()
        if len(row) == 0:
            continue
            
        # Calculate percentile ranks for this date and transform to [-1, 1]
        signals.loc[date, row.index] = row.rank(pct=True) * 2 - 1
    
    return signals


@pf.register_dataframe_method
def zscore_signals(score_df, cap=2.5):
    """
    Transform cross-sectional z-scores into signals with caps
    
    Parameters:
    -----------
    score_df: pandas.DataFrame
        DataFrame with dates as index and stocks as columns
    cap: float
        Maximum absolute z-score before capping
    
    Returns:
    --------
    pandas.DataFrame with same shape as score_df containing signal sizes
    """
    signals = pd.DataFrame(index=score_df.index, columns=score_df.columns, dtype=float)
    
    for date in score_df.index:
        row = score_df.loc[date].dropna()
        if len(row) == 0:
            continue
            
        # Calculate z-scores cross-sectionally
        z_scores = (row - row.mean()) / row.std()
        
        # Apply capping and scale to [-1, 1]
        signals.loc[date, row.index] = z_scores.clip(-cap, cap) / cap
    
    return signals


def signal_function_demo(score_df, date=None, figsize=(15, 10)):
    """
    Create a visual comparison of different signal function approaches.
    
    Parameters:
    -----------
    score_df: pandas.DataFrame
        DataFrame with dates as index and stocks as columns
    date: datetime or str, optional
        Specific date to use for visualization (defaults to last date in dataset)
    figsize: tuple, optional
        Figure size (width, height) in inches
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing all visualizations
    """
    # If no date specified, use the latest date
    if date is None:
        date = score_df.index[-1]


    # Create a figure with subplots for each signal function
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    # Take a sample of score data points - let's use the most recent date's scores
    score_sample = score_df.loc[date].to_frame().T.stack(future_stack=True)

    # Apply each signal function and visualize it
    signal_types = ['Quantile', 'Quantile Steps', 'Linear Capped', 'Sigmoid', 'Rank', 'Z-Score']
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan']
    
    # Parameters used for each signal function
    params = [
        'long_pct=0.2, short_pct=0.2',
        'steps=10',
        'upper_pct=0.95, lower_pct=0.05',
        'steepness=5',
        '',  # Rank has no parameters
        'cap=2'
    ]

    # Create a DataFrame to store all signals
    signals_df = pd.DataFrame({'score': score_sample})

    # Calculate signals using each method
    signals_df['Quantile'] = score_df.quantile_signals(long_pct=0.2, short_pct=0.2).stack(future_stack=True).loc[score_sample.index]
    signals_df['Quantile Steps'] = score_df.quantile_step_signals(steps=10).stack(future_stack=True).loc[score_sample.index]
    signals_df['Linear Capped'] = score_df.linear_capped_signals(upper_pct=0.95, lower_pct=0.05).stack(future_stack=True).loc[score_sample.index]
    signals_df['Sigmoid'] = score_df.sigmoid_signals(steepness=5).stack(future_stack=True).loc[score_sample.index]
    signals_df['Rank'] = score_df.rank_signals().stack(future_stack=True).loc[score_sample.index]
    signals_df['Z-Score'] = score_df.zscore_signals(cap=2).stack(future_stack=True).loc[score_sample.index]

    # Plot each signal function in its own subplot
    for i, (pos_type, color, param) in enumerate(zip(signal_types, colors, params)):
        axes[i].scatter(signals_df['score'], signals_df[pos_type], alpha=0.4, color=color, s=30)
        
        # Add labels and grid
        title = f'{pos_type} signal Function'
        if param:  # Add parameters if they exist
            title += f'\n({param})'
        axes[i].set_title(title, fontsize=11)
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[i].axvline(x=0, color='black', linestyle='-', alpha=0.3)

    # Set common axis labels
    fig.text(0.5, 0.01, 'score', ha='center', fontsize=14)
    fig.text(0.01, 0.5, 'signal Size', va='center', rotation='vertical', fontsize=14)
    fig.suptitle('Comparison of signal Functions', fontsize=16)

    # Set equal limits for better visualization
    # Determine dynamic limits based on actual data
    x_min, x_max = signals_df['score'].min(), signals_df['score'].max()
    x_padding = (x_max - x_min) * 0.1  # Add 10% padding
    x_limits = (x_min - x_padding, x_max + x_padding)
    
    # For y-axis, get min/max across all signal columns
    signal_cols = signals_df.columns.drop('score')
    y_min, y_max = signals_df[signal_cols].min().min(), signals_df[signal_cols].max().max()
    y_padding = max(0.1, (y_max - y_min) * 0.1)  # At least 0.1 padding
    y_limits = (y_min - y_padding, y_max + y_padding)
    
    # Apply limits to all axes
    for ax in axes:
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    return fig