import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def plot_mojo(df: pd.DataFrame, x: str, y: str, q: int = 10, percentiles: list = [10, 90], ax=None, figsize=(8, 6), title: str = None, show_outliers: bool = True):
    """
    Plot a 'mojo chart', a quantile-based visualization showing the conditional distribution
    of a response variable (`y`) given quantiles of a predictor (`x`).

    For each quantile of `x`, the function plots:
    - A vertical bar from the lower to upper percentile of `y`
    - A red "x" at the mean of `y`
    - Blue dots for outliers (values of `y` outside the given percentile band)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the x and y variables to analyze.
    x : str
        Column name of the predictor variable.
    y : str
        Column name of the response variable.
    q : int, optional
        Number of quantile bins to divide `x` into (default is 10).
    percentiles : list, optional
        List of two percentiles [low, high] to define the inner range of `y` (default is [10, 90]).
        Values outside this range are considered outliers.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axis to plot on. If None, a new figure and axis are created.
    figsize : tuple, optional
        Size of the figure if ax is None (default is (8, 6)).
    title : str, optional
        Title of the subplot.

    Raises
    ------
    ValueError
        If `percentiles` is not a list of two values between 0 and 100.

    Returns
    -------
    None
        The function modifies the provided axis in-place or creates a new one for display.
    """

    df = df[[x, y]].dropna()
    if df.empty or df[x].nunique() < 3:
        if ax:
            ax.set_title("Insufficient data")
        return

    if len(percentiles) != 2 or not all(0 <= p <= 100 for p in percentiles):
        raise ValueError("percentiles must be a list of two values between 0 and 100, e.g., [10, 90]")

    try:
        df['quantile'] = pd.qcut(df[x], q=q, labels=False, duplicates='drop')
    except ValueError:
        if ax:
            ax.set_title("Quantiling failed")
        return

    p_low, p_high = percentiles

    grouped = df.groupby('quantile').agg(
        x_mean=(x, 'mean'),
        p_low=(y, lambda s: np.percentile(s, p_low)),
        p_high=(y, lambda s: np.percentile(s, p_high)),
        y_mean=(y, 'mean')
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.vlines(grouped['x_mean'], grouped['p_low'], grouped['p_high'], color='gray', lw=4)
    ax.scatter(grouped['x_mean'], grouped['y_mean'], color='red', marker='x', s=100, label='Mean')

    for q_idx, group in df.groupby('quantile'):
        x_val = grouped.loc[q_idx, 'x_mean']
        low = grouped.loc[q_idx, 'p_low']
        high = grouped.loc[q_idx, 'p_high']
        if show_outliers:
            # Plot outliers for the current quantile group
            outliers = group[(group[y] < low) | (group[y] > high)]
            ax.scatter([x_val] * len(outliers), outliers[y], color='blue', alpha=0.6, s=30)

    ax.set_xlabel(f"{x} (quantiled)", fontsize=12)
    ax.set_ylabel(y, fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    ax.grid(True)
    ax.legend(loc='lower center')


def plot_scatter_with_line_of_best_fit(x, y, title='', xlabel='', ylabel=''):
    """
    Plot a scatter plot with a line of best fit (linear regression).

    This function creates a scatter plot from x and y data, calculates the linear
    regression, and adds the line of best fit to the plot. It also displays the
    regression equation, R-squared value, and p-value in the legend.

    Parameters
    ----------
    x : array-like or pandas.Series
        Data for the x-axis.
    y : array-like or pandas.Series
        Data for the y-axis.
    title : str, optional
        Title for the plot (default is '').
    xlabel : str, optional
        Label for the x-axis (default is '').
    ylabel : str, optional
        Label for the y-axis (default is '').

    Notes
    -----
    - NaN values are automatically filtered out from both x and y data.
    - The function uses scipy.stats.linregress to calculate the regression.
    - The plot is automatically displayed using plt.show().

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> x = pd.Series([1, 2, 3, 4, 5])
    >>> y = pd.Series([2, 3.5, 5, 6.2, 7.8])
    >>> plot_scatter_with_line_of_best_fit(x, y, title='Example Plot', 
    ...                                  xlabel='X values', ylabel='Y values')
    """
    mask = x.notna() & y.notna()  # mask to filter out NaN values
    x = x[mask]
    y = y[mask]
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.5)
    plt.plot(x, slope * x + intercept, color='red', label=f'Line of Best Fit: y={slope:.5f}*x + {intercept:.5f} (R²={r_value**2:.5f}, p={p_value:.5f})')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.show()


def plot_transition_matrix(x, y, num_of_quantiles:int=5, title='', xlabel='', ylabel='', normalize='index'):
    """
    Plots a transition matrix/heatmap showing the probability distribution between quantiles of two variables.
    This function creates a visualization to analyze how values in the input variable (x) 
    transition to values in the target variable (y) by dividing both into quantiles and
    displaying the conditional probability distribution.
    Parameters
    ----------
    x : array-like or Series
        The input variable to be quantized and used as the "from" state.
    y : array-like or Series
        The target variable to be quantized and used as the "to" state.
    num_of_quantiles : int, default=5
        Number of quantiles to divide each variable into.
    title : str, default=''
        Title for the plot.
    xlabel : str, default=''
        Label for the x-axis (target variable).
    ylabel : str, default=''
        Label for the y-axis (input variable).
    normalize : {'index', 'columns', 'all'}, default='index'
        Normalization method for the transition matrix:
        - 'index': Each row sums to 1 (conditional probability given the row)
        - 'columns': Each column sums to 1
        - 'all': The entire table sums to 1
    Returns
    -------
    None
        The function displays the plot but does not return any value.
    Notes
    -----
    - NaN values in either x or y are filtered out before processing.
    - The heatmap is arranged with the highest quantile of the input variable at the top
      and the lowest quantile of the target variable at the left.
    - Values in cells represent the probability of transitioning from the input quantile 
      to the target quantile.
    """
    mask = x.notna() & y.notna()  # mask to filter out NaN values
    x = x[mask]
    y = y[mask]
    x_quantiles = pd.qcut(x, num_of_quantiles, labels=False, duplicates='drop')
    y_quantiles = pd.qcut(y, num_of_quantiles, labels=False, duplicates='drop')
    
    # Create the transition matrix - normalize by index (rows)
    transition_matrix = pd.crosstab(x_quantiles, y_quantiles, normalize=normalize)
    
    # Sort the transition matrix rows with lowest signal quantile (0) at bottom
    # This makes the visualization more intuitive with stronger signals at top
    transition_matrix = transition_matrix.sort_index(ascending=False)
    
    # Create a figure
    plt.figure(figsize=(10, 6))
    
    # Create heatmap
    im = plt.imshow(transition_matrix, cmap='Blues', aspect='auto')
    plt.colorbar(label='Transition Probability')
    
    # Add text annotations with probabilities
    for i in range(len(transition_matrix)):
        for j in range(len(transition_matrix.columns)):
            text = plt.text(j, i, f'{transition_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black" if transition_matrix.iloc[i, j] < 0.4 else "white")
    
    # Create more meaningful tick labels
    # For y-axis (signals), highest signal (Q5) at top, lowest (Q1) at bottom
    plt.yticks(ticks=np.arange(num_of_quantiles), 
              labels=[f'Q{num_of_quantiles-i} Highest' if i==0 else 
                     f'Q{num_of_quantiles-i} Lowest' if i==num_of_quantiles-1 else 
                     f'Q{num_of_quantiles-i}' for i in range(num_of_quantiles)])
    
    # For x-axis (future returns), lowest (Q1) at left, highest (Q5) at right
    plt.xticks(ticks=np.arange(num_of_quantiles), 
              labels=[f'Q{i+1}\nLowest' if i==0 else 
                     f'Q{i+1}\nHighest' if i==num_of_quantiles-1 else 
                     f'Q{i+1}' for i in range(num_of_quantiles)])
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(False)
    plt.show()