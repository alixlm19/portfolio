---
title: "Statistical Insights into Diversified Portfolio Analysis"
date: "2024-07-05"
description: "Unveiling financial trends and investment strategies through comprehensive statistical analysis. Learn how to analyze portfolio performance, calculate risk metrics, and make data-driven investment decisions."
tags: ["Finance", "Data Analytics", "Python", "Statistics", "Portfolio Management"]
image: ""
featured: true
category: "Finance"
---

In the world of investing, diversification isn't just a buzzword—it's a fundamental principle for managing risk while pursuing returns. But how do you know if your portfolio is truly diversified? And how can you quantify the benefits of diversification?

In this deep dive, I'll walk you through a comprehensive statistical analysis of a diversified investment portfolio. We'll explore modern portfolio theory, calculate key risk metrics, and use Python to uncover insights that can inform smarter investment decisions.

## The Foundation: Modern Portfolio Theory

Harry Markowitz revolutionized finance in 1952 with Modern Portfolio Theory (MPT), which mathematically demonstrates that diversification can reduce portfolio risk without necessarily reducing expected returns. The key insight: it's not just about individual asset performance, but how assets move in relation to each other.

### Core Concepts

- **Expected Return**: The anticipated profit from an investment
- **Volatility (Risk)**: The standard deviation of returns
- **Correlation**: How assets move together
- **Efficient Frontier**: The set of optimal portfolios offering maximum expected return for each risk level

## Building Our Analysis Framework

Let's start by importing the necessary libraries and setting up our analysis environment:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta

# Set styling for professional-looking charts
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

print("Portfolio Analysis Framework Initialized")
```

## Step 1: Data Collection and Preparation

We'll analyze a diversified portfolio spanning multiple asset classes:

- **Stocks**: Large-cap (SPY), tech (QQQ), international (EFA)
- **Bonds**: Treasury (TLT), corporate (LQD)
- **Real Estate**: REIT ETF (VNQ)
- **Commodities**: Gold (GLD)
- **Alternatives**: Emerging markets (EEM)

```python
# Define our portfolio
tickers = {
    'SPY': 'S&P 500',
    'QQQ': 'NASDAQ-100',
    'EFA': 'International Stocks',
    'TLT': 'Treasury Bonds',
    'LQD': 'Corporate Bonds',
    'VNQ': 'Real Estate',
    'GLD': 'Gold',
    'EEM': 'Emerging Markets'
}

# Download 5 years of historical data
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

print(f"Fetching data from {start_date.date()} to {end_date.date()}")

# Download adjusted closing prices
data = yf.download(
    list(tickers.keys()),
    start=start_date,
    end=end_date,
    progress=False
)['Adj Close']

print(f"\nData shape: {data.shape}")
print(f"Missing values: {data.isnull().sum().sum()}")

# Handle missing values
data = data.fillna(method='ffill').fillna(method='bfill')
```

### Calculate Daily Returns

Returns are more statistically useful than prices for portfolio analysis:

```python
# Calculate daily returns
returns = data.pct_change().dropna()

print("\nDaily Returns Statistics:")
print(returns.describe())

# Annualize metrics (252 trading days per year)
annual_returns = returns.mean() * 252
annual_volatility = returns.std() * np.sqrt(252)

# Create summary DataFrame
summary = pd.DataFrame({
    'Asset': [tickers[t] for t in returns.columns],
    'Annual Return (%)': (annual_returns * 100).values,
    'Annual Volatility (%)': (annual_volatility * 100).values,
    'Sharpe Ratio': (annual_returns / annual_volatility).values
}, index=returns.columns)

print("\nAnnualized Performance Metrics:")
print(summary.round(2))
```

## Step 2: Correlation Analysis

Understanding how assets move together is crucial for diversification:

```python
# Calculate correlation matrix
correlation_matrix = returns.corr()

# Visualize correlation heatmap
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

sns.heatmap(
    correlation_matrix,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='RdYlGn',
    center=0,
    square=True,
    linewidths=1,
    cbar_kws={"shrink": 0.8}
)

plt.title('Asset Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Identify highly correlated pairs
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr = correlation_matrix.iloc[i, j]
        if abs(corr) > 0.7:
            high_corr_pairs.append({
                'Asset 1': correlation_matrix.columns[i],
                'Asset 2': correlation_matrix.columns[j],
                'Correlation': corr
            })

print("\nHighly Correlated Assets (|ρ| > 0.7):")
print(pd.DataFrame(high_corr_pairs))
```

### Key Insights from Correlation Analysis

Low or negative correlations between assets indicate better diversification. For example:

- Stocks and bonds typically show negative correlation
- Gold often acts as a hedge during market downturns
- International and domestic stocks may show high correlation during global events

## Step 3: Portfolio Optimization

Let's find the optimal asset allocation using mean-variance optimization:

```python
def portfolio_performance(weights, mean_returns, cov_matrix):
    """
    Calculate portfolio return and risk.
    
    Args:
        weights: Array of asset weights
        mean_returns: Expected returns for each asset
        cov_matrix: Covariance matrix of returns
    
    Returns:
        portfolio_return, portfolio_std
    """
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    return portfolio_return, portfolio_std

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    """
    Calculate negative Sharpe ratio (for minimization).
    """
    p_return, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_return - risk_free_rate) / p_std

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate=0.02):
    """
    Find portfolio with maximum Sharpe ratio.
    """
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    
    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    
    # Bounds: weights between 0 and 1 (long-only portfolio)
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Initial guess: equal weight
    initial_guess = num_assets * [1.0 / num_assets]
    
    # Optimize
    result = minimize(
        negative_sharpe_ratio,
        initial_guess,
        args=args,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result

# Calculate mean returns and covariance matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Find optimal portfolio
optimal_portfolio = max_sharpe_ratio(mean_returns, cov_matrix)
optimal_weights = optimal_portfolio.x
optimal_return, optimal_std = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
optimal_sharpe = (optimal_return - 0.02) / optimal_std

print("\nOptimal Portfolio Allocation:")
allocation_df = pd.DataFrame({
    'Asset': [tickers[t] for t in returns.columns],
    'Weight (%)': (optimal_weights * 100)
}, index=returns.columns)
allocation_df = allocation_df[allocation_df['Weight (%)'] > 0.5]  # Show only significant allocations
print(allocation_df.round(2))

print(f"\nOptimal Portfolio Metrics:")
print(f"Expected Annual Return: {optimal_return*100:.2f}%")
print(f"Annual Volatility: {optimal_std*100:.2f}%")
print(f"Sharpe Ratio: {optimal_sharpe:.2f}")
```

### Visualize Portfolio Allocation

```python
# Create pie chart for optimal allocation
significant_weights = optimal_weights[optimal_weights > 0.01]
significant_tickers = [ticker for ticker, weight in zip(returns.columns, optimal_weights) if weight > 0.01]
significant_labels = [tickers[ticker] for ticker in significant_tickers]

plt.figure(figsize=(10, 8))
colors = plt.cm.Set3(range(len(significant_weights)))

plt.pie(
    significant_weights,
    labels=significant_labels,
    autopct='%1.1f%%',
    startangle=90,
    colors=colors,
    textprops={'fontsize': 11, 'weight': 'bold'}
)

plt.title('Optimal Portfolio Allocation (Max Sharpe Ratio)',
          fontsize=14, fontweight='bold', pad=20)
plt.axis('equal')
plt.tight_layout()
plt.savefig('optimal_allocation.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Step 4: Efficient Frontier

The efficient frontier shows all optimal portfolios—the best possible return for each level of risk:

```python
def efficient_frontier(mean_returns, cov_matrix, num_portfolios=10000):
    """
    Generate portfolios along the efficient frontier.
    """
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    for i in range(num_portfolios):
        # Random weights
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        # Portfolio metrics
        portfolio_return, portfolio_std = portfolio_performance(
            weights, mean_returns, cov_matrix
        )
        
        results[0,i] = portfolio_return
        results[1,i] = portfolio_std
        results[2,i] = (portfolio_return - 0.02) / portfolio_std  # Sharpe ratio
    
    return results, weights_record

# Generate efficient frontier
results, weights_record = efficient_frontier(mean_returns, cov_matrix)

# Plot
plt.figure(figsize=(14, 8))
scatter = plt.scatter(
    results[1,:] * 100,
    results[0,:] * 100,
    c=results[2,:],
    cmap='viridis',
    marker='o',
    s=10,
    alpha=0.3
)

plt.colorbar(scatter, label='Sharpe Ratio')

# Mark optimal portfolio
plt.scatter(
    optimal_std * 100,
    optimal_return * 100,
    marker='*',
    color='red',
    s=500,
    edgecolors='black',
    label='Maximum Sharpe Ratio'
)

# Mark individual assets
for ticker in returns.columns:
    asset_return = annual_returns[ticker]
    asset_std = annual_volatility[ticker]
    plt.scatter(
        asset_std * 100,
        asset_return * 100,
        marker='D',
        s=100,
        edgecolors='black',
        linewidths=1.5
    )
    plt.annotate(
        ticker,
        (asset_std * 100, asset_return * 100),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=9,
        fontweight='bold'
    )

plt.xlabel('Annual Volatility (%)', fontsize=12, fontweight='bold')
plt.ylabel('Annual Return (%)', fontsize=12, fontweight='bold')
plt.title('Efficient Frontier and Asset Performance',
          fontsize=14, fontweight='bold', pad=20)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('efficient_frontier.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Step 5: Risk Analysis

### Value at Risk (VaR)

VaR estimates the maximum loss over a given time period at a specific confidence level:

```python
def calculate_var(returns, confidence_level=0.95):
    """
    Calculate Value at Risk using historical simulation.
    """
    var = np.percentile(returns, (1 - confidence_level) * 100)
    return var

def calculate_cvar(returns, confidence_level=0.95):
    """
    Calculate Conditional Value at Risk (expected loss beyond VaR).
    """
    var = calculate_var(returns, confidence_level)
    cvar = returns[returns <= var].mean()
    return cvar

# Calculate VaR for optimal portfolio
optimal_portfolio_returns = returns.dot(optimal_weights)

daily_var_95 = calculate_var(optimal_portfolio_returns, 0.95)
daily_cvar_95 = calculate_cvar(optimal_portfolio_returns, 0.95)

# Scale to different time horizons
print("\nRisk Metrics for Optimal Portfolio:")
print(f"Daily VaR (95%): {daily_var_95*100:.2f}%")
print(f"Daily CVaR (95%): {daily_cvar_95*100:.2f}%")
print(f"\nMonthly VaR (95%): {daily_var_95*100*np.sqrt(21):.2f}%")
print(f"Annual VaR (95%): {daily_var_95*100*np.sqrt(252):.2f}%")
```

### Maximum Drawdown

Maximum drawdown measures the largest peak-to-trough decline:

```python
def calculate_max_drawdown(returns, weights):
    """
    Calculate maximum drawdown for a portfolio.
    """
    portfolio_returns = returns.dot(weights)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    return max_drawdown, drawdown

max_dd, drawdown_series = calculate_max_drawdown(returns, optimal_weights)

print(f"\nMaximum Drawdown: {max_dd*100:.2f}%")

# Plot drawdown over time
plt.figure(figsize=(14, 6))
plt.plot(drawdown_series.index, drawdown_series * 100, linewidth=1.5, color='darkred')
plt.fill_between(drawdown_series.index, 0, drawdown_series * 100, alpha=0.3, color='red')
plt.xlabel('Date', fontsize=12, fontweight='bold')
plt.ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
plt.title('Portfolio Drawdown Over Time', fontsize=14, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('drawdown_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Step 6: Performance Attribution

Understanding what drives portfolio returns:

```python
def performance_attribution(returns, weights):
    """
    Break down portfolio returns by asset contribution.
    """
    weighted_returns = returns * weights
    contribution = weighted_returns.sum() * 252  # Annualized
    
    attribution_df = pd.DataFrame({
        'Asset': [tickers[t] for t in returns.columns],
        'Weight (%)': weights * 100,
        'Return Contribution (%)': contribution * 100
    }, index=returns.columns)
    
    attribution_df = attribution_df[attribution_df['Weight (%)'] > 0.5]
    attribution_df = attribution_df.sort_values('Return Contribution (%)', ascending=False)
    
    return attribution_df

attribution = performance_attribution(returns, optimal_weights)

print("\nPerformance Attribution:")
print(attribution.round(2))

# Visualize contribution
plt.figure(figsize=(12, 6))
colors = ['green' if x > 0 else 'red' for x in attribution['Return Contribution (%)']]
bars = plt.barh(attribution['Asset'], attribution['Return Contribution (%)'], color=colors, alpha=0.7)

for i, (idx, row) in enumerate(attribution.iterrows()):
    plt.text(
        row['Return Contribution (%)'],
        i,
        f"{row['Return Contribution (%)']:.2f}%",
        va='center',
        ha='left' if row['Return Contribution (%)'] > 0 else 'right',
        fontweight='bold'
    )

plt.xlabel('Return Contribution (%)', fontsize=12, fontweight='bold')
plt.title('Portfolio Return Attribution by Asset', fontsize=14, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('return_attribution.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Step 7: Scenario Analysis

Test portfolio resilience under different market conditions:

```python
def scenario_analysis(returns, weights, scenarios):
    """
    Analyze portfolio performance under different market scenarios.
    """
    results = []
    
    for scenario_name, scenario_returns in scenarios.items():
        # Apply scenario returns to assets
        scenario_portfolio_return = np.sum(weights * scenario_returns)
        results.append({
            'Scenario': scenario_name,
            'Portfolio Return (%)': scenario_portfolio_return * 100
        })
    
    return pd.DataFrame(results)

# Define hypothetical scenarios
scenarios = {
    'Bull Market': np.array([0.20, 0.25, 0.18, 0.05, 0.06, 0.15, 0.10, 0.22]),
    'Bear Market': np.array([-0.15, -0.20, -0.12, 0.08, 0.05, -0.10, 0.15, -0.18]),
    'High Inflation': np.array([0.05, 0.08, 0.03, -0.10, -0.05, 0.12, 0.20, 0.06]),
    'Recession': np.array([-0.10, -0.15, -0.08, 0.12, 0.08, -0.12, 0.08, -0.12]),
    'Stagflation': np.array([-0.05, -0.08, -0.06, 0.03, 0.02, -0.05, 0.15, -0.04])
}

scenario_results = scenario_analysis(returns, optimal_weights, scenarios)

print("\nScenario Analysis:")
print(scenario_results)

# Visualize scenarios
plt.figure(figsize=(12, 6))
colors_scenario = ['green' if x > 0 else 'red' for x in scenario_results['Portfolio Return (%)']]
plt.bar(scenario_results['Scenario'], scenario_results['Portfolio Return (%)'],
        color=colors_scenario, alpha=0.7, edgecolor='black', linewidth=1.5)

for i, (idx, row) in enumerate(scenario_results.iterrows()):
    plt.text(
        i,
        row['Portfolio Return (%)'],
        f"{row['Portfolio Return (%)']:.1f}%",
        ha='center',
        va='bottom' if row['Portfolio Return (%)'] > 0 else 'top',
        fontweight='bold'
    )

plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
plt.ylabel('Portfolio Return (%)', fontsize=12, fontweight='bold')
plt.title('Portfolio Performance Under Different Market Scenarios',
          fontsize=14, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('scenario_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Key Insights and Investment Strategies

### 1. Diversification Works, But It's Not Free

Our analysis shows that diversification can reduce risk significantly, but it also limits upside potential. The optimal portfolio sacrifices some high-risk, high-reward assets for stability.

### 2. Correlation is Dynamic

Asset correlations change over time, especially during market stress when traditionally uncorrelated assets may move together. Regular rebalancing and monitoring are essential.

### 3. Risk-Adjusted Returns Matter More Than Absolute Returns

The Sharpe ratio reveals that lower-return, lower-volatility portfolios often provide better risk-adjusted performance than aggressive strategies chasing maximum returns.

### 4. No Single Optimal Portfolio

The "optimal" portfolio depends on your risk tolerance, investment horizon, and financial goals. The efficient frontier provides a range of options to match different investor profiles.

### 5. Historical Data Has Limitations

Past performance doesn't guarantee future results. Use historical analysis to inform decisions, but incorporate forward-looking assumptions and scenario planning.

## Practical Investment Recommendations

Based on this analysis, here are actionable strategies:

### Rebalancing Strategy

```python
def rebalancing_strategy(current_weights, target_weights, threshold=0.05):
    """
    Determine if rebalancing is needed.
    """
    drift = np.abs(current_weights - target_weights)
    needs_rebalancing = np.any(drift > threshold)
    
    if needs_rebalancing:
        trades = target_weights - current_weights
        return True, trades
    return False, None

# Example: Check if portfolio needs rebalancing
current_weights = np.array([0.30, 0.20, 0.15, 0.10, 0.10, 0.05, 0.05, 0.05])
needs_rebal, trades = rebalancing_strategy(current_weights, optimal_weights, threshold=0.05)

if needs_rebal:
    print("\nRebalancing Recommended:")
    rebal_df = pd.DataFrame({
        'Asset': [tickers[t] for t in returns.columns],
        'Current (%)': current_weights * 100,
        'Target (%)': optimal_weights * 100,
        'Trade (%)': trades * 100
    })
    print(rebal_df[abs(rebal_df['Trade (%)']) > 0.5].round(2))
```

### Dollar-Cost Averaging

For new investors, consider dollar-cost averaging into the optimal portfolio:

- Invest fixed amounts regularly (monthly/quarterly)
- Reduces timing risk
- Takes advantage of market volatility
- Builds discipline

### Tax-Loss Harvesting

Monitor individual positions for tax-loss harvesting opportunities:

- Sell losing positions to offset capital gains
- Replace with similar (but not identical) assets to maintain allocation
- Can save significantly on taxes

## Conclusion

Statistical analysis transforms portfolio management from guesswork to science. By leveraging modern portfolio theory, we can:

- Quantify the benefits of diversification
- Optimize asset allocation for our risk tolerance
- Understand and measure different types of risk
- Make data-driven investment decisions
- Monitor and adjust strategies systematically

Remember that investing involves risk, and no statistical model can predict the future with certainty. Use these tools as part of a comprehensive investment strategy that includes:

- Clear financial goals
- Appropriate time horizon
- Regular monitoring and rebalancing
- Consideration of taxes and fees
- Professional advice when needed

The power of statistical analysis lies not in eliminating uncertainty, but in helping us make better decisions despite it.

**Disclaimer**: This analysis is for educational purposes only and does not constitute financial advice. Consult with a qualified financial advisor before making investment decisions.

The complete code for this analysis is available on my GitHub. Feel free to adapt it for your own portfolio analysis!
