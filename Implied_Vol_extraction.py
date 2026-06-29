import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator, Spline
from mpl_toolkits.mplot3d import Axes3D

from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata


# Black_scholes pricing

def bs_price(F, K, T, sigma, DF_r, option_type="CALLS"):
    if F <= 0 or K <= 0 or DF_r <= 0:
        return np.nan

    if T <= 0 or sigma <= 0:
        if option_type == "CALLS":
            return DF_r * max(F - K, 0.0)
        else:
            return DF_r * max(K - F, 0.0)

    sqrtT = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    if option_type == "CALLS":
        return DF_r * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        return DF_r * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

# Vega

def bs_vega(F, K, T, sigma, DF_r):
    if F <= 0 or K <= 0 or T <= 0 or sigma <= 0 or DF_r <= 0:
        return 0.0

    sqrtT = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)

    return DF_r * F * norm.pdf(d1) * sqrtT


# intrinsic value

def intrinsic_value(F, K, DF_r, option_type="CALLS"):
    if option_type == "CALLS":
        return DF_r * max(F - K, 0.0)
    else:
        return DF_r * max(K - F, 0.0)


# Initial Vol guess

def initial_vol_guess(price, F, K, T, DF_r, option_type="CALLS"):
    intrinsic = intrinsic_value(F, K, DF_r, option_type)
    time_value = max(price - intrinsic, 1e-12)

    guess = np.sqrt(2 * np.pi / T) * time_value / (DF_r * F)             # Brenner-Subrahmanyam style ATM approximation

    return float(np.clip(guess, 0.05, 2.0))            # Keeping guess inside a practical range


# Newton_Raphson IV extractor

def implied_vol_newton( price, F, K, T, DF_r, option_type="CALLS", tol=1e-8, max_iter=20 ):
    sigma = initial_vol_guess(price, F, K, T, DF_r, option_type)

    for _ in range(max_iter):
        price_est = bs_price(F, K, T, sigma, DF_r, option_type)
        diff = price_est - price

        if abs(diff) < tol:
            return sigma

        vega = bs_vega(F, K, T, sigma, DF_r)

        if vega < 1e-10:
            return None

        step = diff / vega
        step = np.clip(step, -0.5, 0.5)                 # Damping prevents Newton from jumping too aggressively
        sigma_new = sigma - step

        if sigma_new <= 0 or sigma_new > 5:
            return None
        sigma = sigma_new

    return None


# Brent's Optimization extractor

def implied_vol_brent( price, F, K, T, DF_r, option_type="CALLS", low=1e-6, high=5.0 ):
    
    def objective(sigma):
        return bs_price(F, K, T, sigma, DF_r, option_type) - price

    try:
        f_low = objective(low)
        f_high = objective(high)

        if np.isnan(f_low) or np.isnan(f_high):
            return np.nan

        if f_low * f_high > 0:
            return np.nan

        return brentq(objective, low, high, xtol=1e-10, rtol=1e-10, maxiter=100)

    except Exception:
        return np.nan


def implied_vol(price, F, K, T, DF_r, option_type="CALLS"):
    values = [price, F, K, T, DF_r]

    if any(pd.isna(x) for x in values):
        return np.nan

    if price <= 0 or F <= 0 or K <= 0 or T <= 0 or DF_r <= 0:
        return np.nan

    intrinsic = intrinsic_value(F, K, DF_r, option_type)

    if price < intrinsic - 1e-10:                          # No-arbitrage lower bound
        return np.nan

    sigma = implied_vol_newton(price, F, K, T, DF_r, option_type)

    if sigma is not None and np.isfinite(sigma):
        return sigma

    return implied_vol_brent(price, F, K, T, DF_r, option_type)


def otm_filter(df, atm_tol = 1e-3):
    call_mask = ( ( df['Strike'] > df['F'] ) & ( df['OptionType'] == 'CALLS'))
    put_mask = (( df['Strike'] < df['F']) & ( df['OptionType'] == 'PUTS'))

    atm = np.abs( df['k'] ) < 0.01 #1e-3
    df_otm = df[ call_mask | put_mask | atm ].copy()
    
    return df_otm


def compute_iv(df):
    df['IV'] = df.apply( lambda row: implied_vol( 
        price = row['Mid'], 
        F = row['F'], 
        K = row['Strike'], 
        T = row['Maturity'], 
        DF_r = row['DF_r'], 
        option_type = row['OptionType']), axis = 1)          
    df['w'] = ( df['IV']**2) * df['Maturity']
    df_ = df.copy()
    df_ = df_[ ( df_['IV'].notna()) & ( df['IV'] > 0.01) & ( df['IV'] < 5.0) ]
    #df_ = ( df.sort_values('Open int.', ascending = False).drop_duplicates( subset = ['Maturity', 'Strike' ] ) )
    return df_


def plot_iv_smile(df):
    maturities = sorted(df['Maturity'].unique())
    
    n_cols = 3
    nrows = np.ceil(len(maturities)/ n_cols).astype(int)
    
    fig, axes = plt.subplots(nrows, n_cols, figsize = (20,10))
    axes = axes.flatten()
    
    for ax, T in zip(axes, maturities):
        tmp = df[df['Maturity']==T]
    
        ax.scatter(tmp['k'], tmp['IV'])
        ax.set_title(f'IV smile | T = {T:.3f} years | {T*365:.1f} days')
        ax.set_xlabel('log-moneyness')
        ax.set_ylabel('Implied Volatility')
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_raw_iv(df):
    k = df['k'].values
    T = df['Maturity'].values
    iv = df['IV'].values

    k_lin = np.linspace(k.min(), k.max(), 500)
    T_lin = np.linspace(T.min(), T.max(), 500)
    K, TT = np.meshgrid(k_lin, T_lin)

    IV_grid = griddata((k, T), iv, (K, TT), method = 'linear')   # can use method = 'linear' for faster interpolation

    fig = plt.figure(figsize= (12, 7))
    ax = fig.add_subplot(111, projection = '3d')
    
    surf = ax.plot_surface(K, TT, IV_grid, cmap = 'viridis', linewidth = 0.5, antialiased = True)
    #surf = ax.scatter(df['k'], df['Maturity'], df['IV'], cmap = 'viridis')
    fig.colorbar(surf, shrink = 0.6, aspect = 10)

    ax.set_xlabel(' log-moneyness ')
    ax.set_ylabel(' Maturity ')
    ax.set_zlabel(' Implied volatility ')
    ax.set_title(' Raw market - Implied voaltility Surface')
    plt.show()


def surface_statistics(df):
    stats = {}

    stats["Total Options"] = len(df)
    stats["Unique Maturities"] = df["Maturity"].nunique()
    stats["Unique Strikes"] = df["Strike"].nunique()

    stats["Mean IV"] = df["IV"].mean()
    stats["Median IV"] = df["IV"].median()
    stats["Std IV"] = df["IV"].std()

    stats["Min IV"] = df["IV"].min()
    stats["Max IV"] = df["IV"].max()

    stats["Min Strike"] = df["Strike"].min()
    stats["Max Strike"] = df["Strike"].max()

    stats["Min k"] = df["k"].min()
    stats["Max k"] = df["k"].max()
    

    stats["Shortest Maturity"] = df["Maturity"].min()
    stats["Longest Maturity"] = df["Maturity"].max()

    stats["Missing IV"] = df["IV"].isna().sum()
    stats["Duplicate (T,K)"] = (df.duplicated(subset=["Maturity", "Strike", "OptionType"]).sum())

# ATM IV
    atm_iv = (df.loc[df.groupby("Maturity")["k"].apply(lambda x: x.abs().idxmin())]
              [["Maturity", "Strike", "IV"]].rename(columns={"IV": "ATM_IV"}))

# Summary Table
    maturity_summary = (df.groupby("Maturity").agg(Strikes=("Strike", "nunique"),
                                                   Options=("IV", "count"),
                                                   Mean_IV=("IV", "mean"),
                                                   Std_IV=("IV", "std"),
                                                   Min_IV=("IV", "min"),
                                                   Max_IV=("IV", "max"),
                                                   Min_k = ('IV', 'min'),
                                                   Max_k = ("IV", 'max')).round(4))
    maturity_summary["Smile Width"] = (maturity_summary['Max_k'] - maturity_summary['Min_k']).round(4)

# Final print
    print("=" * 70)
    print("RAW IMPLIED VOLATILITY SURFACE SUMMARY")
    print("=" * 70)

    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k:<25}: {v:.6f}")
        else:
            print(f"{k:<25}: {v}")

    print("\nATM IMPLIED VOLATILITY")
    print("-" * 70)
    print(atm_iv.to_string(index=False))

    print("\nMATURITY SUMMARY")
    print("-" * 70)
    print(maturity_summary)

    return {"summary": stats, "atm_iv": atm_iv, "maturity_summary": maturity_summary,}

def iv_plots(df):
    df_iv = compute_iv(df)
    maturity_wise_plot = plot_iv_smile(df_iv)
    iv_surface = plot_raw_iv(df_iv)
    surface_stats = surface_statistics(df_iv)
    return df_iv
