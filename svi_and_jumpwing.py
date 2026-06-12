import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import (differential_evolution, minimize)
from scipy.interpolate import CubicSpline, PchipInterpolator

# SVI total variance
def svi_total_variance(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

# Initial params guess
def _initial_guesses(k, w):
    atm_idx = np.argmin(np.abs(k))
    min_idx = np.argmin(w)

    atm_w = max(float(w[atm_idx]), 1e-6)
    min_k = float(k[min_idx])
    left_wing = max(float(np.max(w) - np.min(w)), 1e-3)

    return [
        [0.70 * atm_w, 0.10, -0.30, min_k, 0.08],
        [0.85 * atm_w, 0.20, -0.50, min_k, 0.12],
        [0.95 * atm_w, 0.35, -0.70, min_k, 0.18],
        [0.60 * atm_w, 0.15, -0.10, 0.00, 0.06],
        [max(np.min(w) - 0.25 * left_wing, 1e-6), 0.25, -0.40, min_k, 0.10]  ]


# SVI fit to maturity
def fit_single_maturity(market_df, prev_params = None): 
    k = market_df['k'].to_numpy(dtype=float)
    w = market_df['w'].to_numpy(dtype=float)

    valid = np.isfinite(k) & np.isfinite(w) & (w >= 0)
    k = k[valid]
    w = w[valid]

    if len(k) < 5:
        raise ValueError("Need at least 5 finite option points to fit one SVI maturity.")

    #w = np.maximum(w_df, 1e-8)

    width = max(np.std(k), 0.05)
    #weights = 1.0 + 4.0 * np.exp(-(k / width) ** 2)
    liq_weights = 1 / (market_df['spread'] / (market_df['Mid'] + 1e-6))**2
    atm_weights = np.exp(-4 * np.abs(k))
    weights = liq_weights * atm_weights
    weights = np.clip(weights, 1, 1000)
    
# OBJECTIVE FUNCTION
    def objective(params):
        a, b, rho, m, sigma = params

        if b <= 0:
            return 1e10

        if sigma <= 0:
            return 1e10

        if abs(rho) >= 1:
            return 1e10

        if a + b * sigma * np.sqrt(1 - rho**2) < 0:
            return 1e10

        penalty = 0
        if prev_params is not None:
            b_prev = prev_params['b']
            rho_prev = prev_params['rho']
            sigma_prev = prev_params['sigma']
            
            penalty += 50*(b-b_prev)**2
            penalty += 10*(rho-rho_prev)**2
            penalty += 5*(sigma-sigma_prev)**2

        #if pre_surface is not None:                                ( For Calendar arbitrage )
           # cal_penalty = np.sum(np.max(pre_surface - current_surface, 0)**2)
           # penalty += 1000*cal_penalty
    
        models = svi_total_variance( k, a, b, rho, m, sigma )

        left_wing = b * (1-rho)
        right_wing = b * (1+rho)

        wing_penalty = 0.0
        if left_wing > 0.35:
            wing_penalty += 100 * (left_wing - 0.35)**2

        if right_wing > 0.35:
            wing_penalty += 100 * (right_wing - 0.35)**2
        
        if np.any(~np.isfinite(models)):
            return 1e10
            
# Weighted residual sum of squares
        error = np.sum( weights * (w - models)**2 ) 
        if not np.isfinite(error):
            return 1e10
            
        regularization = ( 0.05 * (sigma - 0.15)**2 + 0.10 * ( rho + 0.5)**2 + 0.05*m**2)
        #regularization = ( (0.01 * (sigma - 0.08)**2) + (0.005 * (rho + 0.4)**2) + (0.001 * m**2) )  ## for smoother and stable parameters
        #regularization = ( 1e-5 * ( sigma - 0.08 )**2 + 1e-5 * (rho + 0.4 )**2 + 1e-6 * (m**2) )
        #regularization = 0
        return error + regularization + wing_penalty + penalty

# PARAMETER BOUNDS        a	       b	      rho          m     	sigma
    #bounds = [ (0, 1), (1e-6, 5), (-0.999,0.999), (-2, 2), (1e-6, 5)  ]   # RELAXED
    
    #bounds = [ (-1.0, 2.0), (1e-5, 5.0), (-0.999, 0.999), (-3.0, 3.0), (1e-5, 3.0) ]       #universal 

    #bounds = [ ( -0.1, 0.5), (1e-4, 2), (-0.999, 0.999), (-0.5, 0.5), (1e-4, 1) ]     # tighter

    #bounds = [ (1e-6, 1.0), (1e-3, 2.0), (-0.85, -0.05), (-1.0, 1.0), ( 0.05, 1.0) ] 

    #bounds = [ (1e-6, 1.0), (1e-3, 2.0), (-0.999, 0.999), (-0.5, 0.5), (0.05, 0.5) ]

    bounds = [ (-0.1, 2.0), (0.01, 0.15), (-0.99, -0.3), (-0.5, 0.5), (0.05, 0.5) ]

    #bounds =  [ (-0.05, 0.5), (0.01, 1.5), (-0.85, -0.05), (-0.05, 0.5), (0.03, 0.5) ]
# GLOBAL OPTIMIZATION
    global_result = differential_evolution( objective, bounds, strategy='best1bin', popsize=20, maxiter=150, tol=1e-7, polish=False, seed = 42 )
    best_result_g = None
    local_result_g = minimize( objective, x0 = global_result.x, method='L-BFGS-B', bounds=bounds )
    if np.isfinite(local_result_g.fun) and (best_result_g is None or local_result_g.fun < best_result_g.fun):
            best_result_g = local_result_g

    if best_result_g is None or not np.all(np.isfinite(best_result_g.x)) or not np.isfinite(best_result_g.fun):
        raise RuntimeError("SVI calibration failed to produce finite parameters.")
    
    
# LOCAL REFINEMENT
    best_result = None
    for init in _initial_guesses(k, w):    
        local_result = minimize( objective, x0 = init, method='L-BFGS-B', bounds=bounds )
        if np.isfinite(local_result.fun) and (best_result is None or local_result.fun < best_result.fun):
            best_result = local_result
        
    if best_result is None or not np.all(np.isfinite(best_result.x)) or not np.isfinite(best_result.fun):
        raise RuntimeError("SVI calibration failed to produce finite parameters.")

    return best_result_g


def svi_results(market_df):
    
    market_df = market_df.copy()
    market_df['Maturity'] = pd.to_numeric(market_df['Maturity'], errors='coerce')
    maturities = np.sort(market_df['Maturity'].dropna().unique())
    results = []

    prev = None
    
    for t in maturities:
        df_T = market_df[ market_df['Maturity'] == t ]
        
        result = fit_single_maturity(df_T, prev_params = prev)
        prev ={'b': result.x[1],
              'rho': result.x[2],
              'sigma': result.x[4]}
        
        a,b,rho,m,sigma = result.x
        
        
        results.append({ 'T'     : t, 
                        'a'     : a,
                        'b'     : b,
                        'rho'   : rho,
                        'm'     : m,
                        'sigma' : sigma,
                        'loss'  : result.fun,
                        'success': result.success })

    results_df = pd.DataFrame(results)
    return results_df, result 
    

# ATM variance
def compute_atm_variance(a, b, rho, m, sigma):
    return svi_total_variance(0.0, a, b, rho, m, sigma)


# ATM volatility
def compute_atm_vol(a, b, rho, m, sigma, tau):
    w = compute_atm_variance(a, b, rho, m, sigma)
    return np.sqrt(w / tau)


# ATM skew
def compute_atm_skew(a, b, rho, m, sigma):
    return b * (rho + (-m / np.sqrt(m**2 + sigma**2)))


# ATM curvature
def compute_atm_curvature(a, b, rho, m, sigma):
    return (b * sigma**2 / (m**2 + sigma**2)**1.5)


# Wing slopes
def compute_left_wing_slope(b, rho):
    return b * (rho - 1)

def compute_right_wing_slope(b, rho):
    return b * (1 + rho)


def svi_shape_features(results_df):
    
 # ATM VARIANCE
    results_df['ATMVar'] = results_df.apply( 
        lambda row: compute_atm_variance( row['a'], row['b'], row['rho'], row['m'], row['sigma'] ), axis=1 )

# ATM VOLATILITY
    results_df['ATMVol'] = results_df.apply(
        lambda row: compute_atm_vol( row['a'], row['b'], row['rho'], row['m'], row['sigma'], row['T'] ), axis=1 )

# ATM SKEW
    results_df['ATMSkew'] = results_df.apply(
        lambda row: compute_atm_skew( row['a'], row['b'], row['rho'], row['m'], row['sigma'] ), axis=1 )

# ATM CURVATURE
    results_df['Curvature'] = results_df.apply(
        lambda row: compute_atm_curvature( row['a'], row['b'], row['rho'], row['m'], row['sigma'] ), axis=1 )

# LEFT WING SLOPE
    results_df['LeftWing'] = results_df.apply(
        lambda row: compute_left_wing_slope( row['b'], row['rho'] ), axis=1 )

# RIGHT WING SLOPE
    results_df['RightWing'] = results_df.apply(
        lambda row: compute_right_wing_slope( row['b'], row['rho'] ), axis=1 )
    
    return results_df

def plot_svi_shape(results_df):
    
    fig, ax = plt.subplots(3, 2, figsize=(10, 8))

    ax[0, 0].plot(results_df['T'], results_df['ATMVar'], 'o-', label = 'ATM variance')
    ax[0, 0].set_title('ATM Variance')

    ax[0, 1].plot(results_df['T'], results_df['ATMVol'], 'o-', label = 'ATM Vol')
    ax[0, 1].set_title('ATM Vol')

    ax[1, 0].plot(results_df['T'], results_df['ATMSkew'], 'o-', label = 'ATM Skew')
    ax[1, 0].set_title('ATM Skew')

    ax[1, 1].plot(results_df['T'], results_df['Curvature'], 'o-', label = 'Curvature')
    ax[1, 1].set_title('Curvature')

    ax[2, 0].plot(results_df['T'], results_df['LeftWing'], 'o-', label = 'left_wing')
    ax[2, 0].set_title('Left Wing')

    ax[2, 1].plot(results_df['T'], results_df['RightWing'], 'o-', label = 'right_wing')
    ax[2, 1].set_title('Right Wing')
    plt.tight_layout()


### GATHERAL JUMP WINGS

def svi_to_jumpwing(T,a,b,rho,m,sigma):
    w0 = a + b*( -rho*m + np.sqrt(m*m + sigma*sigma) )

    wp0 = b*( rho - m/np.sqrt(m*m + sigma*sigma) )

    atm_vol = np.sqrt(w0/T)

    return { "T":T,
            "w0":w0,
            "v_t":w0/T,
            "atm_vol":atm_vol,
            "psi_t":wp0/np.sqrt(w0),
            "p_t":b*(1-rho)/np.sqrt(w0),
            "c_t":b*(1+rho)/np.sqrt(w0) }

def jump_wings(df):
    rows = []
    for _, row in df.iterrows():
        rows.append( svi_to_jumpwing(row['T'], row['a'], row['b'], row['rho'], row['m'], row['sigma']))
    
    return pd.DataFrame(rows)
    
def plot_jump_wings(jw_df):
    fig, ax = plt.subplots(2,2, figsize=(12,8))
    ax[0,0].plot(jw_df["T"], jw_df["v_t"], marker='o')
    ax[0,0].set_title("ATM Variance")

    ax[0,1].plot(jw_df["T"], jw_df["psi_t"], marker='o')
    ax[0,1].set_title("ATM Skew")

    ax[1,0].plot(jw_df["T"], jw_df["p_t"], marker='o')
    ax[1,0].set_title("Left Wing")

    ax[1,1].plot(jw_df["T"], jw_df["c_t"], marker='o')
    ax[1,1].set_title("Right Wing")

    plt.tight_layout()
    
    
#def svi_first_derivative(k, b, rho, m, sigma):
   # return b * ( rho + (k - m) / np.sqrt((k - m)**2 + sigma**2) )


#def svi_second_derivative(k, b, m, sigma):
   # return ( b * sigma**2 / ((k - m)**2 + sigma**2)**(1.5) )

def butterfly_arbitrage(k, a, b, rho, m, sigma):
    k = np.asarray(k, dtype=float)
    w = svi_total_variance(k, a, b, rho, m, sigma)

    x = k - m
    root = np.sqrt(x**2 + sigma**2)
    w_k = b * (rho + x / root)
    w_kk = b * sigma**2 / root**3

    w = np.where(w > 1e-12, w, np.nan)
        
    g = ( (1 - (k * w_k) / (2 * w))**2 - (w_k**2 / 4) * (1 / w + 1/4) + w_kk / 2 )
        
    return g


def run_butterfly_checks(k, results_df):
    butterfly_checks = []
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, row in results_df.iterrows():
        g_vals = butterfly_arbitrage( k, row['a'], row['b'], row['rho'], row['m'], row['sigma'] )
        min_g = np.nanmin(g_vals)
        
        violations = np.sum(g_vals < 0 )
        butterfly_checks.append({'T' : row['T'], 'Min g(k)' : min_g, 'Violations' : violations})
        butterfly_ch = pd.DataFrame(butterfly_checks)
        ax.plot(k, g_vals, 'o-', label = f"{row['T']:.3f}")

    ax.set_title('Butterfly Spread')
    ax.legend()
    ax.axhline(0, color='black', linestyle='--')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    
        #if min_g >= 0:
            #print(" No Butterfly Arbitrage")
        #else:
           # print(f" Butterfly Arbitrage Detected :  {violations}")
    return butterfly_ch



def run_calendar_checks(k, results_df):
    calendar_checks = []
    fig, ax = plt.subplots(figsize=(10, 6))
        
    for i in range(len(results_df)-1):
        row1 = results_df.iloc[i]
        row2 = results_df.iloc[i+1]
        
        w1 = svi_total_variance( k, row1['a'], row1['b'], row1['rho'], row1['m'], row1['sigma'] )
        w2 = svi_total_variance( k, row2['a'], row2['b'], row2['rho'], row2['m'], row2['sigma'] )
        
        diff = w2 - w1
        min_diff = np.min(diff)
        violations = np.sum( w2 < w1 )
        
        calendar_checks.append({'T' : f"{row1['T']:.3f} -> {row2['T']:.3f}", 'Min_diff' : min_diff, 'Violations' : violations})
        calendar_ch = pd.DataFrame(calendar_checks)
        ax.plot(k, diff, 'o-', label = f"{row1['T']:.3f} -> {row2['T']:.3f}")
        
    ax.set_title('Calendar Spread')
    ax.legend()
    ax.axhline(0, color='black', linestyle='--')
    ax.grid(alpha=0.25)
    fig.tight_layout()
 
        

        #if min_diff >= 0:
           #print(" No Calendar Arbitrage")
        #else:
           # print(f" Calendar Arbitrage Detected : {violations}" )
        #print(f"{row1['T']:.3f} -> {row2['T']:.3f}")


       #print("Minimum Difference:", min_diff)

       # print("-"*40)
    return calendar_ch



def arbitrage_checks(k, results_df):
    butterfly = run_butterfly_checks(k, results_df)
    calendar = run_calendar_checks(k, results_df)
    return butterfly, calendar


def svi_market_plot(df, results_df):
    maturities = sorted(df['Maturity'].unique())
    fig, axes = plt.subplots(3, 3, figsize = (15, 10))
    axes = axes.flatten()
    
    for ax, T in zip(axes, maturities):
        df_T = df[df['Maturity'] == T]
        
        #result = fit_single_maturity(df_T)
        #a,b,rho,m,sigma = result.x
        
        k_grid = np.linspace(df_T['k'].min(), df_T['k'].max(), 500)

        fit_row = results_df.loc[np.isclose(results_df['T'], T)].iloc[0]
        
        w_svi = svi_total_variance(k_grid, fit_row['a'], fit_row['b'], fit_row['rho'], fit_row['m'], fit_row['sigma'] )
        ax.scatter(df_T['k'], df_T['w'], s = 10, label = 'Market Variance')
        ax.plot(k_grid, w_svi, 'r', lw = 2, label = 'SVI variance')
        ax.set_title(f'T = {T:.3f}')
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.show()
    
       
        








