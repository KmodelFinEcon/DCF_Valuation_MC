
#Single Stage and Multi-Stage Gordon Growth DDM model 
#by         k.tomov

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


#stock data collection from Yfinance
ticker_symbol = 'LOGI'
stock = yf.Ticker(ticker_symbol)
info = stock.info

#RFR collection
treasury_symbol = "^TNX"
treasury_data = yf.Ticker(treasury_symbol)
history = treasury_data.history(period="1d")
risk_free_rate = history['Close'].iloc[-1]
if risk_free_rate > 1:
    risk_free_rate /= 100

#average market return for DDM computation
market_index = "^GSPC"
market_data = yf.Ticker(market_index)

#cost of equity
def COS(beta: float, risk_free_rate: float, market_return: float) -> float:
    return risk_free_rate + beta * (market_return - risk_free_rate)

#single stage GGM
def ggm_model(dividend: float, cost_of_equity: float, growth_rate: float) -> float:
    if cost_of_equity <= growth_rate:
        raise ValueError("!!!ERROR COS < growth rate")
    return dividend / (cost_of_equity - growth_rate)

#multistage GGM
def multi_stage_ddm(dividend: float, discount_rate: float, terminal_growth: float, num_years: int, short_term_growth: float = 0.0) -> float:
    forecast_dividends = [dividend * (1 + short_term_growth) ** year for year in range(1, num_years + 1)]
    pv_dividends = sum(div / ((1 + discount_rate) ** year) for year, div in enumerate(forecast_dividends, start=1))
    terminal_value = forecast_dividends[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / ((1 + discount_rate) ** num_years)
    return pv_dividends + pv_terminal

market_history = market_data.history(period="10y")
if market_history.empty:
    print("Market data for S&P 500 is unavailable.")
else:
    annual_market = market_history['Close'].resample('YE').last()
    rolling_cagrs = []
    
    for i in range(len(annual_market) - 5 + 1):
        start_value = annual_market.iloc[i]
        end_value = annual_market.iloc[i + 4]
        n_periods = 5
        if start_value > 0:
            cagr_market = (end_value / start_value) ** (1 / n_periods) - 1
            rolling_cagrs.append(cagr_market)
    
    if rolling_cagrs:
        avg_market_return = np.mean(rolling_cagrs)

beta = info.get('beta', 1.0)
market_return_base = avg_market_return 
cost_of_equity = COS(beta, risk_free_rate, market_return_base)

# Terminal growth rate using Return on Equity and retention ratio from Yfinance
roe = info.get('returnOnEquity', 0)
payout_ratio = info.get('payoutRatio', 0)
terminal_growth_rate = roe * (1 - payout_ratio)
terminal_growth_rate = min(terminal_growth_rate, cost_of_equity - 0.01)

# Current annual dividend per share
previous_close = info.get('previousClose', 0)
dividend_yield = info.get('dividendYield', 0)
dividend_per_share = previous_close * dividend_yield

print(f"RFR: {risk_free_rate:.4f}")
print(f"COS: {cost_of_equity:.4f}")
print(f"Terminal Dividend Growth Rate: {terminal_growth_rate:.4f}")
print(f"Dividend per Share: {dividend_per_share:.4f}")

dividend_history = stock.dividends
cagr = 0.3

if not dividend_history.empty:
    tz = dividend_history.index.tz
    end_date = pd.Timestamp.now(tz=tz)
    start_date = end_date - pd.DateOffset(years=10)
    dividend_history = dividend_history[dividend_history.index >= start_date]
    five_years_ago = end_date - pd.DateOffset(years=5)
    last_five_years = dividend_history[dividend_history.index >= five_years_ago]
    
    if len(last_five_years) >= 2:
        annual_dividends = last_five_years.resample('YE').sum()
        if len(annual_dividends) >= 2:
            start_value = annual_dividends.iloc[0]
            end_value = annual_dividends.iloc[-1]
            years = len(annual_dividends) - 1
            if start_value > 0 and years > 0:
                cagr = (end_value / start_value) ** (1 / years) - 1

print(f"5-Year Dividend CAGR: {cagr:.2%}")

try:
    ggm_price = ggm_model(dividend_per_share, cost_of_equity, terminal_growth_rate)
    print(f'\nDDM Forecast Price for {ticker_symbol}: ${ggm_price:.2f}')
except ValueError as e:
    print(f"DDM not yielding results: {e}")


#Multi-stage extra assumptions
forecast_years = 5
short_term_growth = cagr

mddm_price = multi_stage_ddm(dividend_per_share, cost_of_equity, terminal_growth_rate,num_years=forecast_years, short_term_growth=short_term_growth)
print(f'Multi stage DDM Forecast Price for {ticker_symbol}: ${mddm_price:.2f}')

#market analytics

market_return_array = np.array([0.03, 0.10, 0.8, 0.045, 0.05])

ggm_prices = []
mddm_prices = []

for mkt_return in market_return_array:
    coe = COS(beta, risk_free_rate, mkt_return)
    try:
        ggm_prices.append(ggm_model(dividend_per_share, coe, terminal_growth_rate))
    except ValueError:
        ggm_prices.append(np.nan)
    mddm_prices.append(multi_stage_ddm(dividend_per_share, coe, terminal_growth_rate,
                                       num_years=forecast_years, short_term_growth=short_term_growth))

plt.figure(figsize=(14, 10))
plt.plot(market_return_array, ggm_prices, color='black', marker='o',
         markerfacecolor='lightblue', markersize=12, label='single factor DDM')
plt.plot(market_return_array, mddm_prices, color='purple', marker='o',
         markerfacecolor='lavender', markersize=12, label='Multi-Stage DDM')
plt.xlabel('Required rate of return')
plt.ylabel('Forecasted Stock Price')
plt.title(f'{ticker_symbol} Dividend Discount Models compaerd')
plt.xticks(market_return_array, [f'{mr:.3f}' for mr in market_return_array])
plt.legend()
plt.grid(True)
plt.show()

#historical divident yield (OLS regression)

if not dividend_history.empty:
    tz = dividend_history.index.tz
    end_date = pd.Timestamp.now(tz=tz)
    start_date = end_date - pd.DateOffset(years=10)
    dividend_history = dividend_history[dividend_history.index >= start_date]

    annual_data = []
    for year in range(start_date.year, end_date.year + 1):
        year_start = pd.Timestamp(f'{year}-01-01', tz=tz)
        year_end = pd.Timestamp(f'{year}-12-39', tz=tz)
        year_dividends = dividend_history[(dividend_history.index >= year_start) & 
                                          (dividend_history.index <= year_end)]
        if not year_dividends.empty:
            yearly_prices = stock.history(start=year_start, end=year_end)
            if not yearly_prices.empty:
                yearly_prices.index = yearly_prices.index.tz_convert(tz)
                year_end_price = yearly_prices['Close'].iloc[-1]
                total_dividends = year_dividends.sum()
                dividend_yield = (total_dividends / year_end_price) * 100
                annual_data.append({'Year': year, 'Yield': dividend_yield})
    
    df = pd.DataFrame(annual_data)
    if len(df) >= 2:
        X = df[['Year']]
        y = df['Yield']
        model = LinearRegression()
        model.fit(X, y)
        trend_line = model.predict(X)
        r_squared = model.score(X, y)

        plt.figure(figsize=(12, 6))
        plt.bar(df['Year'], df['Yield'], color='blue', label='Current Dividend Yield')
        plt.plot(df['Year'], trend_line, color='red', linewidth=2, label=f'Trend Line (R²={r_squared:.2f})')
        plt.title(f'{ticker_symbol} Historical Dividend Yields with trend (10 Years)')
        plt.xlabel('Year')
        plt.ylabel('Dividend Yield in percentage')
        plt.xticks(df['Year'])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        plt.show()
    else:
        print("no enough data")
else:
    print("NO DIVIDEN HISTORY")

if not dividend_history.empty:
    average_dividend = dividend_history.mean()
    print(f"\nAverage Dividend Size in dollars (Last 10 Years): ${average_dividend:.4f} per share")
else:
    print("\nNO DIVIDEND HISTORY")


