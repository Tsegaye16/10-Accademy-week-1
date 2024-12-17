def CalculateDailyReturn(stock_data):
    stock_data['daily_return'] = stock_data['Close'].pct_change()
    return stock_data