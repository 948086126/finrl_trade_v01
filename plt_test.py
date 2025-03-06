import akshare as ak

stock_zh_a_hist_min_em_df = ak.rv_from_stock_zh_a_hist_min_em(
    symbol="000001",
    start_date="2021-10-20 09:30:00",
    end_date="2024-11-01 15:00:00",
    period="5",
    adjust="hfq",
)

print(stock_zh_a_hist_min_em_df.head())