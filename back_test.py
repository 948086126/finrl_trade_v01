import matplotlib
import pandas as pd
import data_loader as dl
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

trade = pd.read_csv('./datasets/trade_data.csv') # 交易数据
train = pd.read_csv('./datasets/train_data.csv') # 训练数据

# 统一列名
trade = trade.rename(columns={
    "成交额": "turnover",
    "振幅": "amplitude",
    "涨跌幅": "pct_change",
    "涨跌额": "price_change",
    "换手率": "turnover_rate",
})
train = train.rename(columns={
    "成交额": "turnover",
    "振幅": "amplitude",
    "涨跌幅": "pct_change",
    "涨跌额": "price_change",
    "换手率": "turnover_rate",
})



train = train.set_index(train.columns[0])
train.index.names = ['']
trade = trade.set_index(trade.columns[0])
trade.index.names = ['']

if_using_a2c = True
if_using_ddpg = True
if_using_ppo = True
if_using_td3 = True
if_using_sac = True

trained_a2c = A2C.load(dl.Config.TRAINED_MODEL_DIR + "/agent_a2c") if if_using_a2c else None
trained_ddpg = DDPG.load(dl.Config.TRAINED_MODEL_DIR + "/agent_ddpg") if if_using_ddpg else None
trained_ppo = PPO.load(dl.Config.TRAINED_MODEL_DIR + "/agent_ppo") if if_using_ppo else None
trained_td3 = TD3.load(dl.Config.TRAINED_MODEL_DIR + "/agent_td3") if if_using_td3 else None
trained_sac = SAC.load(dl.Config.TRAINED_MODEL_DIR + "/agent_sac") if if_using_sac else None

stock_dimension = len(trade.tic.unique())
state_space = 1 + 2 * stock_dimension + len(dl.Config.TECHNICAL_INDICATORS) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": dl.Config.TECHNICAL_INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}

e_trade_gym = StockTradingEnv(df = trade, turbulence_threshold = 5,risk_indicator_col='amplitude', **env_kwargs)

df_account_value_a2c, df_actions_a2c = DRLAgent.DRL_prediction(
    model=trained_a2c,
    environment = e_trade_gym) if if_using_a2c else (None, None)

df_account_value_ddpg, df_actions_ddpg = DRLAgent.DRL_prediction(
    model=trained_ddpg,
    environment = e_trade_gym) if if_using_ddpg else (None, None)

df_account_value_ppo, df_actions_ppo = DRLAgent.DRL_prediction(
    model=trained_ppo,
    environment = e_trade_gym) if if_using_ppo else (None, None)

df_account_value_td3, df_actions_td3 = DRLAgent.DRL_prediction(
    model=trained_td3,
    environment = e_trade_gym) if if_using_td3 else (None, None)

df_account_value_sac, df_actions_sac = DRLAgent.DRL_prediction(
    model=trained_sac,
    environment = e_trade_gym) if if_using_sac else (None, None)

def process_df_for_mvo(df):
  return df.pivot(index="date", columns="tic", values="close")


def StockReturnsComputing(StockPrice, Rows, Columns):
    import numpy as np
    StockReturn = np.zeros([Rows - 1, Columns])
    for j in range(Columns):  # j: Assets
        for i in range(Rows - 1):  # i: Daily Prices
            StockReturn[i, j] = ((StockPrice[i + 1, j] - StockPrice[i, j]) / StockPrice[i, j]) * 100

    return StockReturn

StockData = process_df_for_mvo(train)
TradeData = process_df_for_mvo(trade)

TradeData.to_numpy()

arStockPrices = np.asarray(StockData)
[Rows, Cols] = arStockPrices.shape
arReturns = StockReturnsComputing(arStockPrices, Rows, Cols)

# compute mean returns and variance covariance matrix of returns
meanReturns = np.mean(arReturns, axis=0)
covReturns = np.cov(arReturns, rowvar=False)

# set precision for printing results
np.set_printoptions(precision=3, suppress=True)

# display mean returns and variance-covariance matrix of returns
print('Mean returns of assets in k-portfolio 1\n', meanReturns)
print('Variance-Covariance matrix of returns\n', covReturns)

# ef_mean = EfficientFrontier(meanReturns, covReturns, weight_bounds=(0, 0.5))
# raw_weights_mean = ef_mean.max_sharpe()
# cleaned_weights_mean = ef_mean.clean_weights()
# mvo_weights = np.array([1000000 * cleaned_weights_mean[i] for i in range(len(cleaned_weights_mean))])
#
# LastPrice = np.array([1/p for p in StockData.tail(1).to_numpy()[0]])
# Initial_Portfolio = np.multiply(mvo_weights, LastPrice)
#
# Portfolio_Assets = TradeData @ Initial_Portfolio
# MVO_result = pd.DataFrame(Portfolio_Assets, columns=["Mean Var"])

processor = dl.AkshareProcessor(config=dl.Config())

fulian = processor.download_data(
    ticker_list=dl.Config.TICKER_LIST,
    start_date=dl.Config.TRAIN_START_DATE,
    end_date=dl.Config.TRAIN_END_DATE,
)

df_fulian = fulian[["date", "close"]]
fst_day = fulian["close"][0]
dji = pd.merge(
    fulian["date"],
    fulian["close"].div(fst_day).mul(1000000),
    how="outer",
    left_index=True,
    right_index=True,
).set_index("date")

df_result_a2c = (
    df_account_value_a2c.set_index(df_account_value_a2c.columns[0])
    if if_using_a2c
    else None
)
df_result_ddpg = (
    df_account_value_ddpg.set_index(df_account_value_ddpg.columns[0])
    if if_using_ddpg
    else None
)
df_result_ppo = (
    df_account_value_ppo.set_index(df_account_value_ppo.columns[0])
    if if_using_ppo
    else None
)
df_result_td3 = (
    df_account_value_td3.set_index(df_account_value_td3.columns[0])
    if if_using_td3
    else None
)
df_result_sac = (
    df_account_value_sac.set_index(df_account_value_sac.columns[0])
    if if_using_sac
    else None
)

result = pd.DataFrame(
    {
        "a2c": df_result_a2c["account_value"] if if_using_a2c else None,
        "ddpg": df_result_ddpg["account_value"] if if_using_ddpg else None,
        "ppo": df_result_ppo["account_value"] if if_using_ppo else None,
        "td3": df_result_td3["account_value"] if if_using_td3 else None,
        "sac": df_result_sac["account_value"] if if_using_sac else None,
    }
)

# 删除全为NaN的列（未启用的模型）
result = result.dropna(axis=1, how='all')

# 打印数据验证
print("数据前5行:\n", result.head())
print("数据统计:\n", result.describe())
# 绘图设置
plt.rcParams["figure.figsize"] = (15, 5)
plt.figure()
result.plot(
    title="sh601138",
    xlabel="Date",
    ylabel="Account (USD)",
    grid=True
)
plt.tight_layout()
plt.show()