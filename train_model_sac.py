import os
import pandas as pd
from stable_baselines3.common.logger import configure
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.vec_env import DummyVecEnv


# 配置参数
class Config:
    TICKER_LIST = ["sh601138"]  # 示例股票代码
    # TICKER_LIST = ["sh601138", "sz000938", "sh600050", "sz000977", "sh603259",
    #                "sh603019", "sh600415", "sz002230", "sh603501", "sh601336",
    #                "sz002475", "sz300124", "sh601766", "sh601628", "sz000002",
    #                "sz002304", "sz002241", "sh601888", "sz002466", "sz000725",
    #                "sh601633", "sh601012", "sh600150", "sh601989", "sz000100",
    #                "sh601919", "sh600111", "sh600010", "sh601939", "sh601398"]  # 示例股票代码
    TECHNICAL_INDICATORS = ['macd', 'rsi_30', 'boll_ub', 'boll_lb', 'close_5_sma', 'close_30_sma', 'volume_30_ma',
                            'rsi_14']
    TIME_INTERVAL = "daily"
    DATA_SAVE_DIR = "./datasets"
    TRAINED_MODEL_DIR = "./trained_models"
    RESULTS_DIR = "./results"
    INITIAL_AMOUNT = 1000000
    BUY_COST_PCT = [0.0015, 0.002, 0.001]  # 佣金+印花税
    SELL_COST_PCT = [0.0015, 0.001, 0.002]

    # 获取股票数据起止日
    STOCK_START_DATE = "20230101"
    STOCK_END_DATE = "20250219"

    TRAIN_START_DATE = "20130121"
    TRAIN_END_DATE = "20240219"


# 主流程
def main():
    # 读取 CSV 文件
    train = pd.read_csv("./datasets/train_data.csv")

    train = train.set_index(train.columns[0])
    train.index.names = ['']

    # 统一列名
    train = train.rename(columns={
        "成交额": "turnover",
        "振幅": "amplitude",
        "涨跌幅": "pct_change",
        "涨跌额": "price_change",
        "换手率": "turnover_rate",
    })

    train["date"] = pd.to_datetime(train["date"], format="%Y-%m-%d")
    train.fillna(0, inplace=True)
    # 创建训练集时使用 datetime.date 对象进行比较
    train_start_date = pd.to_datetime(Config.TRAIN_START_DATE, format="%Y%m%d")
    train_end_date = pd.to_datetime(Config.TRAIN_END_DATE, format="%Y%m%d")


    train = train[
        (train["date"] >= train_start_date) &
        (train["date"] <= train_end_date)
        ]
    stock_dim = len(train.tic.unique())
    # stock_dim = len(train_df["tic"].unique()) if not train_df.empty else 0
    state_space = 1 + 2 * stock_dim + len(Config.TECHNICAL_INDICATORS) * stock_dim
    print(f"Stock Dimension: {stock_dim}, State Space: {state_space}")

    buy_cost_list = sell_cost_list = [0.001] * stock_dim

    if stock_dim == 0:
        print("训练数据为空，请检查日期范围和股票代码！")
        return
    if_using_sac = True
    env_kwargs = {
        "stock_dim": stock_dim,
        "hmax": 10000,
        "initial_amount": Config.INITIAL_AMOUNT,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "reward_scaling": 1e-4,
        "state_space": state_space,
        "action_space": stock_dim,
        "tech_indicator_list": Config.TECHNICAL_INDICATORS,
        "num_stock_shares": [0] * stock_dim  # 初始股票持有数量
    }
    # 确保 close 列是 pandas.Series
    #train_df['close'] = pd.Series(train_df['close'])

    # 处理缺失值
    #train_df['close'].fillna(method='ffill', inplace=True)

    # env_train = DummyVecEnv([lambda: StockTradingEnv(df=train_df, **env_kwargs)])
    env_train = StockTradingEnv(df=train, **env_kwargs)

    env_train, _ = env_train.get_sb_env()
    print(type(env_train))
    agent = DRLAgent(env=env_train)
    ################################# SAC #################################
    SAC_PARAMS = {
        "batch_size": 512,
        "buffer_size": 100000,
        "learning_rate": 1e-5,
        "learning_starts": 100,
        "ent_coef": 0.2,
    }

    model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)

    if if_using_sac:
        # set up logger
        tmp_path = Config.RESULTS_DIR + '/sac'
        new_logger_sac = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        # Set new logger
        model_sac.set_logger(new_logger_sac)

    trained_sac = agent.train_model(model=model_sac,
                             tb_log_name='sac',
                             total_timesteps=5000) if if_using_sac else None
    trained_sac.save(Config.TRAINED_MODEL_DIR + "/agent_sac") if if_using_sac else None


if __name__ == "__main__":
    main()