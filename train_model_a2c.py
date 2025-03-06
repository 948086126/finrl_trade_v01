import os
import pandas as pd
from stable_baselines3.common.logger import configure
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.vec_env import DummyVecEnv


# 配置参数
class Config:
    TICKER_LIST = ["sh601138","sz000938", "sh600050", "sz000977", "sh603259"]  # 示例股票代码
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
    # 修改后的代码段
    stock_dim = len(train.tic.unique())

    # 验证实际可用的技术指标
    valid_indicators = [col for col in Config.TECHNICAL_INDICATORS
                        if col in train.columns]
    effective_tech_count = len(valid_indicators)

    # 计算正确的状态空间
    state_space = 1 + stock_dim + effective_tech_count * stock_dim
    print(f"Real State Space: {state_space} (Cash:1, Holdings:{stock_dim}, Tech:{effective_tech_count}×{stock_dim})")

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

    agent = DRLAgent(env=env_train)
    model_a2c = agent.get_model("a2c")

    # set up logger
    tmp_path = Config.RESULTS_DIR + '/a2c'
    new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_a2c.set_logger(new_logger_a2c)

    trained_a2c = agent.train_model(model=model_a2c,
                                    tb_log_name='a2c',
                                    total_timesteps=50000)
    trained_a2c.save(Config.TRAINED_MODEL_DIR + "/a2c") if if_using_sac else None


if __name__ == "__main__":
    main()