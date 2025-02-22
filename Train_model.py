import os
import pandas as pd
from stable_baselines3.common.logger import configure
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.vec_env import DummyVecEnv


# 配置参数
class Config:
    TICKER_LIST = ["sh601138"]  # 示例股票代码
    TECHNICAL_INDICATORS = ['macd', 'rsi_30', 'boll_ub', 'boll_lb']
    TIME_INTERVAL = "daily"
    DATA_SAVE_DIR = "./datasets"
    TRAINED_MODEL_DIR = "./trained_models"
    RESULTS_DIR = "./results"
    INITIAL_AMOUNT = 1000000
    BUY_COST_PCT = [0.000687]
    SELL_COST_PCT = [0.0010687]

    # 获取股票数据起止日
    STOCK_START_DATE = "20230101"
    STOCK_END_DATE = "20250219"

    TRAIN_START_DATE = "20230101"
    TRADE_END_DATE = "20250219"


# 主流程
def main():
    # 读取 CSV 文件
    processed_df = pd.read_csv("./datasets/dataset.csv")

    # 统一列名
    processed_df = processed_df.rename(columns={
        "成交额": "turnover",
        "振幅": "amplitude",
        "涨跌幅": "pct_change",
        "涨跌额": "price_change",
        "换手率": "turnover_rate",
    })

    processed_df["date"] = pd.to_datetime(processed_df["date"], format="%Y-%m-%d")
    processed_df.fillna(0, inplace=True)
    # 创建训练集时使用 datetime.date 对象进行比较
    train_start_date = pd.to_datetime(Config.TRAIN_START_DATE, format="%Y%m%d")
    trade_end_date = pd.to_datetime(Config.TRADE_END_DATE, format="%Y%m%d")


    train_df = processed_df[
        (processed_df["date"] >= train_start_date) &
        (processed_df["date"] <= trade_end_date)
        ]

    stock_dim = len(train_df["tic"].unique()) if not train_df.empty else 0

    if stock_dim == 0:
        print("训练数据为空，请检查日期范围和股票代码！")
        return

    env_kwargs = {
        "stock_dim": stock_dim,
        "hmax": 100,
        "initial_amount": Config.INITIAL_AMOUNT,
        "buy_cost_pct": Config.BUY_COST_PCT,
        "sell_cost_pct": Config.SELL_COST_PCT,
        "reward_scaling": 1e-4,
        "state_space": stock_dim * (len(Config.TECHNICAL_INDICATORS) + 2) + 1,
        "action_space": stock_dim,
        "tech_indicator_list": Config.TECHNICAL_INDICATORS,
        "num_stock_shares": [0] * stock_dim  # 初始股票持有数量
    }

    e_train = StockTradingEnv(df=train_df, **env_kwargs)
    env_train = DummyVecEnv([lambda: e_train])

    agent = DRLAgent(env=env_train)

    if_using_a2c = True  # 年化率 3.06%
    if_using_ddpg = True # 年化率 5.77%
    if_using_ppo = True
    if_using_td3 = True
    if_using_sac = True  # 年化率19%
    ################################# A2C #################################
    model_a2c = agent.get_model("a2c")
    if if_using_a2c:
        # set up logger
        tmp_path = Config.RESULTS_DIR + '/a2c'
        new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        # Set new logger
        model_a2c.set_logger(new_logger_a2c)

    trained_a2c = agent.train_model(model=model_a2c,
                                    tb_log_name='a2c',
                                    total_timesteps=50000) if if_using_a2c else None
    trained_a2c.save(Config.TRAINED_MODEL_DIR + "/agent_a2c") if if_using_a2c else None
    ################################# DDPG #################################
    model_ddpg = agent.get_model("ddpg")
    if if_using_ddpg:
        # set up logger
        tmp_path = Config.RESULTS_DIR + '/ddpg'
        new_logger_ddpg = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        # Set new logger
        model_ddpg.set_logger(new_logger_ddpg)

    trained_ddpg = agent.train_model(model=model_ddpg,
                                     tb_log_name='ddpg',
                                     total_timesteps=50000) if if_using_ddpg else None
    trained_ddpg.save(Config.TRAINED_MODEL_DIR + "/agent_ddpg") if if_using_ddpg else None

    ################################# PPO #################################
    PPO_PARAMS = {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": 64,
    }
    model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)

    if if_using_ppo:
        # set up logger
        tmp_path = Config.RESULTS_DIR + '/ppo'
        new_logger_ppo = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        # Set new logger
        model_ppo.set_logger(new_logger_ppo)

    trained_ppo = agent.train_model(
        model=model_ppo,
        tb_log_name="ppo",
        total_timesteps=100000
    )
    trained_ppo.save(os.path.join(Config.TRAINED_MODEL_DIR, "ppo_akshare"))
    ################################# TD3 #################################
    TD3_PARAMS = {"batch_size": 100,
                  "buffer_size": 1000000,
                  "learning_rate": 0.001}

    model_td3 = agent.get_model("td3", model_kwargs=TD3_PARAMS)

    if if_using_td3:
        # set up logger
        tmp_path = Config.RESULTS_DIR + '/td3'
        new_logger_td3 = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        # Set new logger
        model_td3.set_logger(new_logger_td3)

    trained_td3 = agent.train_model(model=model_td3,
                                    tb_log_name='td3',
                                    total_timesteps=50000) if if_using_td3 else None
    trained_td3.save(Config.TRAINED_MODEL_DIR + "/agent_td3") if if_using_td3 else None

    ################################# SAC #################################
    SAC_PARAMS = {
        "batch_size": 128,
        "buffer_size": 100000,
        "learning_rate": 0.0001,
        "learning_starts": 100,
        "ent_coef": "auto_0.1",
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
                             total_timesteps=70000) if if_using_sac else None
    trained_sac.save(Config.TRAINED_MODEL_DIR + "/agent_sac") if if_using_sac else None


if __name__ == "__main__":
    main()