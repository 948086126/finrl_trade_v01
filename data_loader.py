import os
import numpy as np
import pandas as pd
import akshare as ak
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split


# 配置参数
class Config:
    TICKER_LIST = ["sh601138"]  # 示例股票代码
    TECHNICAL_INDICATORS = ['macd', 'rsi_30', 'boll_ub', 'boll_lb']
    TIME_INTERVAL = "daily"
    DATA_SAVE_DIR = "./datasets"
    TRAINED_MODEL_DIR = "./trained_models"
    RESULTS_DIR = "./results"
    INITIAL_AMOUNT = 1000000
    BUY_COST_PCT = 0.000687
    SELL_COST_PCT = 0.0010687

    # 交易数据起止日期
    TRADE_START_DATE = "20240220"
    TRADE_END_DATE = "20250222"
    # 训练数据起止日期
    TRAIN_START_DATE = "20230121"
    TRAIN_END_DATE = "20240219"



# 数据处理器类
class AkshareProcessor:
    """AKShare数据处理器"""

    def __init__(self, config: Config):
        self.config = config
        self.dataframe = pd.DataFrame()

    def download_data(self, ticker_list: list, start_date: str, end_date: str) -> pd.DataFrame:
        all_data = []
        for ticker in ticker_list:
            try:
                code = ticker[2:]
                df = ak.stock_zh_a_hist(
                    symbol=code,
                    period=self.config.TIME_INTERVAL,
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq"
                )

                # 检查必要列是否存在
                required_columns = {"日期", "开盘", "最高", "最低", "收盘", "成交量"}
                if not required_columns.issubset(df.columns):
                    raise ValueError(f"股票 {ticker} 数据缺失必要列")

                df = df.rename(columns={
                    "日期": "date",
                    "开盘": "open",
                    "最高": "high",
                    "最低": "low",
                    "收盘": "close",
                    "成交量": "volume"
                })
                df["tic"] = ticker
                df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")  # 明确转换为 datetime 类型
                all_data.append(df)
            except Exception as e:
                print(f"股票 {ticker} 下载失败: {str(e)}")
                continue

        if not all_data:
            raise ValueError("未获取到有效数据")
        self.dataframe = pd.concat(all_data)
        return self.dataframe

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])
        df = df[df["volume"] > 0]
        df = df[df["close"] > 0]
        df = df.sort_values(["tic", "date"]).drop_duplicates()
        return df

    def add_technical_indicator(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.compute_technical_indicators(
            df=df,  # ✔️ 仅保留 tech_list 显式传递
            tech_list=self.config.TECHNICAL_INDICATORS
        )

    def compute_technical_indicators(self, df: pd.DataFrame, tech_list: list) -> pd.DataFrame:
        if "macd" in tech_list:
            df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
            df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
            df["macd"] = df["ema12"] - df["ema26"]

        if "rsi_30" in tech_list:
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.ewm(span=14, adjust=False).mean()
            avg_loss = loss.ewm(span=14, adjust=False).mean()

            rs = avg_gain / avg_loss
            df["rsi_30"] = np.where(rs != 0, 100 - (100 / (1 + rs)), 50)

            df["rsi_30"].fillna(50, inplace=True)

        if "boll_ub" in tech_list or "boll_lb" in tech_list:

            sma = df["close"].ewm(span=20, adjust=False).mean()
            std = df["close"].rolling(window=20, center=False).std()
            df["boll_ub"] = sma + 2 * std
            df["boll_lb"] = sma - 2 * std
            mean_boll_ub = df["boll_ub"].mean()
            median_boll_lb = df["boll_lb"].mean()

            df["boll_ub"].fillna(mean_boll_ub, inplace=True)
            df["boll_lb"].fillna(median_boll_lb, inplace=True)
        return df


# 主流程
def main():
    os.makedirs(Config.DATA_SAVE_DIR, exist_ok=True)
    os.makedirs(Config.TRAINED_MODEL_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    processor = AkshareProcessor(config=Config())

    try:
        print("开始下载数据...")
        raw_df = processor.download_data(
            ticker_list=Config.TICKER_LIST,
            start_date=Config.TRAIN_START_DATE,
            end_date=Config.TRADE_END_DATE,
        )
        print("数据清洗...")
        cleaned_df = processor.clean_data(raw_df)
        print("添加技术指标...")
        processed_df = processor.add_technical_indicator(cleaned_df)

        train = data_split(processed_df, Config.TRAIN_START_DATE, Config.TRAIN_END_DATE)
        trade = data_split(processed_df, Config.TRADE_START_DATE, Config.TRADE_END_DATE)

        train.to_csv(os.path.join(Config.DATA_SAVE_DIR, "train_data.csv"))
        trade.to_csv(os.path.join(Config.DATA_SAVE_DIR, "trade_data.csv"))
        print(f"数据集已保存至 {Config.DATA_SAVE_DIR}")
    except Exception as e:
        print(f"数据处理失败: {str(e)}")
        return
if __name__ == "__main__":
    main()