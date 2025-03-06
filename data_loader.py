import os
import numpy as np
import pandas as pd
import akshare as ak
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split


# 配置参数
class Config:
    TICKER_LIST = ["sh601138"]  # 示例股票代码
    #TICKER_LIST = ["sh601138","sz000938", "sh600050", "sz000977", "sh603259"]  # 示例股票代码
    TECHNICAL_INDICATORS = ['macd', 'rsi_30', 'boll_ub', 'boll_lb','close_5_sma', 'close_30_sma', 'volume_30_ma', 'rsi_14']
    TIME_INTERVAL = "5" # 5 分时线
    DATA_SAVE_DIR = "./datasets"
    TRAINED_MODEL_DIR = "./trained_models"
    RESULTS_DIR = "./results"
    INITIAL_AMOUNT = 1000000
    BUY_COST_PCT = [0.0015, 0.002, 0.001]  # 佣金+印花税
    SELL_COST_PCT = [0.0015, 0.001, 0.002]

    # 交易数据起止日期
    TRADE_START_DATE = "20240220"
    TRADE_END_DATE = "20250222"
    # 训练数据起止日期
    TRAIN_START_DATE = "2020-02-14 09:30:00"
    TRAIN_END_DATE = "2024-02-14 15:00:00"



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
                code = ticker[2:]  # 去除市场前缀，如sh→601138
                print(f"正在下载 {ticker}，代码参数:{code}...")

                # 调用AKShare接口
                df = ak.stock_zh_a_hist_min_em(
                    symbol=code,
                    period=self.config.TIME_INTERVAL,
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq"
                )

                # 检查空数据
                if df.empty:
                    print(f"⚠️ 股票 {ticker} 无数据，请检查代码或日期范围")
                    continue

                # 验证列名
                required_columns = {"时间", "开盘", "最高", "最低", "收盘", "成交量"}
                if not required_columns.issubset(df.columns):
                    missing = required_columns - set(df.columns)
                    print(f"⚠️ 股票 {ticker} 缺失列: {missing}")
                    continue

                # 重命名列
                df = df.rename(columns={
                    "时间": "date",
                    "开盘": "open",
                    "最高": "high",
                    "最低": "low",
                    "收盘": "close",
                    "成交量": "volume"
                })

                # 转换日期格式
                df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S")
                df["tic"] = ticker  # 保留原始ticker代码
                all_data.append(df)

                print(f"✅ 成功下载 {ticker}，数据量:{len(df)}")

            except Exception as e:
                print(f"❌ 股票 {ticker} 下载失败: {str(e)}")
                continue

        if not all_data:
            raise ValueError("所有股票下载失败，请检查参数")

        self.dataframe = pd.concat(all_data)
        return self.dataframe

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])
        df = df[df["volume"] > 0]
        df = df[df["close"] > 0]
        df = df.sort_values(["tic", "date"]).drop_duplicates()
        return df

    def add_technical_indicator(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.compute_technical_indicators(df=df, tech_list=self.config.TECHNICAL_INDICATORS)

        # 对每只股票的技术指标进行NaN处理
        df = df.groupby('tic').apply(
            lambda x: x.ffill().bfill()  # 先向前填充，再向后填充
        ).reset_index(drop=True)

        return df


    def compute_technical_indicators(self, df: pd.DataFrame, tech_list: list) -> pd.DataFrame:
        # 确保数据按股票和时间排序
        df = df.sort_values(['tic', 'date']).copy()
        # 分组对象（修复'grouped'错误）
        grouped = df.groupby('tic', group_keys=False)

        def calculate_rsi(series, window):
            delta = series.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)

            avg_gain = gain.rolling(window=window, min_periods=1).mean()
            avg_loss = loss.rolling(window=window, min_periods=1).mean()

            rs = avg_gain / (avg_loss + 1e-10)
            return 100 - (100 / (1 + rs))

        if "macd" in tech_list:
            # MACD计算（带最小周期保护）
            df['ema12'] = grouped['close'].transform(
                lambda x: x.ewm(span=12, min_periods=1, adjust=False).mean()
            )
            df['ema26'] = grouped['close'].transform(
                lambda x: x.ewm(span=26, min_periods=1, adjust=False).mean()
            )
            df['macd'] = df['ema12'] - df['ema26']

        if "rsi_30" in tech_list:
            df['rsi_30'] = grouped['close'].transform(lambda x: calculate_rsi(x, 30))
        if "boll_ub" in tech_list or "boll_lb" in tech_list:

            sma = df["close"].ewm(span=20, adjust=False).mean()
            std = df["close"].rolling(window=20, center=False).std()
            df["boll_ub"] = sma + 2 * std
            df["boll_lb"] = sma - 2 * std
            mean_boll_ub = df["boll_ub"].mean()
            median_boll_lb = df["boll_lb"].mean()

            df["boll_ub"].fillna(mean_boll_ub, inplace=True)
            df["boll_lb"].fillna(median_boll_lb, inplace=True)
        if "rsi_14" in tech_list:
            df["rsi_14"] = grouped['close'].transform(lambda x: calculate_rsi(x, 14))

        if "close_5_sma" in tech_list:
            df['close_5_sma'] = grouped['close'].transform(lambda x: x.rolling(5).mean())

        if "close_30_sma" in tech_list:
            df['close_30_sma'] = grouped['close'].transform(lambda x: x.rolling(30).mean())

        if "volume_30_ma" in tech_list:
            df['volume_30_ma'] = grouped['volume'].transform(lambda x: x.rolling(30).mean())
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
            end_date=Config.TRAIN_END_DATE,
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