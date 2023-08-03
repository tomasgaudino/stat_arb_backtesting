import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas_ta as ta # noqa

st.set_page_config(page_title="Statistical Arbitrage", page_icon=":bar_chart:", layout="wide", initial_sidebar_state="collapsed")


def pnl_over_time(positions: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(name="PnL Over Time",
                             x=positions.position_id,
                             y=positions.net_cum_composed_pnl_usd.cumsum(),
                             mode='lines'))
    return fig


def net_pnl_usd(positions: pd.DataFrame):
    return positions.net_cum_composed_pnl_usd.sum()


def net_pnl_usd_pct(positions: pd.DataFrame, initial_portfolio: float = 150.0):
    return positions.net_cum_composed_pnl_usd.sum() / initial_portfolio


def accuracy(positions: pd.DataFrame):
    win_pos = len(positions[positions["exit_type"] == 'TP'])
    loss_pos = len(positions[positions["exit_type"] == 'SL'])
    return win_pos / (win_pos + loss_pos)


def n_positions(positions: pd.DataFrame):
    return len(positions)


def composed_pnl_backtesting(merged_df: pd.DataFrame,
                             arbitrage_sl: float = 0.002,
                             arbitrage_tp: float = 0.002,
                             fee: float = 0.0006):
    positions_matrix = []
    active_positions_pnl = []
    position_id = 0
    merged_df["exit_type"] = ""
    for datetime, row in merged_df.iterrows():
        if len(positions_matrix) < 1:
            signal = row["signal"]
            if signal in [1, -1]:
                positions_matrix.append([signal, -signal])
                active_positions_pnl.append([0, 0])
                position_id += 1
        if len(positions_matrix) > 0:
            for i in range(0, len(positions_matrix)):
                pnl_0 = row["ret"] * positions_matrix[i][0]
                pnl_1 = row["ret_2"] * positions_matrix[i][1]
                composed_pnl = pnl_0 + pnl_1

                active_positions_pnl[i][0] += pnl_0
                active_positions_pnl[i][1] += pnl_1

                cum_pnl_0 = active_positions_pnl[i][0]
                cum_pnl_1 = active_positions_pnl[i][1]
                cum_composed_pnl = cum_pnl_0 + cum_pnl_1

                # TODO: Add time limit
                if cum_composed_pnl < - arbitrage_sl:
                    cum_composed_pnl = - arbitrage_sl
                    del positions_matrix[i]
                    del active_positions_pnl[i]
                    merged_df.loc[datetime, "exit_type"] = "SL"
                if cum_composed_pnl > arbitrage_tp:
                    cum_composed_pnl = arbitrage_tp
                    del positions_matrix[i]
                    del active_positions_pnl[i]
                    merged_df.loc[datetime, "exit_type"] = "TP"

                merged_df.loc[datetime, "pnl_0"] = pnl_0
                merged_df.loc[datetime, "pnl_1"] = pnl_1

                # TODO: may be not store this values
                merged_df.loc[datetime, "cum_pnl_0"] = cum_pnl_0
                merged_df.loc[datetime, "cum_pnl_1"] = cum_pnl_1
                merged_df.loc[datetime, "composed_pnl"] = composed_pnl
                merged_df.loc[datetime, "cum_composed_pnl"] = cum_composed_pnl
                merged_df.loc[datetime, "position_id"] = position_id

    merged_df["net_cum_composed_pnl"] = merged_df["cum_composed_pnl"] - fee
    merged_df["net_cum_composed_pnl_usd"] = merged_df["net_cum_composed_pnl"] * order_amount_usd
    return merged_df


def spread_over_time(candles_df: pd.DataFrame, hedge: float):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
    fig.add_trace(go.Scatter(name="close_1", x=candles_df.index, y=candles_df.close),
                  secondary_y=False,
                  row=1, col=1)
    fig.add_trace(go.Scatter(name="close_2", x=candles_df.index, y=candles_df.close_2 * hedge),
                  secondary_y=False,
                  row=1, col=1)
    fig.add_trace(go.Bar(name="spread", x=candles_df.index, y=candles_df.spread, marker=dict(color="aqua", opacity=0.5)),
                  secondary_y=True,
                  row=1, col=1)
    fig.add_trace(go.Scatter(name="z-score", x=candles_df.index, y=candles_df.z_score),
                  secondary_y=False,
                  row=2, col=1)
    return fig


def get_candlesticks(candles_1: pd.DataFrame, candles_2: pd.DataFrame):
    fig = make_subplots(rows=2,
                        cols=1,
                        shared_xaxes=True)
    fig.add_trace(go.Candlestick(name="Candles 1",
                                 x=candles_1.datetime,
                                 open=candles_1.open,
                                 high=candles_1.high,
                                 low=candles_1.low,
                                 close=candles_1.close),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=candles_1.loc[(candles_1['side'] != 0), 'datetime'],
                             y=candles_1.loc[candles_1['side'] != 0, 'close'],
                             name='Entry Price: $',
                             mode='markers',
                             marker_symbol=candles_1.loc[(candles_1['side'] != 0), 'symbol'],
                             marker_size=20,
                             marker_line={'color': 'black', 'width': 0.7}))

    fig.add_trace(go.Candlestick(name="Candles 2",
                                 x=candles_2.datetime,
                                 open=candles_2.open,
                                 high=candles_2.high,
                                 low=candles_2.low,
                                 close=candles_2.close),
                  row=2, col=1)

    fig.add_trace(go.Scatter(x=candles_2.loc[(candles_2['side'] != 0), 'datetime'],
                             y=candles_2.loc[candles_2['side'] != 0, 'close'],
                             name='Entry Price: $',
                             mode='markers',
                             marker_symbol=candles_2.loc[(candles_2['side'] != 0), 'symbol'],
                             marker_size=20,
                             marker_line={'color': 'black', 'width': 0.7}),
                  row=2, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, xaxis2_rangeslider_visible=False)
    return fig


def get_signal(merged_df: pd.DataFrame, candles_1: pd.DataFrame, candles_2: pd.DataFrame):
    long_cond = merged_df["z_score"] < zscore_long_threshold
    short_cond = merged_df["z_score"] > zscore_short_threshold

    merged_df["signal"] = 0
    merged_df.loc[long_cond, "signal"] = 1
    merged_df.loc[short_cond, "signal"] = -1

    df_1 = merged_df[["datetime", "signal"]].merge(candles_1, on="datetime", how="left")
    df_2 = merged_df[["datetime", "signal"]].merge(candles_2, on="datetime", how="left")

    df_1["side"] = 0
    df_2["side"] = 0
    df_1.loc[df_1["signal"] == 1, "side"] = 1
    df_2.loc[df_2["signal"] == 1, "side"] = -1
    df_1.loc[df_1["signal"] == -1, "side"] = -1
    df_2.loc[df_2["signal"] == -1, "side"] = 1

    df_1.loc[df_1['side'] == -1, 'symbol'] = 'triangle-down'
    df_2.loc[df_2['side'] == -1, 'symbol'] = 'triangle-down'
    df_1.loc[df_1['side'] == 1, 'symbol'] = 'triangle-up'
    df_2.loc[df_2['side'] == 1, 'symbol'] = 'triangle-up'
    return df_1, df_2


def get_candles(trading_pairs: [], interval: str):
    data_list = []
    for trading_pair in trading_pairs:
        data = pd.read_csv(f"candles/candles_{trading_pair}_{interval}.csv")
        data["datetime"] = pd.to_datetime(data.timestamp, unit="ms")
        data.set_index("datetime", inplace=True)
        data["ret"] = data.close.pct_change()
        data_list.append(data)
    return data_list


st.title("Statistical Arbitrage Analysis")

length = st.number_input("Length", value=100, min_value=50, max_value=200, step=10)
zscore_long_threshold = st.number_input("Z-Score Long Threshold", value=-1.5, min_value=-10.0, max_value=0.0, step=0.1)
zscore_short_threshold = st.number_input("Z-Score Short Threshold", value=1.5, min_value=0.0, max_value=10.0, step=0.1)
arbitrage_take_profit = st.number_input("Arbitrage Take Profit %", value=1.0, min_value=0.01, max_value=100.0, step=0.01)
arbitrage_stop_loss = st.number_input("Arbitrage Stop Loss %", value=1.0, min_value=0.01, max_value=100.0, step=0.01)
order_amount_usd = st.number_input("Order Amount (USD)", value=10.0, min_value=10.0, max_value=100000.0, step=10.0)
initial_portfolio = st.number_input("Initial Portfolio (USD)", value=100.0, min_value=10.0, max_value=100000.0, step=1.0)

if st.button("Run backtesting"):
    candles = get_candles(["ETH-BUSD", "BTC-BUSD"], "5m")
    df = pd.merge(candles[0], candles[1], on="timestamp", how='inner', suffixes=('', '_2'))
    df["datetime"] = df.index = pd.to_datetime(df["timestamp"], unit="ms")
    hedge_ratio = df["close"].tail(length).mean() / df["close_2"].tail(length).mean()
    df["spread"] = df["close"] - (df["close_2"] * hedge_ratio)
    df["z_score"] = ta.zscore(df["spread"], length=length)

    candles[0], candles[1] = get_signal(merged_df=df,
                                        candles_1=candles[0],
                                        candles_2=candles[1])

    df = composed_pnl_backtesting(merged_df=df,
                                  arbitrage_sl=arbitrage_stop_loss / 100,
                                  arbitrage_tp=arbitrage_take_profit / 100)
    closed_positions = df.groupby(["position_id"]).last().reset_index()

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Ret USD",
                  f"${net_pnl_usd(closed_positions):.2f}",
                  f"{100 * net_pnl_usd_pct(closed_positions, initial_portfolio=initial_portfolio):.2f}%")
    with col2:
        st.metric("N° Positions",
                  n_positions(closed_positions))
    with col3:
        st.metric("Accuracy",
                  f"{100 * accuracy(closed_positions):.2f}%")
    with col4:
        pass
    with col5:
        pass
    with col6:
        pass
    st.plotly_chart(pnl_over_time(positions=closed_positions), use_container_width=True)

    st.subheader("Spread analysis")
    st.plotly_chart(spread_over_time(df, hedge_ratio), use_container_width=True)
    st.subheader("Candlestick")
    st.plotly_chart(get_candlesticks(candles[0], candles[1]), use_container_width=True)
    st.subheader("Tables")
    with st.expander("Merged df"):
        st.dataframe(df, use_container_width=True)
    with st.expander("Closed positions"):
        st.dataframe(df, use_container_width=True)
