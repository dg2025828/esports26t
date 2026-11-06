
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="PUBG 시청자 예측",
    page_icon="📈",
    layout="wide"
)

st.title("📈 PUBG Twitch 시청자 수 예측")
st.write("Prophet 모델을 이용한 미래 시청자 수 예측")

# 데이터 업로드
uploaded_file = st.file_uploader(
    "Twitch 시청자 데이터 CSV 업로드",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("1. 데이터 확인")
    st.dataframe(df.head())

    # 컬럼 선택
    col1, col2 = st.columns(2)

    with col1:
        date_col = st.selectbox(
            "날짜 컬럼 선택",
            df.columns
        )

    with col2:
        viewer_col = st.selectbox(
            "시청자 수 컬럼 선택",
            df.columns
        )

    # Prophet 데이터 형식으로 변환
    data = df[[date_col, viewer_col]].copy()

    data.columns = ["ds", "y"]

    data["ds"] = pd.to_datetime(
        data["ds"], errors="coerce"
    )

    data["y"] = pd.to_numeric(
        data["y"], errors="coerce"
    )

    data = data.dropna()

    # 날짜별 평균 시청자 수
    data = (
        data.groupby("ds")["y"]
        .mean()
        .reset_index()
        .sort_values("ds")
    )

    if len(data) < 30:
        st.warning(
            "정확한 추세 분석을 위해 "
            "30일 이상의 데이터를 권장합니다."
        )

    if len(data) < 10:
        st.error("학습 데이터가 너무 적습니다.")
        st.stop()

    st.subheader("2. 시청자 데이터 분석")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data["ds"],
        y=data["y"],
        mode="lines",
        name="실제 시청자 수"
    ))

    fig.update_layout(
        title="PUBG Twitch 실제 시청자 추이",
        xaxis_title="날짜",
        yaxis_title="평균 시청자 수"
    )

    st.plotly_chart(fig, use_container_width=True)

    # 예측 기간 선택
    days = st.slider(
        "미래 예측 기간 (일)",
        min_value=7,
        max_value=90,
        value=30
    )

    if st.button("Prophet 모델 학습 및 예측"):

        with st.spinner("Prophet 모델 학습 중..."):

            # 80% 학습 / 20% 테스트
            split = int(len(data) * 0.8)

            train = data.iloc[:split].copy()
            test = data.iloc[split:].copy()

            # 모델 생성
            model = Prophet(
                weekly_seasonality=True,
                yearly_seasonality="auto",
                daily_seasonality=False,
                interval_width=0.95
            )

            # 학습
            model.fit(train)

            # 테스트 기간 예측
            test_future = test[["ds"]].copy()
            test_forecast = model.predict(test_future)

            actual = test["y"].values
            predicted = test_forecast["yhat"].values

            mae = mean_absolute_error(
                actual, predicted
            )

            rmse = np.sqrt(
                mean_squared_error(actual, predicted)
            )

            # 전체 데이터로 재학습
            final_model = Prophet(
                weekly_seasonality=True,
                yearly_seasonality="auto",
                daily_seasonality=False,
                interval_width=0.95
            )

            final_model.fit(data)

            # 미래 예측
            future = final_model.make_future_dataframe(
                periods=days,
                freq="D"
            )

            forecast = final_model.predict(future)

            st.success("예측 완료!")

        # 정확도
        st.subheader("3. 모델 예측 정확도")

        c1, c2 = st.columns(2)

        c1.metric("MAE", f"{mae:,.2f}")
        c2.metric("RMSE", f"{rmse:,.2f}")

        st.caption(
            "MAE와 RMSE는 학습에 사용하지 않은 "
            "테스트 데이터에서 계산했습니다. "
            "값이 낮을수록 예측 오차가 작습니다."
        )

        # 예측 그래프
        st.subheader("4. 미래 시청자 수 예측")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=data["ds"],
            y=data["y"],
            mode="lines",
            name="실제 시청자 수"
        ))

        fig.add_trace(go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat"],
            mode="lines",
            name="예측 시청자 수"
        ))

        fig.add_trace(go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        ))

        fig.add_trace(go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_lower"],
            mode="lines",
            fill="tonexty",
            line=dict(width=0),
            name="95% 예측 구간"
        ))

        fig.update_layout(
            title="PUBG Twitch 미래 시청자 수 예측",
            xaxis_title="날짜",
            yaxis_title="평균 시청자 수",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # 미래 예측 데이터
        st.subheader("5. 미래 예측 데이터")

        future_result = forecast[
            forecast["ds"] > data["ds"].max()
        ][["ds", "yhat", "yhat_lower", "yhat_upper"]]

        future_result.columns = [
            "날짜",
            "예측 시청자 수",
            "최솟값(95% 구간)",
            "최댓값(95% 구간)"
        ]

        st.dataframe(
            future_result.round(2),
            use_container_width=True
        )

        csv = future_result.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            "예측 결과 CSV 다운로드",
            data=csv,
            file_name="PUBG_Twitch_예측.csv",
            mime="text/csv"
        )
