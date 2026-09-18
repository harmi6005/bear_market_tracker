# -*- coding: utf-8 -*-
"""
하락장 심리매매법 - 자동추적 프로그램 (텔레그램 알림 버전)
================================================================

기능:
  1) 매일 장 마감 후 자동으로 종목 스크리닝 실행 (기본 15:40)
  2) 조건 충족 종목 발견 시 텔레그램으로 알림 전송
  3) `python bear_market_tracker.py --now` 로 언제든 수동 즉시 실행 가능

사전 준비 (텔레그램 봇 설정, 5분 소요):
  1. 텔레그램에서 @BotFather 검색 → /newbot 실행 → 봇 이름 설정
     완료되면 "BOT TOKEN"을 줍니다. (예: 123456789:AAExxxxxxxxxxxxxxxxxxxxx)
  2. 방금 만든 내 봇을 텔레그램에서 검색해서 아무 메시지나 보내기 (예: "안녕")
  3. 아래 URL을 브라우저에 입력해서 내 chat_id 확인:
     https://api.telegram.org/bot<BOT_TOKEN>/getUpdates
     응답 JSON에서 "chat":{"id": 123456789, ...} 부분의 숫자가 CHAT_ID
  4. 아래 CONFIG의 TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID에 채워넣기

실행 방법:
  - 수동 1회 실행:       python bear_market_tracker.py --now
  - 자동 스케줄러 시작:   python bear_market_tracker.py
                        (터미널을 계속 켜두거나, 서버/PC에서 백그라운드로 돌려야 함)
  - 완전 자동화하려면 맨 아래 "배포 참고" 섹션 참고 (cron / 작업 스케줄러)
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
import schedule

try:
    from pykrx import stock as krx
    _HAS_PYKRX = True
except ImportError:
    _HAS_PYKRX = False


# ======================================================================
# 0. 설정 (CONFIG) — 여기만 채우면 바로 동작합니다.
# ======================================================================
# 우선순위: 환경변수(GitHub Secrets 등) > 아래 직접 입력값
# GitHub에 올릴 파일이라면 절대 토큰을 직접 여기 적지 마세요! 환경변수/Secrets로만 넣으세요.
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "여기에_봇_토큰_입력")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "여기에_챗ID_입력")

DAILY_RUN_TIME = "15:40"   # 매일 이 시각(장 마감 후)에 자동 실행. 24시간제, "HH:MM"

# 스크리닝 대상 종목 (원하는 만큼 추가 가능. 전체 시장 스캔은 get_universe("ALL") 사용)
UNIVERSE = [
    "005930",  # 삼성전자
    "000660",  # SK하이닉스
    "035420",  # NAVER
    "035720",  # 카카오
    "003670",  # 포스코퓨처엠
    "051910",  # LG화학
    "373220",  # LG에너지솔루션
    "042700",  # 한미반도체
    "402340",  # SK스퀘어
    "034020",  # 두산에너빌리티
    "005380",  # 현대차
    "267260",  # HD현대일렉트릭
    "000270",  # 기아
]


# ======================================================================
# 1. 전략 파라미터 (지난번 스크리너와 동일한 로직)
# ======================================================================
@dataclass
class ScreenConfig:
    lookback_52w: int = 252
    min_prior_rally: float = 0.50
    min_drawdown: float = 0.30
    max_drawdown: float = 0.65
    consolidation_days: int = 7
    consolidation_range_pct: float = 0.08
    no_new_low_days: int = 5
    breakout_volume_multiple: float = 1.5
    breakout_lookback: int = 20


CFG = ScreenConfig()


# ======================================================================
# 2. 데이터 로딩 (KRX)
# ======================================================================
def get_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    if not _HAS_PYKRX:
        raise RuntimeError("pykrx가 설치되어 있지 않습니다. pip install pykrx")
    df = krx.get_market_ohlcv(start, end, ticker)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={
        "시가": "open", "고가": "high", "저가": "low",
        "종가": "close", "거래량": "volume",
    })
    df.index = pd.to_datetime(df.index)
    return df[["open", "high", "low", "close", "volume"]].sort_index()


def get_universe(market: str = "ALL") -> List[str]:
    if not _HAS_PYKRX:
        raise RuntimeError("pykrx가 설치되어 있지 않습니다.")
    today = datetime.today().strftime("%Y%m%d")
    tickers: List[str] = []
    if market in ("KOSPI", "ALL"):
        tickers += krx.get_market_ticker_list(today, market="KOSPI")
    if market in ("KOSDAQ", "ALL"):
        tickers += krx.get_market_ticker_list(today, market="KOSDAQ")
    return tickers


def get_name(ticker: str) -> str:
    if _HAS_PYKRX:
        try:
            return krx.get_market_ticker_name(ticker)
        except Exception:
            pass
    return ticker


# ======================================================================
# 3. 피처 계산 (주도주 / 조정폭 / 횡보 / 돌파)
# ======================================================================
def compute_features(df: pd.DataFrame, cfg: ScreenConfig) -> Optional[dict]:
    if len(df) < cfg.lookback_52w // 2:
        return None

    window = df.tail(cfg.lookback_52w).copy()
    close, high, low, volume = window["close"], window["high"], window["low"], window["volume"]

    week52_high = high.max()
    week52_high_date = high.idxmax()
    pre_rally_low = low.loc[:week52_high_date].min()
    if pre_rally_low <= 0 or pd.isna(pre_rally_low):
        return None
    prior_rally_pct = (week52_high / pre_rally_low) - 1

    last_close = close.iloc[-1]
    drawdown_pct = 1 - (last_close / week52_high)

    recent = window.tail(cfg.consolidation_days)
    recent_high = recent["high"].max()
    recent_low = recent["low"].min()
    range_pct = (recent_high - recent_low) / recent_low if recent_low > 0 else np.inf

    lows_before = low.iloc[: -cfg.no_new_low_days] if len(low) > cfg.no_new_low_days else low
    recent_min_low = low.tail(cfg.no_new_low_days).min()
    made_new_low_recently = len(lows_before) > 0 and recent_min_low <= lows_before.min()

    is_consolidating = (range_pct <= cfg.consolidation_range_pct) and (not made_new_low_recently)

    avg_volume = volume.tail(cfg.breakout_lookback).iloc[:-1].mean()
    today_volume = volume.iloc[-1]
    volume_ratio = today_volume / avg_volume if avg_volume > 0 else 0
    is_breakout = (
        last_close > recent.iloc[:-1]["high"].max()
        and volume_ratio >= cfg.breakout_volume_multiple
    )

    return {
        "last_close": last_close,
        "week52_high": week52_high,
        "prior_rally_pct": prior_rally_pct,
        "drawdown_pct": drawdown_pct,
        "consolidation_range_pct": range_pct,
        "is_consolidating": is_consolidating,
        "volume_ratio": volume_ratio,
        "is_breakout": is_breakout,
    }


def screen_universe(tickers: List[str], start: str, end: str, cfg: ScreenConfig,
                     require_breakout: bool = True, sleep_sec: float = 0.05) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        try:
            df = get_price_data(ticker, start, end)
            feat = compute_features(df, cfg)
        except Exception:
            feat = None
        if feat is None:
            continue
        if feat["prior_rally_pct"] < cfg.min_prior_rally:
            continue
        if not (cfg.min_drawdown <= feat["drawdown_pct"] <= cfg.max_drawdown):
            continue
        if not feat["is_consolidating"] and not feat["is_breakout"]:
            continue
        if require_breakout and not feat["is_breakout"]:
            continue

        rows.append({
            "ticker": ticker,
            "name": get_name(ticker),
            "close": feat["last_close"],
            "prior_rally_%": round(feat["prior_rally_pct"] * 100, 1),
            "drawdown_%": round(feat["drawdown_pct"] * 100, 1),
            "range_%": round(feat["consolidation_range_pct"] * 100, 1),
            "volume_ratio": round(feat["volume_ratio"], 2),
            "breakout_today": feat["is_breakout"],
        })
        if sleep_sec:
            time.sleep(sleep_sec)

    result = pd.DataFrame(rows)
    if not result.empty:
        result["score"] = result["drawdown_%"] * 0.5 + result["prior_rally_%"] * 0.5
        result = result.sort_values("score", ascending=False).reset_index(drop=True)
    return result


# ======================================================================
# 4. 텔레그램 알림
# ======================================================================
def send_telegram(message: str) -> None:
    if "여기에" in TELEGRAM_BOT_TOKEN or "여기에" in TELEGRAM_CHAT_ID:
        print("[알림 생략] 텔레그램 BOT_TOKEN / CHAT_ID가 설정되지 않았습니다.")
        print("--- 아래는 전송될 메시지 내용입니다 ---")
        print(message)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown",
        }, timeout=10)
        if resp.status_code != 200:
            print(f"[텔레그램 전송 실패] status={resp.status_code}, body={resp.text}")
    except Exception as e:
        print(f"[텔레그램 전송 오류] {e}")


def format_message(breakout_df: pd.DataFrame, watch_df: pd.DataFrame) -> str:
    today_str = datetime.today().strftime("%Y-%m-%d")
    lines = [f"*하락장 심리매매법 스캔 결과* ({today_str})", ""]

    if breakout_df.empty:
        lines.append("🚀 *오늘 돌파 신호 종목*: 없음")
    else:
        lines.append("🚀 *오늘 돌파 신호 종목 (매수 후보)*")
        for _, r in breakout_df.iterrows():
            lines.append(
                f"- {r['name']}({r['ticker']}) | 종가 {r['close']:,.0f} | "
                f"고점대비 -{r['drawdown_%']}% | 거래량 {r['volume_ratio']}배"
            )

    lines.append("")
    if watch_df.empty:
        lines.append("👀 *바닥 다지기 중(관심종목)*: 없음")
    else:
        lines.append("👀 *바닥 다지기 중 - 돌파 대기(관심종목)*")
        for _, r in watch_df.iterrows():
            lines.append(
                f"- {r['name']}({r['ticker']}) | 종가 {r['close']:,.0f} | "
                f"고점대비 -{r['drawdown_%']}% | 횡보폭 {r['range_%']}%"
            )

    lines.append("")
    lines.append("_주의: 투자 추천이 아니며 참고용 스크리닝 결과입니다._")
    return "\n".join(lines)


# ======================================================================
# 5. 실행 잡(Job): 스캔 → 텔레그램 전송
# ======================================================================
def run_job(universe: Optional[List[str]] = None) -> None:
    universe = universe or UNIVERSE
    end_date = datetime.today()
    start_date = end_date - timedelta(days=400)
    end_str = end_date.strftime("%Y%m%d")
    start_str = start_date.strftime("%Y%m%d")

    print(f"[{datetime.now()}] 스캔 시작 (대상 {len(universe)}종목)...")
    breakout_df = screen_universe(universe, start_str, end_str, CFG, require_breakout=True)
    watch_df = screen_universe(universe, start_str, end_str, CFG, require_breakout=False)
    # watch_df 에는 이미 돌파한 종목도 포함되므로, 아직 돌파 전인 종목만 남긴다
    if not watch_df.empty:
        watch_df = watch_df[~watch_df["breakout_today"]].reset_index(drop=True)

    message = format_message(breakout_df, watch_df)
    send_telegram(message)
    print(f"[{datetime.now()}] 스캔 완료. 돌파 {len(breakout_df)}건 / 관심 {len(watch_df)}건")


# ======================================================================
# 6. 실행부: --now(수동 즉시 실행) 또는 스케줄러(자동 매일 실행)
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="하락장 심리매매법 자동추적 프로그램")
    parser.add_argument("--now", action="store_true", help="스케줄 기다리지 않고 지금 바로 1회 실행")
    args = parser.parse_args()

    if args.now:
        run_job()
        return

    print(f"자동 스케줄러 시작. 매일 {DAILY_RUN_TIME}에 자동 스캔합니다. (Ctrl+C로 종료)")
    print("지금 바로 확인하고 싶으면 'python bear_market_tracker.py --now' 로 실행하세요.")
    schedule.every().day.at(DAILY_RUN_TIME).do(run_job)

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()


# ======================================================================
# 배포 참고: 컴퓨터를 계속 켜두지 않고 완전 자동화하고 싶다면
# ======================================================================
# [방법 A] 리눅스/맥 - crontab (매일 15:40 실행)
#   crontab -e 로 편집기를 열고 아래 줄 추가:
#   40 15 * * 1-5 /usr/bin/python3 /경로/bear_market_tracker.py --now >> /경로/log.txt 2>&1
#
# [방법 B] 윈도우 - 작업 스케줄러(Task Scheduler)
#   "기본 작업 만들기" → 트리거: 매일 15:40 → 동작: 프로그램 시작
#   프로그램: python.exe, 인수: bear_market_tracker.py --now
#
# [방법 C] 클라우드 서버(예: 저렴한 VPS)에 이 파일 그대로 올리고
#   `python bear_market_tracker.py` (스케줄러 모드)를 nohup / systemd 서비스로 상시 실행
