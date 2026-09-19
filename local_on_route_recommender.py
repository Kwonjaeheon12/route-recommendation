# -*- coding: utf-8 -*-
"""
LOCAL:ON 실제 데이터 기반 경로 추천 모델
========================================

현재 사용하는 원본 파일 4종
1. 03_로컬발견가능성_지역별.csv
2. 08_관광지ID_음식점매칭_최종_카카오좌표보완_최종.csv
3. 01_인기관광지_전국통합_점수_최종.csv
4. 01_인기관광지_전국통합_위경도_최최종.csv

추천 흐름
---------
사용자 시도 선택
→ 해당 시도 안 시군구를 로컬발견가능성 순위로 추천
→ 사용자가 선택한 시군구(미선택 시 1위 지역)
→ 맛집 / 관광지 / 카페·베이커리 선택
→ 기존 순위 데이터로 장소 후보 선정
→ 위경도 + 이동시간 + 시간대/카테고리 흐름을 고려해 당일치기/1박2일 경로 생성

원본 파일은 읽기만 하며 절대 수정하지 않는다.
"""

from __future__ import annotations

import os
import re
import json
import math
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:
    requests = None


# ============================================================
# 0. 파일 경로
# ============================================================

BASE_DIR = Path(r"C:\Users\User\Desktop\LOCAL_ON")

REGION_SCORE_PATH = BASE_DIR / "03_로컬발견가능성_지역별.csv"
FOOD_PATH = BASE_DIR / "08_관광지ID_음식점매칭_최종_카카오좌표보완_최종.csv"
TOUR_SCORE_PATH = BASE_DIR / "01_인기관광지_전국통합_점수_최종.csv"
TOUR_GEO_PATH = BASE_DIR / "01_인기관광지_전국통합_위경도_최최종.csv"

OUTPUT_DIR = BASE_DIR / "경로추천_결과"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# PowerShell:
# $env:KAKAO_REST_API_KEY="REST_API_KEY"
KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY", "").strip()


# ============================================================
# 1. 모델 설정
# ============================================================

VALID_CATEGORIES = {"맛집", "관광지", "카페/베이커리"}

TRIP_CONFIG = {
    "2h": {
        "budget_min": 120,
        "target_stops": 3,
        "max_stops": 3,
    },
    "4h": {
        "budget_min": 240,
        "target_stops": 4,
        "max_stops": 5,
    },
    "6h": {
        "budget_min": 360,
        "target_stops": 5,
        "max_stops": 6,
    },
    "day": {
        "budget_min": 480,
        "target_stops": 6,
        "max_stops": 7,
    },
}

# 장소별 기본 체류시간
STAY_MINUTES = {
    "맛집": 70,
    "관광지": 70,
    "카페/베이커리": 45,
}


# ============================================================
# 일정 자연스러움 설정
# ============================================================
# 기본 여행 시작 시각. CLI --start-time 으로 변경 가능.
DEFAULT_START_TIME = "10:00"

# 식사/카페를 실제 여행 시간대에 가깝게 배치하기 위한 권장 구간
LUNCH_WINDOW = (11 * 60 + 30, 14 * 60)       # 11:30 ~ 14:00
DINNER_WINDOW = (17 * 60 + 30, 20 * 60)      # 17:30 ~ 20:00
CAFE_WINDOW = (13 * 60, 17 * 60 + 30)        # 13:00 ~ 17:30

# 여러 카테고리를 선택했을 때 한 종류가 경로를 과점하지 않도록 제한.
# 단, 사용자가 한 카테고리만 선택한 경우에는 적용하지 않는다.
MULTI_CATEGORY_MAX_VISITS = {
    "맛집": 1,
    "카페/베이커리": 1,
    "관광지": 99,
}


# 동일 관광권역으로 판단할 거리 기준(km)
NEARBY_TOUR_CLUSTER_KM = 0.5

# 동일 세부분류가 반복될 때 감점
SAME_TOUR_SUBCATEGORY_PENALTY = 10.0

# 경로 최적화 전에 각 카테고리에서 몇 개까지 후보로 남길지
TOP_CANDIDATES_PER_CATEGORY = 10

# Beam Search 폭
BEAM_WIDTH = 150

# 관광지에서 기본적으로 제외할 분류
# 필요하면 삭제/추가 가능
EXCLUDED_TOUR_CATEGORIES = {
    "쇼핑몰",
    "백화점",
    "기타쇼핑시설",
    "면세점",
    "전문매장/상가",
    "호스텔",
}


# ============================================================
# 2. 공통 함수
# ============================================================

def read_csv_safely(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"파일이 없습니다: {path}")

    last_error = None

    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_error = e

    raise RuntimeError(f"CSV 읽기 실패: {path}\n{last_error}")


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0088

    lat1 = math.radians(float(lat1))
    lon1 = math.radians(float(lon1))
    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * r * math.asin(math.sqrt(a))


def normalize_100(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()

    if valid.empty:
        return pd.Series(np.nan, index=s.index, dtype=float)

    lo = valid.min()
    hi = valid.max()

    if hi == lo:
        out = pd.Series(np.nan, index=s.index, dtype=float)
        out[s.notna()] = 50.0
        return out

    return (s - lo) / (hi - lo) * 100.0


def food_rank_to_scores(
    local_rank: pd.Series,
    outsider_rank: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    기존 순위 설계와 동일한 방향 사용.

    현지인:
      1위 100점 → 100위 50.5점 (0.5 간격)

    외지인:
      1위 50점 → 100위 0.5점 (0.5 간격)

    두 점수가 모두 있으면 평균,
    하나만 있으면 해당 점수 사용.
    """
    lr = pd.to_numeric(local_rank, errors="coerce")
    er = pd.to_numeric(outsider_rank, errors="coerce")

    local_score = 100.5 - 0.5 * lr
    outsider_score = 50.5 - 0.5 * er

    local_score = local_score.where(lr.between(1, 100))
    outsider_score = outsider_score.where(er.between(1, 100))

    combined = pd.concat(
        [local_score, outsider_score],
        axis=1
    ).mean(axis=1, skipna=True)

    combined[
        local_score.isna() & outsider_score.isna()
    ] = np.nan

    return local_score, outsider_score, combined


# ============================================================
# 3. 지역별 로컬발견가능성
# ============================================================

def load_region_scores() -> pd.DataFrame:
    df = read_csv_safely(REGION_SCORE_PATH).copy()

    required = {
        "시도",
        "시군구",
        "데이터개월수",
        "신뢰도",
        "로컬발견가능성",
        "단기임시점수",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"지역 점수 파일 필수 컬럼 누락: {sorted(missing)}"
        )

    if df.duplicated(["시도", "시군구"]).any():
        raise ValueError(
            "지역 점수 파일에 시도+시군구 중복이 존재합니다."
        )

    df["로컬발견가능성"] = to_numeric(
        df["로컬발견가능성"]
    )
    df["단기임시점수"] = to_numeric(
        df["단기임시점수"]
    )

    return df


def get_region_ranking(
    region_df: pd.DataFrame,
    selected_sido: str,
) -> pd.DataFrame:
    """
    같은 시도 안에서 지역 추천 순위를 만든다.

    12개월 로컬발견가능성 점수와 2개월 단기임시점수는
    직접 같은 척도로 섞지 않는다.

    1) 12개월 정식 점수가 있는 지역 우선
    2) 나머지는 '참고용'으로 별도 순위
    """
    x = region_df[
        region_df["시도"] == selected_sido
    ].copy()

    if x.empty:
        available = sorted(region_df["시도"].unique())
        raise ValueError(
            f"'{selected_sido}' 데이터가 없습니다.\n"
            f"가능한 시도: {available}"
        )

    formal = x[
        x["로컬발견가능성"].notna()
    ].copy()

    formal = formal.sort_values(
        ["로컬발견가능성", "시군구"],
        ascending=[False, True],
    )

    formal["지역추천구분"] = "정식_12개월"
    formal["지역추천점수"] = formal["로컬발견가능성"]
    formal["시도내순위"] = np.arange(1, len(formal) + 1)

    short = x[
        x["로컬발견가능성"].isna()
    ].copy()

    short = short.sort_values(
        ["단기임시점수", "시군구"],
        ascending=[False, True],
        na_position="last",
    )

    short["지역추천구분"] = "참고_단기"
    short["지역추천점수"] = short["단기임시점수"]
    short["시도내순위"] = np.arange(1, len(short) + 1)

    result = pd.concat(
        [formal, short],
        ignore_index=True
    )

    cols = [
        "시도",
        "시군구",
        "데이터개월수",
        "신뢰도",
        "지역추천구분",
        "지역추천점수",
        "시도내순위",
        "로컬발견가능성",
        "단기임시점수",
    ]

    return result[cols]


# ============================================================
# 4. 맛집 / 카페·베이커리 데이터
# ============================================================

def classify_food(row: pd.Series) -> str:
    text = " ".join([
        str(row.get("업소명", "")),
        str(row.get("분류", "")),
        str(row.get("업태구분명", "")),
        str(row.get("음식점구분", "")),
    ]).lower()

    cafe_keywords = [
        "카페",
        "찻집",
        "커피",
        "coffee",
        "cafe",
        "베이커리",
        "제과",
        "제빵",
        "빵",
        "디저트",
    ]

    if any(k in text for k in cafe_keywords):
        return "카페/베이커리"

    return "맛집"


def load_food_places() -> pd.DataFrame:
    df = read_csv_safely(FOOD_PATH).copy()

    required = {
        "관광지ID",
        "업소명",
        "분류",
        "맛집시도",
        "맛집시군구",
        "현지인순위",
        "외지인순위",
        "경도",
        "위도",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"음식점 파일 필수 컬럼 누락: {sorted(missing)}"
        )

    df["위도"] = to_numeric(df["위도"])
    df["경도"] = to_numeric(df["경도"])

    local_score, outsider_score, combined = (
        food_rank_to_scores(
            df["현지인순위"],
            df["외지인순위"],
        )
    )

    df["현지인점수_계산"] = local_score
    df["외지인점수_계산"] = outsider_score
    df["장소추천점수"] = combined

    df["추천카테고리"] = df.apply(
        classify_food,
        axis=1
    )

    out = pd.DataFrame({
        "place_id": df["관광지ID"].astype(str),
        "장소명": df["업소명"].astype(str),
        "추천카테고리": df["추천카테고리"],
        "세부분류": df["분류"].astype(str),
        "시도": df["맛집시도"].astype(str),
        "시군구": df["맛집시군구"].astype(str),
        "현지인순위": to_numeric(df["현지인순위"]),
        "외지인순위": to_numeric(df["외지인순위"]),
        "현지인점수": local_score,
        "외지인점수": outsider_score,
        "장소추천점수": combined,
        "위도": df["위도"],
        "경도": df["경도"],
        "주소": df.get(
            "도로명주소",
            pd.Series("", index=df.index)
        ).fillna(""),
        "원본구분": "음식점",
    })

    # 좌표/지역/점수 사용 가능 행만
    out = out[
        out["위도"].between(32, 39.5)
        & out["경도"].between(124, 132.5)
        & out["장소추천점수"].notna()
        & out["시도"].ne("")
        & out["시군구"].ne("")
    ].copy()

    if out["place_id"].duplicated().any():
        raise ValueError(
            "음식점 데이터 관광지ID가 중복되어 있습니다."
        )

    return out.reset_index(drop=True)


# ============================================================
# 5. 관광지 점수 + 좌표 데이터
# ============================================================

SIDO_PREFIX_MAP = {
    "서울": "서울특별시",
    "부산": "부산광역시",
    "대구": "대구광역시",
    "인천": "인천광역시",
    "광주": "광주광역시",
    "대전": "대전광역시",
    "울산": "울산광역시",
    "세종": "세종특별자치시",
    "경기": "경기도",
    "강원": "강원특별자치도",
    "충북": "충청북도",
    "충남": "충청남도",
    "전북": "전북특별자치도",
    "전남": "전라남도",
    "경북": "경상북도",
    "경남": "경상남도",
    "제주": "제주특별자치도",
}

ADDRESS_SIDO_PREFIXES = [
    ("서울특별시", "서울특별시"),
    ("서울", "서울특별시"),
    ("부산광역시", "부산광역시"),
    ("부산", "부산광역시"),
    ("대구광역시", "대구광역시"),
    ("대구", "대구광역시"),
    ("인천광역시", "인천광역시"),
    ("인천", "인천광역시"),
    ("광주광역시", "광주광역시"),
    ("광주", "광주광역시"),
    ("대전광역시", "대전광역시"),
    ("대전", "대전광역시"),
    ("울산광역시", "울산광역시"),
    ("울산", "울산광역시"),
    ("세종특별자치시", "세종특별자치시"),
    ("세종", "세종특별자치시"),
    ("경기도", "경기도"),
    ("경기", "경기도"),
    ("강원특별자치도", "강원특별자치도"),
    ("강원도", "강원특별자치도"),
    ("강원", "강원특별자치도"),
    ("충청북도", "충청북도"),
    ("충북", "충청북도"),
    ("충청남도", "충청남도"),
    ("충남", "충청남도"),
    ("전북특별자치도", "전북특별자치도"),
    ("전라북도", "전북특별자치도"),
    ("전북", "전북특별자치도"),
    ("전라남도", "전라남도"),
    ("전남", "전라남도"),
    ("경상북도", "경상북도"),
    ("경북", "경상북도"),
    ("경상남도", "경상남도"),
    ("경남", "경상남도"),
    ("제주특별자치도", "제주특별자치도"),
    ("제주도", "제주특별자치도"),
    ("제주", "제주특별자치도"),
]


def extract_sigungu_from_source(filename: str) -> Optional[str]:
    """
    예:
    20260913003004_고양시+일산서구_202509-202608_...csv
    → 고양시+일산서구

    강원20260913024754_강릉시_202509-202608_...csv
    → 강릉시
    """
    text = str(filename)

    m = re.search(
        r"_([^_]+)_\d{6}-\d{6}_",
        text
    )

    if m:
        return m.group(1).strip()

    return None


def infer_sido_from_tour_row(
    source_file: str,
    address: str,
) -> Optional[str]:

    source = str(source_file)
    address = str(address)

    # 광주/대전/부산 등 파일 자체에 시도 prefix가 있을 때 가장 신뢰
    for prefix, standard in SIDO_PREFIX_MAP.items():
        if source.startswith(prefix):
            return standard

    # 주소에 '전남광주통합특별시'가 있더라도
    # source가 광주 prefix였다면 위에서 이미 광주로 처리됨.
    for prefix, standard in ADDRESS_SIDO_PREFIXES:
        if address.startswith(prefix):
            return standard

    return None


def load_tour_places(
    region_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    score = read_csv_safely(
        TOUR_SCORE_PATH
    ).copy()

    geo = read_csv_safely(
        TOUR_GEO_PATH
    ).copy()

    score_required = {
        "관광지ID",
        "관광지명",
        "분류",
        "현지인순위",
        "외지인순위",
        "현지인점수",
        "외지인점수",
        "추천점수",
        "전국추천순위",
        "원본파일",
    }

    geo_required = {
        "관광지ID",
        "관광지명",
        "위도",
        "경도",
        "주소",
    }

    if score_required - set(score.columns):
        raise ValueError(
            "관광지 점수 파일 필수 컬럼이 부족합니다."
        )

    if geo_required - set(geo.columns):
        raise ValueError(
            "관광지 좌표 파일 필수 컬럼이 부족합니다."
        )

    # --------------------------------------------------------
    # 좌표 파일 중복 ID 검증
    # 서로 다른 좌표를 가진 중복 ID는 임의 선택하지 않고 모델에서 제외
    # --------------------------------------------------------
    geo["위도"] = to_numeric(geo["위도"])
    geo["경도"] = to_numeric(geo["경도"])

    duplicate_ids = set(
        geo.loc[
            geo["관광지ID"].duplicated(keep=False),
            "관광지ID"
        ].astype(str)
    )

    qa_rows = []

    for pid in sorted(duplicate_ids):
        g = geo[
            geo["관광지ID"].astype(str) == pid
        ]

        unique_coords = g[
            ["위도", "경도"]
        ].drop_duplicates()

        if len(unique_coords) > 1:
            qa_rows.append({
                "관광지ID": pid,
                "문제": "동일ID_서로다른좌표",
                "행수": len(g),
            })

    bad_geo_ids = {
        r["관광지ID"]
        for r in qa_rows
    }

    geo_clean = geo[
        ~geo["관광지ID"].astype(str).isin(
            bad_geo_ids
        )
    ].drop_duplicates(
        subset=["관광지ID"],
        keep="first",
    )

    # --------------------------------------------------------
    # 점수 + 좌표를 메모리에서만 JOIN
    # 원본 CSV 파일 자체는 변경하지 않음
    # --------------------------------------------------------
    merged = score.merge(
        geo_clean[
            [
                "관광지ID",
                "위도",
                "경도",
                "주소",
            ]
        ],
        on="관광지ID",
        how="left",
        validate="one_to_one",
    )

    # 좌표가 없는 점수 데이터 QA
    missing_geo = merged[
        merged["위도"].isna()
        | merged["경도"].isna()
    ]

    for _, r in missing_geo.iterrows():
        qa_rows.append({
            "관광지ID": str(r["관광지ID"]),
            "문제": "좌표없음",
            "행수": 1,
        })

    # --------------------------------------------------------
    # 지역 추출
    # --------------------------------------------------------
    merged["시군구"] = (
        merged["원본파일"]
        .apply(extract_sigungu_from_source)
    )

    merged["시도"] = [
        infer_sido_from_tour_row(
            source,
            address,
        )
        for source, address in zip(
            merged["원본파일"],
            merged["주소"].fillna(""),
        )
    ]

    # 지역 점수 테이블을 이용해 시도 보완
    # 시군구 이름이 전국에서 유일할 때만 안전하게 보완
    sigungu_to_sidos = (
        region_df.groupby("시군구")["시도"]
        .agg(lambda x: list(pd.unique(x)))
        .to_dict()
    )

    for idx in merged.index[
        merged["시도"].isna()
    ]:
        sg = merged.at[idx, "시군구"]

        sidos = sigungu_to_sidos.get(
            sg,
            []
        )

        if len(sidos) == 1:
            merged.at[idx, "시도"] = sidos[0]

    # 지역키가 03 파일에 실제 존재하는지 검증
    valid_region_keys = set(
        zip(
            region_df["시도"].astype(str),
            region_df["시군구"].astype(str),
        )
    )

    valid_region_mask = [
        (str(sido), str(sigungu))
        in valid_region_keys
        for sido, sigungu in zip(
            merged["시도"],
            merged["시군구"],
        )
    ]

    bad_region = merged[
        ~pd.Series(
            valid_region_mask,
            index=merged.index
        )
    ]

    for _, r in bad_region.iterrows():
        qa_rows.append({
            "관광지ID": str(r["관광지ID"]),
            "문제": (
                f"지역매칭실패:"
                f"{r.get('시도','')} "
                f"{r.get('시군구','')}"
            ),
            "행수": 1,
        })

    merged["추천점수"] = to_numeric(
        merged["추천점수"]
    )

    merged["위도"] = to_numeric(
        merged["위도"]
    )
    merged["경도"] = to_numeric(
        merged["경도"]
    )

    usable = merged[
        pd.Series(
            valid_region_mask,
            index=merged.index
        )
        & merged["위도"].between(32, 39.5)
        & merged["경도"].between(124, 132.5)
        & merged["추천점수"].notna()
        & ~merged["분류"].isin(
            EXCLUDED_TOUR_CATEGORIES
        )
    ].copy()

    out = pd.DataFrame({
        "place_id": usable["관광지ID"].astype(str),
        "장소명": usable["관광지명"].astype(str),
        "추천카테고리": "관광지",
        "세부분류": usable["분류"].astype(str),
        "시도": usable["시도"].astype(str),
        "시군구": usable["시군구"].astype(str),
        "현지인순위": to_numeric(
            usable["현지인순위"]
        ),
        "외지인순위": to_numeric(
            usable["외지인순위"]
        ),
        "현지인점수": to_numeric(
            usable["현지인점수"]
        ),
        "외지인점수": to_numeric(
            usable["외지인점수"]
        ),
        "장소추천점수": usable["추천점수"],
        "위도": usable["위도"],
        "경도": usable["경도"],
        "주소": usable["주소"].fillna(""),
        "원본구분": "관광지",
    })

    qa = pd.DataFrame(
        qa_rows,
        columns=[
            "관광지ID",
            "문제",
            "행수",
        ]
    )

    return (
        out.reset_index(drop=True),
        qa
    )


# ============================================================
# 6. 선택 지역 장소 후보
# ============================================================

def get_place_candidates(
    all_places: pd.DataFrame,
    selected_sido: str,
    selected_sigungu: str,
    selected_categories: list[str],
) -> pd.DataFrame:

    invalid = set(selected_categories) - VALID_CATEGORIES

    if invalid:
        raise ValueError(
            f"지원하지 않는 카테고리: {sorted(invalid)}"
        )

    x = all_places[
        (all_places["시도"] == selected_sido)
        & (
            all_places["시군구"]
            == selected_sigungu
        )
        & all_places[
            "추천카테고리"
        ].isin(selected_categories)
    ].copy()

    if x.empty:
        raise ValueError(
            f"{selected_sido} {selected_sigungu}에서 "
            f"{selected_categories} 후보를 찾지 못했습니다."
        )

    # 카테고리 안에서 기존 장소추천점수 기준 순위
    x["카테고리내순위"] = (
        x.groupby("추천카테고리")[
            "장소추천점수"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    # 경로 탐색량 제한:
    # 각 카테고리 TOP N
    x = x[
        x["카테고리내순위"]
        <= TOP_CANDIDATES_PER_CATEGORY
    ].copy()

    return x.sort_values(
        [
            "추천카테고리",
            "카테고리내순위",
        ]
    ).reset_index(drop=True)



def infer_region_center(
    candidates: pd.DataFrame,
) -> tuple[float, float]:
    """
    출발지 미입력 시 선택 지역 후보 장소들의 중앙값 좌표를
    임시 지역 중심점으로 사용한다.
    평균보다 이상치에 덜 민감한 median 사용.
    """
    if candidates.empty:
        raise ValueError(
            "지역 중심점을 계산할 후보 장소가 없습니다."
        )

    lat = float(
        pd.to_numeric(
            candidates["위도"],
            errors="coerce",
        ).median()
    )

    lon = float(
        pd.to_numeric(
            candidates["경도"],
            errors="coerce",
        ).median()
    )

    if math.isnan(lat) or math.isnan(lon):
        raise ValueError(
            "지역 중심점 계산에 실패했습니다."
        )

    return lat, lon


def near_duplicate_tourist_stop(
    new_idx: int,
    route: list[int],
    candidates: pd.DataFrame,
    threshold_km: float = NEARBY_TOUR_CLUSTER_KM,
) -> bool:
    """
    이미 경로에 들어간 관광지와 500m 이내이고,
    새 장소도 관광지라면 같은 관광권역 중복으로 판단.
    """
    new_row = candidates.loc[new_idx]

    if new_row["추천카테고리"] != "관광지":
        return False

    for idx in route:
        old_row = candidates.loc[idx]

        if old_row["추천카테고리"] != "관광지":
            continue

        km = haversine_km(
            old_row["위도"],
            old_row["경도"],
            new_row["위도"],
            new_row["경도"],
        )

        if km <= threshold_km:
            return True

    return False


def tourist_subcategory_diversity_score(
    route: list[int],
    candidates: pd.DataFrame,
) -> float:
    """
    관광지 세부분류 다양성을 평가.
    같은 세부분류가 반복될수록 감점한다.
    """
    subs = []

    for idx in route:
        row = candidates.loc[idx]

        if row["추천카테고리"] == "관광지":
            subs.append(
                str(row["세부분류"])
            )

    if not subs:
        return 0.0

    repeats = len(subs) - len(set(subs))

    return -(
        repeats
        * SAME_TOUR_SUBCATEGORY_PENALTY
    )


# ============================================================
# 7. 이동시간
# ============================================================

class TravelTimeProvider:

    KAKAO_URL = (
        "https://apis-navi.kakaomobility.com"
        "/v1/directions"
    )

    def __init__(
        self,
        cache_file: Path,
        api_key: str = "",
    ):
        self.api_key = api_key
        self.cache_file = cache_file

        if cache_file.exists():
            try:
                self.cache = json.loads(
                    cache_file.read_text(
                        encoding="utf-8"
                    )
                )
            except Exception:
                self.cache = {}
        else:
            self.cache = {}

    @staticmethod
    def _key(
        lat1,
        lon1,
        lat2,
        lon2,
    ):
        return (
            f"{float(lat1):.6f},"
            f"{float(lon1):.6f}|"
            f"{float(lat2):.6f},"
            f"{float(lon2):.6f}"
        )

    def save(self):
        self.cache_file.write_text(
            json.dumps(
                self.cache,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _estimate(
        self,
        lat1,
        lon1,
        lat2,
        lon2,
    ):
        straight_km = haversine_km(
            lat1,
            lon1,
            lat2,
            lon2,
        )

        # 실제 도로는 직선거리보다 길다는 점을 반영
        road_km = straight_km * 1.25

        # 관광지/도심 혼합 평균속도 30 km/h
        # 주차·교차로 등을 고려한 3분 추가
        minutes = (
            road_km / 30.0 * 60.0
            + 3.0
        )

        return {
            "distance_km": round(
                road_km,
                3
            ),
            "minutes": round(
                max(2.0, minutes),
                2
            ),
            "source": "거리기반추정",
        }

    def get(
        self,
        lat1,
        lon1,
        lat2,
        lon2,
    ):
        key = self._key(
            lat1,
            lon1,
            lat2,
            lon2,
        )

        if key in self.cache:
            return self.cache[key]

        result = None

        if self.api_key and requests is not None:
            try:
                response = requests.get(
                    self.KAKAO_URL,
                    headers={
                        "Authorization":
                            f"KakaoAK {self.api_key}",
                        "Content-Type":
                            "application/json",
                    },
                    params={
                        "origin":
                            f"{lon1},{lat1}",
                        "destination":
                            f"{lon2},{lat2}",
                        "priority":
                            "RECOMMEND",
                        "summary":
                            "true",
                    },
                    timeout=8,
                )

                response.raise_for_status()
                payload = response.json()

                routes = payload.get(
                    "routes",
                    []
                )

                if routes:
                    summary = routes[0].get(
                        "summary",
                        {}
                    )

                    if (
                        "distance" in summary
                        and "duration" in summary
                    ):
                        result = {
                            "distance_km":
                                summary[
                                    "distance"
                                ] / 1000.0,
                            "minutes":
                                summary[
                                    "duration"
                                ] / 1000.0 / 60.0,
                            "source":
                                "Kakao자동차",
                        }

            except Exception:
                result = None

        if result is None:
            result = self._estimate(
                lat1,
                lon1,
                lat2,
                lon2,
            )

        self.cache[key] = result

        return result


# ============================================================
# 8. 경로 탐색
# ============================================================

def parse_hhmm(value: str) -> int:
    """HH:MM 문자열을 0시 기준 분으로 변환."""
    try:
        hh, mm = map(int, str(value).split(":"))
    except Exception as e:
        raise ValueError(
            f"시간 형식은 HH:MM 이어야 합니다: {value}"
        ) from e

    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise ValueError(
            f"올바르지 않은 시간입니다: {value}"
        )

    return hh * 60 + mm


def minutes_to_hhmm(minutes: float) -> str:
    """분 값을 HH:MM으로 표현. 날짜를 넘으면 +1일 표시."""
    total = int(round(minutes))
    day = total // (24 * 60)
    total = total % (24 * 60)
    hh = total // 60
    mm = total % 60

    if day:
        return f"+{day}일 {hh:02d}:{mm:02d}"

    return f"{hh:02d}:{mm:02d}"


def in_window(value: float, window: tuple[int, int]) -> bool:
    return window[0] <= value <= window[1]


def category_quota_bonus(
    route_categories: list[str],
    selected_categories: list[str],
) -> float:
    """
    사용자가 선택한 카테고리를 실제 경로가 얼마나 충족하는지 평가.
    카테고리 하나를 처음 포함할 때 보너스를 크게 부여한다.
    """
    covered = len(
        set(route_categories)
        & set(selected_categories)
    )

    return covered * 20.0


def get_category_limit(
    category: str,
    selected_categories: list[str],
) -> int:
    """
    여러 카테고리 선택 시:
    - 맛집 1곳
    - 카페/베이커리 1곳
    - 관광지는 나머지 일정 구성

    단, 사용자가 한 종류만 골랐다면 해당 제한을 적용하지 않는다.
    """
    if len(set(selected_categories)) <= 1:
        return 99

    return MULTI_CATEGORY_MAX_VISITS.get(
        category,
        99,
    )


def evaluate_time_fit(
    category: str,
    activity_start_minute: float,
) -> float:
    """
    해당 장소를 그 시각에 방문하는 것이 자연스러운지 점수화.
    장소 원래 추천점수는 건드리지 않고 '경로 순서' 평가에만 사용한다.
    """
    if category == "맛집":
        if in_window(activity_start_minute, LUNCH_WINDOW):
            return 28.0
        if in_window(activity_start_minute, DINNER_WINDOW):
            return 24.0

        # 오전 너무 이른 식당 방문은 강하게 감점
        if activity_start_minute < 11 * 60:
            return -32.0

        # 점심과 저녁 사이에는 약한 감점
        if 14 * 60 < activity_start_minute < 17 * 60 + 30:
            return -12.0

        return -8.0

    if category == "카페/베이커리":
        if in_window(activity_start_minute, CAFE_WINDOW):
            return 18.0

        if activity_start_minute < 11 * 60:
            return -12.0

        return 4.0

    if category == "관광지":
        # 오전~오후 관광을 기본적으로 선호
        if 9 * 60 <= activity_start_minute <= 17 * 60 + 30:
            return 8.0

        return -5.0

    return 0.0


def simulate_route(
    route: list[int],
    candidates: pd.DataFrame,
    provider: TravelTimeProvider,
    start_lat: Optional[float],
    start_lon: Optional[float],
    start_minute: int,
) -> tuple[list[dict], float, float, float]:
    """
    경로를 실제 시간 순서대로 시뮬레이션한다.

    반환:
    - 각 장소 일정
    - 장소추천점수 합
    - 총 이동시간(분)
    - 총 이동거리(km)
    """
    schedule = []
    current_minute = float(start_minute)
    prev_lat = start_lat
    prev_lon = start_lon

    total_place_score = 0.0
    total_travel_min = 0.0
    total_distance_km = 0.0

    for idx in route:
        row = candidates.loc[idx]

        if (
            prev_lat is None
            or prev_lon is None
        ):
            travel = {
                "minutes": 0.0,
                "distance_km": 0.0,
                "source": "첫장소",
            }
        else:
            travel = provider.get(
                prev_lat,
                prev_lon,
                row["위도"],
                row["경도"],
            )

        travel_min = float(travel["minutes"])
        distance_km = float(travel["distance_km"])

        arrival = current_minute + travel_min
        category = row["추천카테고리"]
        stay = STAY_MINUTES[category]
        depart = arrival + stay

        schedule.append({
            "idx": idx,
            "category": category,
            "arrival": arrival,
            "depart": depart,
            "stay": stay,
            "travel_min": travel_min,
            "distance_km": distance_km,
            "travel_source": travel["source"],
            "time_fit_score": evaluate_time_fit(
                category,
                arrival,
            ),
        })

        current_minute = depart
        total_place_score += float(
            row["장소추천점수"]
        )
        total_travel_min += travel_min
        total_distance_km += distance_km

        prev_lat = row["위도"]
        prev_lon = row["경도"]

    return (
        schedule,
        total_place_score,
        total_travel_min,
        total_distance_km,
    )


def route_total_minutes(
    route: list[int],
    candidates: pd.DataFrame,
    provider: TravelTimeProvider,
    start_lat: Optional[float],
    start_lon: Optional[float],
    start_minute: int,
) -> float:
    schedule, _, _, _ = simulate_route(
        route,
        candidates,
        provider,
        start_lat,
        start_lon,
        start_minute,
    )

    if not schedule:
        return 0.0

    return (
        schedule[-1]["depart"]
        - start_minute
    )


def natural_flow_score(
    schedule: list[dict],
) -> float:
    """
    카테고리 흐름 평가.

    핵심:
    - 맛집 연속 방문은 사실상 선택되지 않도록 매우 큰 감점
    - 카페 연속 방문도 큰 감점
    - 같은 카테고리 연속 반복은 일반적으로 감점
    - 관광지 → 맛집/카페, 맛집 → 관광지/카페 같은 전환에는 소폭 보너스
    """
    if not schedule:
        return 0.0

    score = sum(
        stop["time_fit_score"]
        for stop in schedule
    )

    for prev, cur in zip(
        schedule,
        schedule[1:],
    ):
        a = prev["category"]
        b = cur["category"]

        if a == b:
            if b == "맛집":
                score -= 100.0
            elif b == "카페/베이커리":
                score -= 60.0
            else:
                score -= 10.0
        else:
            # 서로 다른 성격의 장소로 이동하면 소폭 보너스
            score += 5.0

            if (
                a == "관광지"
                and b in {"맛집", "카페/베이커리"}
            ):
                score += 4.0

            if (
                a == "맛집"
                and b in {"관광지", "카페/베이커리"}
            ):
                score += 4.0

    return score


def violates_category_limit(
    route: list[int],
    candidates: pd.DataFrame,
    selected_categories: list[str],
) -> bool:
    categories = [
        candidates.loc[i]["추천카테고리"]
        for i in route
    ]

    for category in set(categories):
        if categories.count(category) > get_category_limit(
            category,
            selected_categories,
        ):
            return True

    return False


def optimize_route(
    candidates: pd.DataFrame,
    selected_categories: list[str],
    budget_min: float,
    max_stops: int,
    target_stops: int,
    provider: TravelTimeProvider,
    start_lat: Optional[float] = None,
    start_lon: Optional[float] = None,
    start_time: str = DEFAULT_START_TIME,
) -> list[int]:

    candidates = candidates.reset_index(
        drop=True
    )

    if candidates.empty:
        return []

    start_minute = parse_hhmm(
        start_time
    )

    # 상태 = (route, objective)
    beam = [
        ([], 0.0)
    ]

    finished = []

    for _ in range(max_stops):

        next_states = []

        for route, _ in beam:
            used = set(route)

            for idx, row in candidates.iterrows():
                idx = int(idx)

                if idx in used:
                    continue

                # 동일 관광권역의 관광지를 연달아 여러 개 넣는 것 방지
                if near_duplicate_tourist_stop(
                    idx,
                    route,
                    candidates,
                ):
                    continue

                new_route = route + [idx]

                # 여러 카테고리 선택 시 맛집/카페 과다 포함 방지
                if violates_category_limit(
                    new_route,
                    candidates,
                    selected_categories,
                ):
                    continue

                total_minutes = route_total_minutes(
                    new_route,
                    candidates,
                    provider,
                    start_lat,
                    start_lon,
                    start_minute,
                )

                if total_minutes > budget_min:
                    continue

                (
                    schedule,
                    place_score,
                    travel_min,
                    _,
                ) = simulate_route(
                    new_route,
                    candidates,
                    provider,
                    start_lat,
                    start_lon,
                    start_minute,
                )

                categories = [
                    stop["category"]
                    for stop in schedule
                ]

                diversity_bonus = (
                    category_quota_bonus(
                        categories,
                        selected_categories,
                    )
                )

                flow_score = natural_flow_score(
                    schedule
                )

                tour_diversity_score = (
                    tourist_subcategory_diversity_score(
                        new_route,
                        candidates,
                    )
                )

                # 이동이 짧을수록 좋지만, 장소 점수/자연스러운 일정도 함께 고려
                travel_penalty = (
                    travel_min * 0.55
                )

                # 사용자가 선택한 카테고리를 빠르게 충족하도록 보너스
                missing_category_count = len(
                    set(selected_categories)
                    - set(categories)
                )
                missing_penalty = (
                    missing_category_count * 12.0
                )

                stop_bonus = (
                    min(
                        len(new_route),
                        target_stops,
                    )
                    * 2.5
                )

                objective = (
                    place_score
                    + diversity_bonus
                    + flow_score
                    + tour_diversity_score
                    + stop_bonus
                    - travel_penalty
                    - missing_penalty
                )

                state = (
                    new_route,
                    objective,
                )

                next_states.append(state)
                finished.append(state)

        if not next_states:
            break

        next_states.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        # 같은 방문 집합 + 마지막 장소가 동일한 상태는 중복 제거
        unique = []
        seen = set()

        for route, score in next_states:
            signature = (
                frozenset(route),
                route[-1],
            )

            if signature in seen:
                continue

            seen.add(signature)
            unique.append(
                (route, score)
            )

            if len(unique) >= BEAM_WIDTH:
                break

        beam = unique

    if not finished:
        return []

    def final_key(state):
        route, objective = state

        categories = [
            candidates.loc[i][
                "추천카테고리"
            ]
            for i in route
        ]

        coverage = len(
            set(categories)
            & set(selected_categories)
        )

        # 최종 선택 우선순위:
        # 1) 선택 카테고리 충족
        # 2) 목표 방문지 수 충족
        # 3) 자연스러운 일정 + 장소점수 + 이동효율
        return (
            coverage,
            min(
                len(route),
                target_stops,
            ),
            objective,
        )

    best = max(
        finished,
        key=final_key,
    )

    return best[0]


# ============================================================
# 9. 경로 상세 생성
# ============================================================

def build_route_detail(
    route: list[int],
    candidates: pd.DataFrame,
    provider: TravelTimeProvider,
    day: int,
    start_lat: Optional[float],
    start_lon: Optional[float],
    start_time: str = DEFAULT_START_TIME,
) -> pd.DataFrame:

    start_minute = parse_hhmm(
        start_time
    )

    (
        schedule,
        _,
        _,
        _,
    ) = simulate_route(
        route,
        candidates,
        provider,
        start_lat,
        start_lon,
        start_minute,
    )

    rows = []

    for order, stop in enumerate(
        schedule,
        start=1,
    ):
        idx = stop["idx"]
        row = candidates.loc[idx]

        rows.append({
            "day": day,
            "order": order,
            "도착예정":
                minutes_to_hhmm(
                    stop["arrival"]
                ),
            "출발예정":
                minutes_to_hhmm(
                    stop["depart"]
                ),
            "place_id":
                row["place_id"],
            "장소명":
                row["장소명"],
            "카테고리":
                row["추천카테고리"],
            "세부분류":
                row["세부분류"],
            "시도":
                row["시도"],
            "시군구":
                row["시군구"],
            "장소추천점수":
                round(
                    float(
                        row[
                            "장소추천점수"
                        ]
                    ),
                    2,
                ),
            "현지인순위":
                row["현지인순위"],
            "외지인순위":
                row["외지인순위"],
            "시간대적합점수":
                round(
                    float(
                        stop[
                            "time_fit_score"
                        ]
                    ),
                    1,
                ),
            "이전장소에서_이동분":
                round(
                    float(
                        stop["travel_min"]
                    ),
                    1,
                ),
            "이전장소에서_이동km":
                round(
                    float(
                        stop["distance_km"]
                    ),
                    2,
                ),
            "이동시간출처":
                stop["travel_source"],
            "체류시간분":
                stop["stay"],
            "누적소요시간분":
                round(
                    stop["depart"]
                    - start_minute,
                    1,
                ),
            "위도":
                row["위도"],
            "경도":
                row["경도"],
            "주소":
                row["주소"],
            "원본구분":
                row["원본구분"],
        })

    return pd.DataFrame(rows)


# ============================================================
# 10. 1박2일
# ============================================================

def optimize_1n2d(
    candidates: pd.DataFrame,
    selected_categories: list[str],
    provider: TravelTimeProvider,
    start_lat: Optional[float],
    start_lon: Optional[float],
    start_time: str = DEFAULT_START_TIME,
) -> tuple[pd.DataFrame, dict]:

    # 1일차: 8시간
    route1 = optimize_route(
        candidates,
        selected_categories,
        budget_min=480,
        max_stops=6,
        target_stops=5,
        provider=provider,
        start_lat=start_lat,
        start_lon=start_lon,
        start_time=start_time,
    )

    if not route1:
        raise RuntimeError(
            "1일차 경로를 생성하지 못했습니다."
        )

    detail1 = build_route_detail(
        route1,
        candidates,
        provider,
        day=1,
        start_lat=start_lat,
        start_lon=start_lon,
        start_time=start_time,
    )

    used_ids = {
        candidates.loc[i][
            "place_id"
        ]
        for i in route1
    }

    remaining = candidates[
        ~candidates[
            "place_id"
        ].isin(used_ids)
    ].copy().reset_index(
        drop=True
    )

    # 숙소 데이터가 아직 없으므로,
    # 1일차 마지막 장소 인근에서 숙박한다고 가정
    last = candidates.loc[
        route1[-1]
    ]

    day2_start_lat = float(
        last["위도"]
    )
    day2_start_lon = float(
        last["경도"]
    )

    route2 = optimize_route(
        remaining,
        selected_categories,
        budget_min=480,
        max_stops=6,
        target_stops=5,
        provider=provider,
        start_lat=day2_start_lat,
        start_lon=day2_start_lon,
        start_time=start_time,
    )

    if route2:
        detail2 = build_route_detail(
            route2,
            remaining,
            provider,
            day=2,
            start_lat=day2_start_lat,
            start_lon=day2_start_lon,
            start_time=start_time,
        )

        detail = pd.concat(
            [detail1, detail2],
            ignore_index=True,
        )
    else:
        detail = detail1

    meta = {
        "1일차장소수": len(route1),
        "2일차장소수": len(route2),
        "숙박가정":
            (
                f"1일차 마지막 장소 "
                f"'{last['장소명']}' 인근 숙박"
            ),
    }

    return detail, meta


# ============================================================
# 11. 메인 추천 함수
# ============================================================

def recommend_local_on_trip(
    selected_sido: str,
    selected_categories: list[str],
    trip_type: str = "day",
    selected_sigungu: Optional[str] = None,
    start_lat: Optional[float] = None,
    start_lon: Optional[float] = None,
    start_time: str = DEFAULT_START_TIME,
    top_region_n: int = 5,
) -> dict:

    if trip_type not in {
        "2h",
        "4h",
        "6h",
        "day",
        "1n2d",
    }:
        raise ValueError(
            "trip_type은 "
            "2h/4h/6h/day/1n2d 중 하나여야 합니다."
        )

    selected_categories = [
        x.strip()
        for x in selected_categories
        if x.strip()
    ]

    if not selected_categories:
        raise ValueError(
            "최소 1개 카테고리를 선택해야 합니다."
        )

    # --------------------------------------------------------
    # STEP 1. 지역 순위
    # --------------------------------------------------------
    region_df = load_region_scores()

    region_ranking = get_region_ranking(
        region_df,
        selected_sido,
    )

    region_out = (
        OUTPUT_DIR
        / f"01_{selected_sido}_지역추천순위.csv"
    )

    region_ranking.to_csv(
        region_out,
        index=False,
        encoding="utf-8-sig",
    )

    formal_regions = region_ranking[
        region_ranking[
            "지역추천구분"
        ] == "정식_12개월"
    ]

    # 사용자가 시군구를 직접 선택하지 않았으면
    # 해당 시도 로컬발견가능성 1위 지역 사용
    if selected_sigungu is None:
        if not formal_regions.empty:
            selected_sigungu = (
                formal_regions.iloc[0][
                    "시군구"
                ]
            )
        else:
            usable_short = region_ranking[
                region_ranking[
                    "지역추천점수"
                ].notna()
            ]

            if usable_short.empty:
                raise RuntimeError(
                    "추천 가능한 지역 점수가 없습니다."
                )

            selected_sigungu = (
                usable_short.iloc[0][
                    "시군구"
                ]
            )

    valid_selected = (
        (
            region_ranking["시군구"]
            == selected_sigungu
        )
    ).any()

    if not valid_selected:
        raise ValueError(
            f"{selected_sido} 안에 "
            f"'{selected_sigungu}' 지역이 없습니다."
        )

    # --------------------------------------------------------
    # STEP 2. 장소 데이터
    # --------------------------------------------------------
    food = load_food_places()

    tour, tour_qa = load_tour_places(
        region_df
    )

    qa_path = (
        OUTPUT_DIR
        / "00_관광지데이터_QA.csv"
    )

    tour_qa.to_csv(
        qa_path,
        index=False,
        encoding="utf-8-sig",
    )

    all_places = pd.concat(
        [food, tour],
        ignore_index=True,
    )

    # 같은 데이터셋 내부 동일 ID는 이미 제거/검증.
    # 음식점/관광지는 ID 출처가 달라도 충돌 가능하므로
    # 원본구분을 붙인 내부키 사용.
    all_places["internal_id"] = (
        all_places["원본구분"]
        + "_"
        + all_places["place_id"]
    )

    # --------------------------------------------------------
    # STEP 3. 선택 지역 후보
    # --------------------------------------------------------
    candidates = get_place_candidates(
        all_places,
        selected_sido,
        selected_sigungu,
        selected_categories,
    )

    # 출발지가 없으면 선택 지역 후보들의 중앙값 좌표를 자동 사용
    auto_start_used = False

    if start_lat is None or start_lon is None:
        start_lat, start_lon = infer_region_center(
            candidates
        )
        auto_start_used = True

    candidate_path = (
        OUTPUT_DIR
        / (
            f"02_{selected_sido}_"
            f"{selected_sigungu}_장소후보.csv"
        )
    )

    candidates.to_csv(
        candidate_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # STEP 4. 경로
    # --------------------------------------------------------
    provider = TravelTimeProvider(
        cache_file=(
            OUTPUT_DIR
            / "travel_time_cache.json"
        ),
        api_key=KAKAO_REST_API_KEY,
    )

    route_meta = {}

    if trip_type == "1n2d":

        route_detail, route_meta = (
            optimize_1n2d(
                candidates,
                selected_categories,
                provider,
                start_lat,
                start_lon,
                start_time,
            )
        )

    else:
        cfg = TRIP_CONFIG[
            trip_type
        ]

        route = optimize_route(
            candidates,
            selected_categories,
            budget_min=cfg[
                "budget_min"
            ],
            max_stops=cfg[
                "max_stops"
            ],
            target_stops=cfg[
                "target_stops"
            ],
            provider=provider,
            start_lat=start_lat,
            start_lon=start_lon,
            start_time=start_time,
        )

        if not route:
            raise RuntimeError(
                "주어진 시간 안에 경로를 생성하지 못했습니다."
            )

        route_detail = build_route_detail(
            route,
            candidates,
            provider,
            day=1,
            start_lat=start_lat,
            start_lon=start_lon,
            start_time=start_time,
        )

    provider.save()

    # 선택한 모든 카테고리가 가능한 경우
    # 실제 최종 경로에 포함됐는지 체크
    available_categories = set(
        candidates["추천카테고리"]
    )

    expected_categories = (
        set(selected_categories)
        & available_categories
    )

    actual_categories = set(
        route_detail["카테고리"]
    )

    missing_route_categories = (
        expected_categories
        - actual_categories
    )

    route_path = (
        OUTPUT_DIR
        / (
            f"03_{selected_sido}_"
            f"{selected_sigungu}_"
            f"{trip_type}_추천경로.csv"
        )
    )

    route_detail.to_csv(
        route_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # STEP 5. 요약
    # --------------------------------------------------------
    selected_region_row = (
        region_ranking[
            region_ranking[
                "시군구"
            ] == selected_sigungu
        ]
        .iloc[0]
    )

    summary = {
        "선택시도":
            selected_sido,
        "추천/선택시군구":
            selected_sigungu,
        "시도내순위":
            int(
                selected_region_row[
                    "시도내순위"
                ]
            ),
        "지역추천구분":
            selected_region_row[
                "지역추천구분"
            ],
        "지역추천점수":
            (
                float(
                    selected_region_row[
                        "지역추천점수"
                    ]
                )
                if pd.notna(
                    selected_region_row[
                        "지역추천점수"
                    ]
                )
                else None
            ),
        "선택카테고리":
            selected_categories,
        "여행유형":
            trip_type,
        "일정시작시간":
            start_time,
        "출발지자동설정":
            auto_start_used,
        "출발위도":
            round(float(start_lat), 6),
        "출발경도":
            round(float(start_lon), 6),
        "경로장소수":
            int(
                len(route_detail)
            ),
        "경로포함카테고리":
            sorted(
                actual_categories
            ),
        "포함하지못한카테고리":
            sorted(
                missing_route_categories
            ),
        "관광지세부분류":
            route_detail.loc[
                route_detail["카테고리"] == "관광지",
                "세부분류"
            ].astype(str).tolist(),
        "총이동시간분":
            round(
                float(
                    route_detail[
                        "이전장소에서_이동분"
                    ].sum()
                ),
                1,
            ),
        "총이동거리km":
            round(
                float(
                    route_detail[
                        "이전장소에서_이동km"
                    ].sum()
                ),
                2,
            ),
        "이동시간방식":
            (
                "Kakao Mobility 자동차 길찾기"
                if KAKAO_REST_API_KEY
                else "위경도 기반 추정"
            ),
        "지역추천TOP":
            (
                region_ranking
                .head(top_region_n)[
                    [
                        "시군구",
                        "지역추천구분",
                        "지역추천점수",
                        "시도내순위",
                    ]
                ]
                .to_dict(
                    orient="records"
                )
            ),
        **route_meta,
    }

    summary_path = (
        OUTPUT_DIR
        / (
            f"04_{selected_sido}_"
            f"{selected_sigungu}_"
            f"{trip_type}_요약.json"
        )
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("LOCAL:ON 추천 완료")
    print("=" * 70)
    print(
        f"선택 시도: {selected_sido}"
    )
    print(
        f"추천/선택 지역: {selected_sigungu}"
    )
    print(
        "선택 카테고리:",
        ", ".join(
            selected_categories
        )
    )
    print(
        f"여행 유형: {trip_type}"
    )
    print()
    print(
        route_detail[
            [
                "day",
                "order",
                "도착예정",
                "출발예정",
                "장소명",
                "카테고리",
                "장소추천점수",
                "이전장소에서_이동분",
                "체류시간분",
            ]
        ].to_string(
            index=False
        )
    )
    print()
    print("생성 파일")
    print(" -", region_out)
    print(" -", candidate_path)
    print(" -", route_path)
    print(" -", summary_path)
    print(" -", qa_path)

    return {
        "region_ranking":
            region_ranking,
        "candidates":
            candidates,
        "route":
            route_detail,
        "summary":
            summary,
        "tour_qa":
            tour_qa,
    }


# ============================================================
# 12. CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "LOCAL:ON 지역/장소/경로 추천"
        )
    )

    parser.add_argument(
        "--sido",
        default="충청남도",
        help="예: 충청남도",
    )

    parser.add_argument(
        "--sigungu",
        default=None,
        help=(
            "예: 공주시. "
            "생략 시 해당 시도 로컬발견가능성 "
            "1위 지역을 자동 선택"
        ),
    )

    parser.add_argument(
        "--trip",
        default="day",
        choices=[
            "2h",
            "4h",
            "6h",
            "day",
            "1n2d",
        ],
    )

    parser.add_argument(
        "--categories",
        default=(
            "맛집,관광지,카페/베이커리"
        ),
        help=(
            "쉼표 구분. "
            "맛집,관광지,카페/베이커리"
        ),
    )

    parser.add_argument(
        "--start-lat",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--start-lon",
        type=float,
        default=None,
    )


    parser.add_argument(
        "--start-time",
        default=DEFAULT_START_TIME,
        help="여행 시작 시각 HH:MM (기본 10:00)",
    )

    args = parser.parse_args()

    categories = [
        x.strip()
        for x in args.categories.split(",")
        if x.strip()
    ]

    recommend_local_on_trip(
        selected_sido=args.sido,
        selected_sigungu=args.sigungu,
        selected_categories=categories,
        trip_type=args.trip,
        start_lat=args.start_lat,
        start_lon=args.start_lon,
        start_time=args.start_time,
    )


if __name__ == "__main__":
    main()
