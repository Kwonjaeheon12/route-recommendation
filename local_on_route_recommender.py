# -*- coding: utf-8 -*-
"""
LOCAL:ON 경로 추천 모델 v5.1
==========================
분리형 DB + 카테고리 공정비교 + 시장 정규화 고도화 버전

[지역]
- 03_로컬발견가능성_지역별.csv

[음식점]
- restaurant_place_지역정규화.csv
- restaurant_score.csv
- restaurant_location.csv
- place_restaurant.csv

[전통시장]
- traditional_market.csv
- traditional_market_facility.csv
- place_market.csv

[관광지]
- 01_인기관광지_전국통합_점수_최종.csv
- 01_인기관광지_전국통합_위경도_최최종.csv

[관광지 QA/비교]
- 인기관광지_전국통합_점수(2).csv

핵심 원칙
---------
1. 원본 CSV를 물리적으로 합치지 않는다.
2. 실행 시 ID를 기준으로 메모리에서 JOIN한다.
3. 음식점 점수는 restaurant_score.csv의 최종 추천점수를 그대로 사용한다.
4. 전통시장은 별도 인기순위 데이터가 없으므로,
   시장 규모(점포수) + 주차장 + 화장실을 이용한 '시장경로점수'를 사용한다.
5. 관광지는 최종 점수 파일을 라우팅에 사용하고,
   인기관광지_전국통합_점수(2).csv는 QA/감사용으로만 읽는다.
6. 사용자 선택:
   시도 → 시군구 → 카테고리 → 일정 유형 → 최종 경로
"""

from __future__ import annotations

import os
import re
import json
import math
import argparse
import difflib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:
    requests = None


# ============================================================
# 0. 경로 설정
# ============================================================

DEFAULT_BASE_DIR = r"C:\Users\User\Desktop"
BASE_DIR = Path(os.getenv("LOCAL_ON_BASE_DIR", DEFAULT_BASE_DIR))

OUTPUT_DIR_NAME = "경로추천_결과"


def resolve_file(base_dir: Path, filename: str) -> Path:
    """
    실제 파일명은 바꾸지 않는다.
    먼저 정확한 파일명을 찾고, 테스트 환경에서만 (1)~(9) 복사본도 허용한다.
    """
    exact = base_dir / filename
    if exact.exists():
        return exact

    stem = Path(filename).stem
    suffix = Path(filename).suffix

    for i in range(9, 0, -1):
        alt = base_dir / f"{stem}({i}){suffix}"
        if alt.exists():
            return alt

    raise FileNotFoundError(
        f"파일을 찾을 수 없습니다: {filename}\n"
        f"기준 폴더: {base_dir}"
    )


# 실제 데이터 파일명 그대로 유지
REGION_SCORE_NAME = "03_로컬발견가능성_지역별.csv"

RESTAURANT_PLACE_NAME = "restaurant_place_지역정규화.csv"
RESTAURANT_SCORE_NAME = "restaurant_score.csv"
RESTAURANT_LOCATION_NAME = "restaurant_location.csv"
PLACE_RESTAURANT_NAME = "place_restaurant.csv"

TRADITIONAL_MARKET_NAME = "traditional_market.csv"
TRADITIONAL_MARKET_FACILITY_NAME = "traditional_market_facility.csv"
PLACE_MARKET_NAME = "place_market.csv"

TOUR_SCORE_FINAL_NAME = "01_인기관광지_전국통합_점수_최종.csv"
TOUR_GEO_FINAL_NAME = "01_인기관광지_전국통합_위경도_최최종.csv"

# 최종 경로 계산에는 사용하지 않고 QA에만 사용
TOUR_SCORE_AUDIT_NAME = "인기관광지_전국통합_점수(2).csv"


# ============================================================
# 1. 모델 설정
# ============================================================

VALID_CATEGORIES = {
    "맛집",
    "관광지",
    "카페/베이커리",
    "전통시장",
}

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

STAY_MINUTES = {
    "맛집": 70,
    "관광지": 70,
    "카페/베이커리": 45,
    "전통시장": 60,
}

DEFAULT_START_TIME = "10:00"

LUNCH_WINDOW = (11 * 60 + 30, 14 * 60)
DINNER_WINDOW = (17 * 60 + 30, 20 * 60)
CAFE_WINDOW = (13 * 60, 17 * 60 + 30)
MARKET_WINDOW = (10 * 60, 18 * 60)

MULTI_CATEGORY_MAX_VISITS = {
    "맛집": 1,
    "카페/베이커리": 1,
    "전통시장": 1,
    "관광지": 99,
}

TOP_CANDIDATES_PER_CATEGORY = 10
BEAM_WIDTH = 180

NEARBY_TOUR_CLUSTER_KM = 0.5
SAME_TOUR_SUBCATEGORY_PENALTY = 10.0

# 서로 다른 카테고리의 원점수를 직접 비교하지 않기 위한 공통 경로점수.
# 각 카테고리 후보 내 순위 1위=100, 이후 5점씩 감소(최저 55).
ROUTE_RANK_STEP = 5.0
ROUTE_SCORE_FLOOR = 55.0

# 관광지와 전통시장 DB가 사실상 같은 장소인지 판단할 거리.
MARKET_DUPLICATE_DISTANCE_KM = 0.6
MARKET_NAME_SIMILARITY = 0.72

# 일반 관광 모드에서 특수 체험형 관광지가 과도하게 선정되는 것을 완화.
GENERAL_TOUR_POSITIVE_BONUS = 5.0
GENERAL_TOUR_SPECIAL_PENALTY = -12.0

EXCLUDED_TOUR_CATEGORIES = {
    "쇼핑몰",
    "백화점",
    "기타쇼핑시설",
    "면세점",
    "전문매장/상가",
    "호스텔",
    "호텔",
    "모텔",
    "교통시설",
    "대형마트",
}


# ============================================================
# 2. 공통 함수
# ============================================================

def read_csv_safely(path: Path) -> pd.DataFrame:
    last_error = None

    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_error = e

    raise RuntimeError(
        f"CSV 읽기 실패: {path}\n{last_error}"
    )


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def normalize_100(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()

    if valid.empty:
        return pd.Series(
            50.0,
            index=s.index,
            dtype=float,
        )

    lo = valid.min()
    hi = valid.max()

    if hi == lo:
        out = pd.Series(
            50.0,
            index=s.index,
            dtype=float,
        )
        return out

    return (s - lo) / (hi - lo) * 100.0


def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2,
) -> float:
    r = 6371.0088

    lat1 = math.radians(float(lat1))
    lon1 = math.radians(float(lon1))
    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * r * math.asin(math.sqrt(a))


def require_columns(
    df: pd.DataFrame,
    columns: set[str],
    label: str,
):
    missing = columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{label} 필수 컬럼 누락: {sorted(missing)}"
        )


def assert_unique(
    df: pd.DataFrame,
    key: str,
    label: str,
):
    duplicated = df[key].duplicated().sum()

    if duplicated:
        raise ValueError(
            f"{label}: {key} 중복 {duplicated}건"
        )


# ============================================================
# 3. 지역 데이터
# ============================================================

def load_region_scores(base_dir: Path) -> pd.DataFrame:
    path = resolve_file(
        base_dir,
        REGION_SCORE_NAME,
    )

    df = read_csv_safely(path).copy()

    require_columns(
        df,
        {
            "시도",
            "시군구",
            "데이터개월수",
            "신뢰도",
            "로컬발견가능성",
            "단기임시점수",
        },
        REGION_SCORE_NAME,
    )

    if df.duplicated(
        ["시도", "시군구"]
    ).any():
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

    x = region_df[
        region_df["시도"] == selected_sido
    ].copy()

    if x.empty:
        raise ValueError(
            f"'{selected_sido}' 지역 데이터가 없습니다."
        )

    formal = x[
        x["로컬발견가능성"].notna()
    ].copy()

    formal = formal.sort_values(
        ["로컬발견가능성", "시군구"],
        ascending=[False, True],
    )

    formal["지역추천구분"] = "정식_12개월"
    formal["지역추천점수"] = (
        formal["로컬발견가능성"]
    )
    formal["시도내순위"] = np.arange(
        1,
        len(formal) + 1,
    )

    short = x[
        x["로컬발견가능성"].isna()
    ].copy()

    short = short.sort_values(
        ["단기임시점수", "시군구"],
        ascending=[False, True],
        na_position="last",
    )

    short["지역추천구분"] = "참고_단기"
    short["지역추천점수"] = (
        short["단기임시점수"]
    )
    short["시도내순위"] = np.arange(
        1,
        len(short) + 1,
    )

    result = pd.concat(
        [formal, short],
        ignore_index=True,
    )

    return result[
        [
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
    ]


# ============================================================
# 4. 음식점 데이터
# ============================================================

def classify_restaurant(
    row: pd.Series,
) -> str:

    text = " ".join(
        [
            str(row.get("업소명", "")),
            str(row.get("분류", "")),
            str(row.get("name", "")),
        ]
    ).lower()

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

    if any(
        keyword in text
        for keyword in cafe_keywords
    ):
        return "카페/베이커리"

    return "맛집"


def load_restaurant_places(
    base_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    place_info = read_csv_safely(
        resolve_file(
            base_dir,
            RESTAURANT_PLACE_NAME,
        )
    )

    score = read_csv_safely(
        resolve_file(
            base_dir,
            RESTAURANT_SCORE_NAME,
        )
    )

    location = read_csv_safely(
        resolve_file(
            base_dir,
            RESTAURANT_LOCATION_NAME,
        )
    )

    place = read_csv_safely(
        resolve_file(
            base_dir,
            PLACE_RESTAURANT_NAME,
        )
    )

    require_columns(
        place_info,
        {
            "관광지ID",
            "업소명",
            "분류",
            "현지인순위",
            "외지인순위",
            "표준시도",
            "표준시군구",
        },
        RESTAURANT_PLACE_NAME,
    )

    require_columns(
        score,
        {
            "관광지ID",
            "현지인순위",
            "외지인순위",
            "현지인점수",
            "외지인점수",
            "추천점수",
            "추천순위",
        },
        RESTAURANT_SCORE_NAME,
    )

    require_columns(
        location,
        {
            "관광지ID",
            "경도",
            "위도",
        },
        RESTAURANT_LOCATION_NAME,
    )

    require_columns(
        place,
        {
            "place_id",
            "place_type",
            "name",
            "latitude",
            "longitude",
            "road_address",
            "jibun_address",
        },
        PLACE_RESTAURANT_NAME,
    )

    for df, key, label in [
        (
            place_info,
            "관광지ID",
            RESTAURANT_PLACE_NAME,
        ),
        (
            score,
            "관광지ID",
            RESTAURANT_SCORE_NAME,
        ),
        (
            location,
            "관광지ID",
            RESTAURANT_LOCATION_NAME,
        ),
        (
            place,
            "place_id",
            PLACE_RESTAURANT_NAME,
        ),
    ]:
        assert_unique(
            df,
            key,
            label,
        )

    # --------------------------------------------------------
    # 4개 음식점 테이블 JOIN
    # --------------------------------------------------------
    merged = (
        place_info
        .merge(
            score,
            on="관광지ID",
            how="inner",
            suffixes=(
                "_place",
                "_score",
            ),
            validate="one_to_one",
        )
        .merge(
            location[
                [
                    "관광지ID",
                    "경도",
                    "위도",
                    "좌표보완방법",
                    "좌표검색어",
                ]
            ],
            on="관광지ID",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            place,
            left_on="관광지ID",
            right_on="place_id",
            how="inner",
            validate="one_to_one",
        )
    )

    qa_rows = []

    # ID 개수 검증
    expected = len(place_info)

    if len(merged) != expected:
        qa_rows.append(
            {
                "데이터구분": "음식점",
                "검사항목": "JOIN_행수",
                "결과": len(merged),
                "기대값": expected,
                "상태": "확인필요",
            }
        )

    # 순위 일치 검증
    for col in (
        "현지인순위",
        "외지인순위",
    ):
        left = merged[
            f"{col}_place"
        ]
        right = merged[
            f"{col}_score"
        ]

        mismatch = (
            left.fillna(-999999)
            != right.fillna(-999999)
        ).sum()

        qa_rows.append(
            {
                "데이터구분": "음식점",
                "검사항목":
                    f"{col}_테이블간불일치",
                "결과": int(mismatch),
                "기대값": 0,
                "상태":
                    (
                        "정상"
                        if mismatch == 0
                        else "확인필요"
                    ),
            }
        )

    # 위치 테이블과 place 테이블 좌표 일치 검증
    lat_mismatch = (
        (
            to_numeric(
                merged["위도"]
            )
            - to_numeric(
                merged["latitude"]
            )
        ).abs()
        > 1e-7
    ).sum()

    lon_mismatch = (
        (
            to_numeric(
                merged["경도"]
            )
            - to_numeric(
                merged["longitude"]
            )
        ).abs()
        > 1e-7
    ).sum()

    qa_rows.extend(
        [
            {
                "데이터구분": "음식점",
                "검사항목":
                    "위도_테이블간불일치",
                "결과": int(lat_mismatch),
                "기대값": 0,
                "상태":
                    (
                        "정상"
                        if lat_mismatch == 0
                        else "확인필요"
                    ),
            },
            {
                "데이터구분": "음식점",
                "검사항목":
                    "경도_테이블간불일치",
                "결과": int(lon_mismatch),
                "기대값": 0,
                "상태":
                    (
                        "정상"
                        if lon_mismatch == 0
                        else "확인필요"
                    ),
            },
        ]
    )

    merged["추천카테고리"] = (
        merged.apply(
            classify_restaurant,
            axis=1,
        )
    )

    road = merged[
        "road_address"
    ].fillna("")

    jibun = merged[
        "jibun_address"
    ].fillna("")

    address = road.where(
        road.str.strip().ne(""),
        jibun,
    )

    out = pd.DataFrame(
        {
            "place_id":
                merged["관광지ID"].astype(str),
            "장소명":
                merged["업소명"].astype(str),
            "추천카테고리":
                merged["추천카테고리"],
            "세부분류":
                merged["분류"].astype(str),
            "시도":
                merged["표준시도"].astype(str),
            "시군구":
                merged["표준시군구"].astype(str),
            "현지인순위":
                to_numeric(
                    merged["현지인순위_score"]
                ),
            "외지인순위":
                to_numeric(
                    merged["외지인순위_score"]
                ),
            "현지인점수":
                to_numeric(
                    merged["현지인점수"]
                ),
            "외지인점수":
                to_numeric(
                    merged["외지인점수"]
                ),
            "장소추천점수":
                to_numeric(
                    merged["추천점수"]
                ),
            "원본추천순위":
                to_numeric(
                    merged["추천순위"]
                ),
            "위도":
                to_numeric(
                    merged["위도"]
                ),
            "경도":
                to_numeric(
                    merged["경도"]
                ),
            "주소":
                address,
            "부가정보":
                "",
            "원본구분":
                "음식점",
            "지역매칭상태":
                "정규화완료",
        }
    )

    out = out[
        out["위도"].between(
            32,
            39.5,
        )
        & out["경도"].between(
            124,
            132.5,
        )
        & out["장소추천점수"].notna()
        & out["시도"].ne("")
        & out["시군구"].ne("")
    ].copy()

    qa_rows.append(
        {
            "데이터구분": "음식점",
            "검사항목": "최종사용가능행",
            "결과": len(out),
            "기대값": len(place_info),
            "상태":
                (
                    "정상"
                    if len(out) == len(place_info)
                    else "확인필요"
                ),
        }
    )

    return (
        out.reset_index(drop=True),
        pd.DataFrame(qa_rows),
    )


# ============================================================
# 5. 전통시장 데이터
# ============================================================


def normalize_address_text(address: str) -> str:
    """
    공공데이터 주소의 흔한 표기 차이/오탈자를 모델 내부에서만 보정한다.
    원본 CSV는 수정하지 않는다.
    """
    if pd.isna(address):
        return ""

    text = str(address).strip()

    replacements = {
        "전북특별차치도": "전북특별자치도",
        "전라북도": "전북특별자치도",
        "강원도": "강원특별자치도",
        "제주도": "제주특별자치도",
        "충남 ": "충청남도 ",
        "충북 ": "충청북도 ",
        "경남 ": "경상남도 ",
        "경북 ": "경상북도 ",
        "전남 ": "전라남도 ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def parse_basic_region_from_address(
    address: str,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    주소 자체에서 시도 / 기본 시군구 / 하위구를 먼저 추출한다.
    예:
      충청남도 천안시 서북구 ... -> 충청남도, 천안시, 서북구
      서울특별시 노원구 ...      -> 서울특별시, 노원구, None
    """
    text = normalize_address_text(address)

    if not text:
        return None, None, None

    tokens = text.split()

    if not tokens:
        return None, None, None

    sido = ADDRESS_SIDO_MAP.get(tokens[0])

    if sido is None:
        return None, None, None

    if sido == "세종특별자치시":
        return sido, "세종특별자치시", None

    if len(tokens) < 2:
        return sido, None, None

    primary = tokens[1]
    sub = None

    if (
        len(tokens) >= 3
        and primary.endswith("시")
        and tokens[2].endswith("구")
    ):
        sub = tokens[2]

    return sido, primary, sub


ADDRESS_SIDO_MAP = {
    "서울특별시": "서울특별시",
    "서울": "서울특별시",
    "부산광역시": "부산광역시",
    "부산": "부산광역시",
    "대구광역시": "대구광역시",
    "대구": "대구광역시",
    "인천광역시": "인천광역시",
    "인천": "인천광역시",
    "광주광역시": "광주광역시",
    "광주": "광주광역시",
    "대전광역시": "대전광역시",
    "대전": "대전광역시",
    "울산광역시": "울산광역시",
    "울산": "울산광역시",
    "세종특별자치시": "세종특별자치시",
    "세종": "세종특별자치시",
    "경기도": "경기도",
    "경기": "경기도",
    "강원특별자치도": "강원특별자치도",
    "강원도": "강원특별자치도",
    "충청북도": "충청북도",
    "충북": "충청북도",
    "충청남도": "충청남도",
    "충남": "충청남도",
    "전북특별자치도": "전북특별자치도",
    "전라북도": "전북특별자치도",
    "전라남도": "전라남도",
    "전남": "전라남도",
    "경상북도": "경상북도",
    "경북": "경상북도",
    "경상남도": "경상남도",
    "경남": "경상남도",
    "제주특별자치도": "제주특별자치도",
    "제주도": "제주특별자치도",
}


def parse_region_from_address(
    address: str,
    valid_region_keys: set[tuple[str, str]],
) -> tuple[Optional[str], Optional[str], str]:
    """
    반환:
      (시도, 시군구, 매칭상태)

    매칭상태:
      - exact
      - compound
      - fallback
      - region_score_unsupported
      - parse_failed

    지역 점수 데이터에 없는 지역이라고 해서 주소 파싱 자체를 실패로
    처리하지 않는다. 예: 서울특별시는 현재 로컬발견가능성 데이터가
    없을 수 있으므로 '지역점수미지원'으로 구분한다.
    """
    sido, primary, sub = parse_basic_region_from_address(
        address
    )

    if sido is None:
        return None, None, "parse_failed"

    # 세종
    if sido == "세종특별자치시":
        for candidate in (
            "세종특별자치시",
            "세종시",
            "세종",
        ):
            if (sido, candidate) in valid_region_keys:
                return sido, candidate, "exact"

        return sido, "세종특별자치시", "region_score_unsupported"

    # 시+구 복합 행정구역을 가장 먼저 확인
    if primary and sub:
        compound = f"{primary}+{sub}"

        if (sido, compound) in valid_region_keys:
            return sido, compound, "compound"

    # 기본 시군구
    if primary and (sido, primary) in valid_region_keys:
        return sido, primary, "exact"

    # 주소 전체에서 로컬발견가능성 지역명을 다시 탐색
    normalized_text = normalize_address_text(
        address
    ).replace(" ", "")

    candidates = [
        sigungu
        for s, sigungu in valid_region_keys
        if s == sido
    ]

    candidates.sort(
        key=len,
        reverse=True,
    )

    for sigungu in candidates:
        target = (
            str(sigungu)
            .replace("+", "")
            .replace(" ", "")
        )

        if target and target in normalized_text:
            return sido, sigungu, "fallback"

    # 주소 파싱은 성공했지만 지역 점수 테이블이 해당 지역을 지원하지 않음
    if primary:
        return sido, (
            f"{primary}+{sub}"
            if sub
            else primary
        ), "region_score_unsupported"

    return sido, None, "parse_failed"


def market_route_score(
    stores: pd.Series,
    parking: pd.Series,
    restroom: pd.Series,
) -> pd.Series:
    """
    전통시장은 현지인/외지인 인기순위가 없으므로
    '추천점수'와 동일한 의미로 취급하지 않는다.

    경로 안에서 시장 후보를 정렬하기 위한 시장경로점수:
      - 기본 50점
      - 점포수 규모: 최대 30점
      - 주차장: 10점
      - 공중화장실: 10점

    점포수는 log1p 후 Min-Max 정규화하여 대형시장 이상치 영향을 완화한다.
    """
    stores_num = to_numeric(
        stores
    ).fillna(0)

    store_component = (
        normalize_100(
            np.log1p(stores_num)
        )
        * 0.30
    )

    parking_component = (
        parking.astype(str)
        .str.upper()
        .eq("Y")
        .astype(float)
        * 10.0
    )

    restroom_component = (
        restroom.astype(str)
        .str.upper()
        .eq("Y")
        .astype(float)
        * 10.0
    )

    return (
        50.0
        + store_component
        + parking_component
        + restroom_component
    ).clip(
        lower=0,
        upper=100,
    )


def load_market_places(
    base_dir: Path,
    region_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    market = read_csv_safely(
        resolve_file(
            base_dir,
            TRADITIONAL_MARKET_NAME,
        )
    )

    facility = read_csv_safely(
        resolve_file(
            base_dir,
            TRADITIONAL_MARKET_FACILITY_NAME,
        )
    )

    place = read_csv_safely(
        resolve_file(
            base_dir,
            PLACE_MARKET_NAME,
        )
    )

    require_columns(
        market,
        {
            "market_id",
            "시장명",
            "시장유형",
            "소재지도로명주소",
            "소재지지번주소",
            "위도",
            "경도",
            "점포수",
            "취급품목",
        },
        TRADITIONAL_MARKET_NAME,
    )

    require_columns(
        facility,
        {
            "market_id",
            "공중화장실보유여부",
            "주차장보유여부",
        },
        TRADITIONAL_MARKET_FACILITY_NAME,
    )

    require_columns(
        place,
        {
            "place_id",
            "place_type",
            "name",
            "latitude",
            "longitude",
            "road_address",
            "jibun_address",
        },
        PLACE_MARKET_NAME,
    )

    assert_unique(
        market,
        "market_id",
        TRADITIONAL_MARKET_NAME,
    )
    assert_unique(
        facility,
        "market_id",
        TRADITIONAL_MARKET_FACILITY_NAME,
    )
    assert_unique(
        place,
        "place_id",
        PLACE_MARKET_NAME,
    )

    merged = (
        market
        .merge(
            facility,
            on="market_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            place,
            left_on="market_id",
            right_on="place_id",
            how="inner",
            suffixes=(
                "_market",
                "_place",
            ),
            validate="one_to_one",
        )
    )

    qa_rows = []

    expected = len(market)

    qa_rows.append(
        {
            "데이터구분": "전통시장",
            "검사항목": "JOIN_행수",
            "결과": len(merged),
            "기대값": expected,
            "상태":
                (
                    "정상"
                    if len(merged) == expected
                    else "확인필요"
                ),
        }
    )

    lat_mismatch = (
        (
            to_numeric(
                merged["위도"]
            )
            - to_numeric(
                merged["latitude"]
            )
        ).abs()
        > 1e-7
    ).sum()

    lon_mismatch = (
        (
            to_numeric(
                merged["경도"]
            )
            - to_numeric(
                merged["longitude"]
            )
        ).abs()
        > 1e-7
    ).sum()

    qa_rows.extend(
        [
            {
                "데이터구분": "전통시장",
                "검사항목":
                    "위도_테이블간불일치",
                "결과": int(lat_mismatch),
                "기대값": 0,
                "상태":
                    (
                        "정상"
                        if lat_mismatch == 0
                        else "확인필요"
                    ),
            },
            {
                "데이터구분": "전통시장",
                "검사항목":
                    "경도_테이블간불일치",
                "결과": int(lon_mismatch),
                "기대값": 0,
                "상태":
                    (
                        "정상"
                        if lon_mismatch == 0
                        else "확인필요"
                    ),
            },
        ]
    )

    valid_region_keys = set(
        zip(
            region_df["시도"].astype(str),
            region_df["시군구"].astype(str),
        )
    )

    road = merged[
        "소재지도로명주소"
    ].fillna("")

    jibun = merged[
        "소재지지번주소"
    ].fillna("")

    address = road.where(
        road.str.strip().ne(""),
        jibun,
    )

    parsed = [
        parse_region_from_address(
            addr,
            valid_region_keys,
        )
        for addr in address
    ]

    merged["시도"] = [
        x[0]
        for x in parsed
    ]

    merged["시군구"] = [
        x[1]
        for x in parsed
    ]

    merged["지역매칭상태"] = [
        x[2]
        for x in parsed
    ]

    merged["시장경로점수"] = (
        market_route_score(
            merged["점포수"],
            merged["주차장보유여부"],
            merged["공중화장실보유여부"],
        )
    )

    info = (
        "시장유형="
        + merged["시장유형"].fillna("").astype(str)
        + "; 점포수="
        + merged["점포수"].fillna("").astype(str)
        + "; 주차장="
        + merged[
            "주차장보유여부"
        ].fillna("").astype(str)
        + "; 화장실="
        + merged[
            "공중화장실보유여부"
        ].fillna("").astype(str)
        + "; 취급품목="
        + merged["취급품목"].fillna("").astype(str)
    )

    out = pd.DataFrame(
        {
            "place_id":
                merged["market_id"].astype(str),
            "장소명":
                merged["시장명"].astype(str),
            "추천카테고리":
                "전통시장",
            "세부분류":
                merged["시장유형"].fillna(
                    "전통시장"
                ).astype(str),
            "시도":
                merged["시도"],
            "시군구":
                merged["시군구"],
            "현지인순위":
                np.nan,
            "외지인순위":
                np.nan,
            "현지인점수":
                np.nan,
            "외지인점수":
                np.nan,
            "장소추천점수":
                merged["시장경로점수"],
            "원본추천순위":
                np.nan,
            "위도":
                to_numeric(
                    merged["위도"]
                ),
            "경도":
                to_numeric(
                    merged["경도"]
                ),
            "주소":
                address,
            "부가정보":
                info,
            "원본구분":
                "전통시장",
        }
    )

    # 지역 매칭 상태를 출력에도 보존
    out["지역매칭상태"] = merged[
        "지역매칭상태"
    ].values

    parse_failed_count = int(
        (
            merged["지역매칭상태"]
            == "parse_failed"
        ).sum()
    )

    unsupported_count = int(
        (
            merged["지역매칭상태"]
            == "region_score_unsupported"
        ).sum()
    )

    qa_rows.extend(
        [
            {
                "데이터구분": "전통시장",
                "검사항목": "주소파싱실패",
                "결과": parse_failed_count,
                "기대값": 0,
                "상태":
                    (
                        "정상"
                        if parse_failed_count == 0
                        else "확인필요"
                    ),
            },
            {
                "데이터구분": "전통시장",
                "검사항목": "지역점수미지원",
                "결과": unsupported_count,
                "기대값": "정보",
                "상태": "정보",
            },
        ]
    )

    # 실제 경로 추천에는 로컬발견가능성 지역키가 존재하는 시장만 사용.
    # 단, 미지원 지역과 주소 파싱 오류는 QA에서 구분된다.
    supported_mask = pd.Series(
        [
            (str(sido), str(sigungu))
            in valid_region_keys
            for sido, sigungu in zip(
                out["시도"],
                out["시군구"],
            )
        ],
        index=out.index,
    )

    out = out[
        supported_mask
        & out["위도"].between(
            32,
            39.5,
        )
        & out["경도"].between(
            124,
            132.5,
        )
    ].copy()

    # 지역점수 미지원 시장은 데이터 오류가 아니라 현재 지역추천 범위 밖이므로
    # '최종사용가능행'과 분리해서 평가한다.
    accounted = (
        len(out)
        + unsupported_count
        + parse_failed_count
    )

    qa_rows.append(
        {
            "데이터구분": "전통시장",
            "검사항목":
                "지역점수지원_사용가능행",
            "결과": len(out),
            "기대값":
                expected
                - unsupported_count
                - parse_failed_count,
            "상태":
                (
                    "정상"
                    if accounted == expected
                    else "확인필요"
                ),
        }
    )

    qa_rows.append(
        {
            "데이터구분": "전통시장",
            "검사항목":
                "전체행수_설명가능여부",
            "결과": accounted,
            "기대값": expected,
            "상태":
                (
                    "정상"
                    if accounted == expected
                    else "확인필요"
                ),
        }
    )

    return (
        out.reset_index(drop=True),
        pd.DataFrame(qa_rows),
    )


# ============================================================
# 6. 관광지 데이터
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


def extract_sigungu_from_source(
    filename: str,
) -> Optional[str]:

    text = str(filename)

    m = re.search(
        r"_([^_]+)_\d{6}-\d{6}_",
        text,
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

    for prefix, standard in (
        SIDO_PREFIX_MAP.items()
    ):
        if source.startswith(prefix):
            return standard

    for prefix, standard in (
        ADDRESS_SIDO_MAP.items()
    ):
        if address.startswith(prefix):
            return standard

    return None


def audit_raw_tour_score(
    base_dir: Path,
    final_score: pd.DataFrame,
    geo: pd.DataFrame,
) -> pd.DataFrame:
    """
    인기관광지_전국통합_점수(2).csv를 최종 라우팅에 넣지 않고,
    최종 데이터와 비교하는 QA 자료로 사용.
    """
    rows = []

    try:
        raw_path = resolve_file(
            base_dir,
            TOUR_SCORE_AUDIT_NAME,
        )
    except FileNotFoundError:
        return pd.DataFrame(
            [
                {
                    "데이터구분": "관광지_RAW",
                    "검사항목": "파일존재",
                    "결과": 0,
                    "기대값": 1,
                    "상태": "선택파일없음",
                }
            ]
        )

    raw = read_csv_safely(
        raw_path
    )

    require_columns(
        raw,
        {
            "관광지ID",
            "추천점수",
        },
        TOUR_SCORE_AUDIT_NAME,
    )

    raw_dup_rows = int(
        raw["관광지ID"]
        .duplicated()
        .sum()
    )

    raw_dup_ids = int(
        raw.loc[
            raw["관광지ID"]
            .duplicated(keep=False),
            "관광지ID",
        ].nunique()
    )

    raw_ids = set(
        raw["관광지ID"].astype(str)
    )

    geo_ids = set(
        geo["관광지ID"].astype(str)
    )

    final_ids = set(
        final_score[
            "관광지ID"
        ].astype(str)
    )

    rows.extend(
        [
            {
                "데이터구분": "관광지_RAW",
                "검사항목": "전체행수",
                "결과": len(raw),
                "기대값": "",
                "상태": "정보",
            },
            {
                "데이터구분": "관광지_RAW",
                "검사항목": "중복ID수",
                "결과": raw_dup_ids,
                "기대값": 0,
                "상태":
                    (
                        "정상"
                        if raw_dup_ids == 0
                        else "최종입력사용금지"
                    ),
            },
            {
                "데이터구분": "관광지_RAW",
                "검사항목": "중복추가행수",
                "결과": raw_dup_rows,
                "기대값": 0,
                "상태":
                    (
                        "정상"
                        if raw_dup_rows == 0
                        else "최종입력사용금지"
                    ),
            },
            {
                "데이터구분": "관광지_RAW",
                "검사항목": "좌표파일미매칭ID수",
                "결과":
                    len(
                        raw_ids
                        - geo_ids
                    ),
                "기대값": 0,
                "상태": "정보",
            },
            {
                "데이터구분": "관광지_RAW",
                "검사항목":
                    "최종점수파일에없는RAW_ID수",
                "결과":
                    len(
                        raw_ids
                        - final_ids
                    ),
                "기대값": 0,
                "상태": "정보",
            },
        ]
    )

    return pd.DataFrame(rows)


def load_tour_places(
    base_dir: Path,
    region_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    score = read_csv_safely(
        resolve_file(
            base_dir,
            TOUR_SCORE_FINAL_NAME,
        )
    )

    geo = read_csv_safely(
        resolve_file(
            base_dir,
            TOUR_GEO_FINAL_NAME,
        )
    )

    require_columns(
        score,
        {
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
        },
        TOUR_SCORE_FINAL_NAME,
    )

    require_columns(
        geo,
        {
            "관광지ID",
            "관광지명",
            "위도",
            "경도",
            "주소",
        },
        TOUR_GEO_FINAL_NAME,
    )

    assert_unique(
        score,
        "관광지ID",
        TOUR_SCORE_FINAL_NAME,
    )

    qa_rows = []

    geo["위도"] = to_numeric(
        geo["위도"]
    )
    geo["경도"] = to_numeric(
        geo["경도"]
    )

    duplicate_ids = set(
        geo.loc[
            geo["관광지ID"]
            .duplicated(keep=False),
            "관광지ID",
        ].astype(str)
    )

    bad_geo_ids = set()

    for pid in sorted(
        duplicate_ids
    ):
        g = geo[
            geo["관광지ID"]
            .astype(str)
            == pid
        ]

        unique_coords = (
            g[
                [
                    "위도",
                    "경도",
                ]
            ]
            .drop_duplicates()
        )

        if len(unique_coords) > 1:
            bad_geo_ids.add(pid)

            qa_rows.append(
                {
                    "데이터구분": "관광지",
                    "검사항목":
                        "동일ID_서로다른좌표",
                    "결과": pid,
                    "기대값": "중복없음",
                    "상태": "제외",
                }
            )

    geo_clean = (
        geo[
            ~geo["관광지ID"]
            .astype(str)
            .isin(bad_geo_ids)
        ]
        .drop_duplicates(
            subset=["관광지ID"],
            keep="first",
        )
    )

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

    missing_geo = merged[
        merged["위도"].isna()
        | merged["경도"].isna()
    ]

    qa_rows.append(
        {
            "데이터구분": "관광지",
            "검사항목": "좌표없음",
            "결과": len(missing_geo),
            "기대값": 0,
            "상태":
                (
                    "정상"
                    if len(missing_geo) == 0
                    else "제외"
                ),
        }
    )

    merged["시군구"] = (
        merged["원본파일"]
        .apply(
            extract_sigungu_from_source
        )
    )

    merged["시도"] = [
        infer_sido_from_tour_row(
            source,
            address,
        )
        for source, address
        in zip(
            merged["원본파일"],
            merged["주소"].fillna(""),
        )
    ]

    valid_region_keys = set(
        zip(
            region_df["시도"].astype(str),
            region_df["시군구"].astype(str),
        )
    )

    sigungu_to_sidos = (
        region_df
        .groupby("시군구")["시도"]
        .agg(
            lambda x:
                list(
                    pd.unique(x)
                )
        )
        .to_dict()
    )

    for idx in merged.index[
        merged["시도"].isna()
    ]:
        sg = merged.at[
            idx,
            "시군구",
        ]

        sidos = sigungu_to_sidos.get(
            sg,
            [],
        )

        if len(sidos) == 1:
            merged.at[
                idx,
                "시도",
            ] = sidos[0]

    valid_mask = pd.Series(
        [
            (
                str(sido),
                str(sigungu),
            )
            in valid_region_keys
            for sido, sigungu
            in zip(
                merged["시도"],
                merged["시군구"],
            )
        ],
        index=merged.index,
    )

    invalid_region_count = int(
        (~valid_mask).sum()
    )

    qa_rows.append(
        {
            "데이터구분": "관광지",
            "검사항목": "지역매칭실패",
            "결과": invalid_region_count,
            "기대값": 0,
            "상태":
                (
                    "정상"
                    if invalid_region_count == 0
                    else "제외"
                ),
        }
    )

    merged["추천점수"] = to_numeric(
        merged["추천점수"]
    )

    usable = merged[
        valid_mask
        & merged["위도"].between(
            32,
            39.5,
        )
        & merged["경도"].between(
            124,
            132.5,
        )
        & merged["추천점수"].notna()
        & ~merged["분류"].isin(
            EXCLUDED_TOUR_CATEGORIES
        )
    ].copy()

    out = pd.DataFrame(
        {
            "place_id":
                usable[
                    "관광지ID"
                ].astype(str),
            "장소명":
                usable[
                    "관광지명"
                ].astype(str),
            "추천카테고리":
                "관광지",
            "세부분류":
                usable[
                    "분류"
                ].astype(str),
            "시도":
                usable[
                    "시도"
                ].astype(str),
            "시군구":
                usable[
                    "시군구"
                ].astype(str),
            "현지인순위":
                to_numeric(
                    usable[
                        "현지인순위"
                    ]
                ),
            "외지인순위":
                to_numeric(
                    usable[
                        "외지인순위"
                    ]
                ),
            "현지인점수":
                to_numeric(
                    usable[
                        "현지인점수"
                    ]
                ),
            "외지인점수":
                to_numeric(
                    usable[
                        "외지인점수"
                    ]
                ),
            "장소추천점수":
                usable[
                    "추천점수"
                ],
            "원본추천순위":
                to_numeric(
                    usable[
                        "전국추천순위"
                    ]
                ),
            "위도":
                usable[
                    "위도"
                ],
            "경도":
                usable[
                    "경도"
                ],
            "주소":
                usable[
                    "주소"
                ].fillna(""),
            "부가정보":
                "",
            "원본구분":
                "관광지",
            "지역매칭상태":
                "원본파일기반",
        }
    )

    qa_rows.append(
        {
            "데이터구분": "관광지",
            "검사항목": "최종사용가능행",
            "결과": len(out),
            "기대값": len(score),
            "상태": "정보",
        }
    )

    qa = pd.concat(
        [
            pd.DataFrame(
                qa_rows
            ),
            audit_raw_tour_score(
                base_dir,
                score,
                geo_clean,
            ),
        ],
        ignore_index=True,
    )

    return (
        out.reset_index(drop=True),
        qa,
    )


# ============================================================
# 7. 전체 장소 통합
# ============================================================

def build_all_places(
    base_dir: Path,
    region_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    restaurant, restaurant_qa = (
        load_restaurant_places(
            base_dir
        )
    )

    market, market_qa = (
        load_market_places(
            base_dir,
            region_df,
        )
    )

    tour, tour_qa = (
        load_tour_places(
            base_dir,
            region_df,
        )
    )

    all_places = pd.concat(
        [
            restaurant,
            market,
            tour,
        ],
        ignore_index=True,
        sort=False,
    )

    all_places["internal_id"] = (
        all_places["원본구분"]
        + "_"
        + all_places[
            "place_id"
        ].astype(str)
    )

    qa = pd.concat(
        [
            restaurant_qa,
            market_qa,
            tour_qa,
        ],
        ignore_index=True,
        sort=False,
    )

    return (
        all_places,
        qa,
    )



def normalize_place_name_for_match(
    name: str,
) -> str:
    """
    시장명 중복 비교용 이름 정규화.
    원본 장소명은 변경하지 않는다.
    """
    text = str(name).lower().strip()

    # 괄호 안 별칭 제거
    text = re.sub(r"\([^)]*\)", "", text)

    # 공백/특수기호 제거
    text = re.sub(
        r"[^0-9a-z가-힣]",
        "",
        text,
    )

    # 시장 데이터에서 흔한 수식어 제거
    suffixes = [
        "임시시장",
        "전통시장",
        "공설시장",
        "상설시장",
        "종합시장",
        "시장상점가",
        "시장",
    ]

    for suffix in suffixes:
        text = text.replace(
            suffix,
            "",
        )

    return text


def is_same_market_place(
    tour_row: pd.Series,
    market_row: pd.Series,
) -> bool:
    """
    관광지 분류 '시장'과 전통시장 DB가 동일 장소인지 판단.
    이름 유사도 + 600m 이내 거리 조건을 함께 사용한다.
    """
    if (
        tour_row["추천카테고리"] != "관광지"
        or "시장" not in str(
            tour_row["세부분류"]
        )
    ):
        return False

    a = normalize_place_name_for_match(
        tour_row["장소명"]
    )
    b = normalize_place_name_for_match(
        market_row["장소명"]
    )

    if not a or not b:
        return False

    ratio = difflib.SequenceMatcher(
        None,
        a,
        b,
    ).ratio()

    substring_match = (
        a in b
        or b in a
    )

    if (
        ratio < MARKET_NAME_SIMILARITY
        and not substring_match
    ):
        return False

    km = haversine_km(
        tour_row["위도"],
        tour_row["경도"],
        market_row["위도"],
        market_row["경도"],
    )

    return (
        km
        <= MARKET_DUPLICATE_DISTANCE_KM
    )


def remove_market_tour_duplicates(
    x: pd.DataFrame,
    selected_categories: list[str],
) -> tuple[pd.DataFrame, list[dict]]:
    """
    사용자가 관광지와 전통시장을 동시에 선택한 경우,
    관광지 데이터에 들어있는 '시장'과 전통시장 DB의 동일 장소 중복을 제거한다.

    전통시장 DB를 더 구체적인 원천으로 보고 관광지 쪽 중복 행을 제거한다.
    """
    if not (
        "관광지" in selected_categories
        and "전통시장" in selected_categories
    ):
        return x, []

    tours = x[
        (x["추천카테고리"] == "관광지")
        & x["세부분류"].astype(str).str.contains(
            "시장",
            na=False,
        )
    ]

    markets = x[
        x["추천카테고리"] == "전통시장"
    ]

    drop_indices = []
    records = []

    for tidx, trow in tours.iterrows():
        for midx, mrow in markets.iterrows():
            if is_same_market_place(
                trow,
                mrow,
            ):
                drop_indices.append(
                    tidx
                )

                records.append(
                    {
                        "제거된관광지":
                            trow["장소명"],
                        "유지된전통시장":
                            mrow["장소명"],
                        "거리km":
                            round(
                                haversine_km(
                                    trow["위도"],
                                    trow["경도"],
                                    mrow["위도"],
                                    mrow["경도"],
                                ),
                                3,
                            ),
                    }
                )
                break

    if drop_indices:
        x = x.drop(
            index=drop_indices
        )

    return (
        x.reset_index(
            drop=True
        ),
        records,
    )


def general_tour_type_adjustment(
    row: pd.Series,
) -> float:
    """
    일반 관광 기본 모드의 세부분류 보정.

    자연/문화/생태/역사/경관 계열은 소폭 우대,
    레저/스포츠/웰니스/골프/스파 계열은 사용자의 명시적 테마 선택이
    없는 기본 일정에서 과도하게 선택되지 않도록 감점한다.

    원래 관광지 추천점수는 변경하지 않는다.
    """
    if row["추천카테고리"] != "관광지":
        return 0.0

    text = (
        str(row["세부분류"])
        + " "
        + str(row["장소명"])
    ).lower()

    special = [
        "레저",
        "스포츠",
        "웰니스",
        "골프",
        "스파",
    ]

    general = [
        "자연",
        "생태",
        "문화",
        "역사",
        "경관",
        "공원",
        "해수욕",
        "항",
        "포구",
        "전망",
        "박물",
        "미술",
    ]

    if any(
        k in text
        for k in special
    ):
        return (
            GENERAL_TOUR_SPECIAL_PENALTY
        )

    if any(
        k in text
        for k in general
    ):
        return (
            GENERAL_TOUR_POSITIVE_BONUS
        )

    return 0.0


def add_route_selection_score(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """
    서로 다른 카테고리의 원점수 척도가 다르므로
    경로 최적화용 공통점수를 별도로 생성한다.

    - 원래 '장소추천점수'는 그대로 보존
    - 카테고리내순위 1위 = 100
    - 이후 순위마다 5점 감소, 최저 55
    - 관광지에만 일반 관광 유형 보정 적용
    """
    x = candidates.copy()

    x["관광유형보정점수"] = x.apply(
        general_tour_type_adjustment,
        axis=1,
    )

    x["경로선택기본점수"] = (
        100.0
        - (
            x["카테고리내순위"]
            - 1
        )
        * ROUTE_RANK_STEP
    ).clip(
        lower=ROUTE_SCORE_FLOOR,
        upper=100.0,
    )

    x["경로선택점수"] = (
        x["경로선택기본점수"]
        + x["관광유형보정점수"]
    ).clip(
        lower=0.0,
        upper=100.0,
    )

    return x


# ============================================================
# 8. 지역별 장소 후보
# ============================================================

def get_place_candidates(
    all_places: pd.DataFrame,
    selected_sido: str,
    selected_sigungu: str,
    selected_categories: list[str],
) -> tuple[pd.DataFrame, list[dict]]:

    invalid = (
        set(selected_categories)
        - VALID_CATEGORIES
    )

    if invalid:
        raise ValueError(
            "지원하지 않는 카테고리: "
            f"{sorted(invalid)}"
        )

    x = all_places[
        (
            all_places["시도"]
            == selected_sido
        )
        & (
            all_places["시군구"]
            == selected_sigungu
        )
        & all_places[
            "추천카테고리"
        ].isin(
            selected_categories
        )
    ].copy()

    if x.empty:
        raise ValueError(
            f"{selected_sido} "
            f"{selected_sigungu}에서 "
            f"{selected_categories} "
            "후보를 찾지 못했습니다."
        )

    # 전통시장 카테고리를 별도로 선택한 경우,
    # 관광지 데이터의 '시장' 분류는 전통시장 DB와 의미가 겹치므로 제외한다.
    # 전통시장 DB를 시장 카테고리의 단일 기준(source of truth)으로 사용한다.
    semantic_market_records = []

    if "전통시장" in selected_categories:
        semantic_market_mask = (
            (x["추천카테고리"] == "관광지")
            & x["세부분류"].astype(str).str.contains(
                "시장",
                na=False,
            )
        )

        for _, row in x[
            semantic_market_mask
        ].iterrows():
            semantic_market_records.append(
                {
                    "제거된관광지":
                        row["장소명"],
                    "유지된전통시장":
                        "전통시장 DB 사용",
                    "거리km":
                        None,
                    "제거사유":
                        "전통시장 카테고리 선택 시 관광지 시장분류 의미중복 제거",
                }
            )

        x = x[
            ~semantic_market_mask
        ].copy()

    # 남아있는 관광지 시장과 전통시장 간 실제 동일 장소 중복도 추가 제거
    x, exact_duplicate_records = (
        remove_market_tour_duplicates(
            x,
            selected_categories,
        )
    )

    duplicate_records = (
        semantic_market_records
        + exact_duplicate_records
    )

    # 원래 점수는 해당 카테고리 내부 후보 선정에만 사용
    x["카테고리내순위"] = (
        x.groupby(
            "추천카테고리"
        )["장소추천점수"]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    x = x[
        x["카테고리내순위"]
        <= TOP_CANDIDATES_PER_CATEGORY
    ].copy()

    # 실제 경로 최적화에는 공통 척도의 별도 점수를 사용
    x = add_route_selection_score(
        x
    )

    return (
        x.sort_values(
            [
                "추천카테고리",
                "카테고리내순위",
            ]
        ).reset_index(
            drop=True
        ),
        duplicate_records,
    )


def infer_region_center(
    candidates: pd.DataFrame,
) -> tuple[float, float]:

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

    if (
        math.isnan(lat)
        or math.isnan(lon)
    ):
        raise ValueError(
            "지역 중심점 계산 실패"
        )

    return lat, lon


# ============================================================
# 9. 이동시간
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
        self.api_key = (
            api_key.strip()
        )
        self.cache_file = (
            cache_file
        )

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
        straight_km = (
            haversine_km(
                lat1,
                lon1,
                lat2,
                lon2,
            )
        )

        road_km = (
            straight_km
            * 1.25
        )

        minutes = (
            road_km
            / 30.0
            * 60.0
            + 3.0
        )

        return {
            "distance_km":
                round(
                    road_km,
                    3,
                ),
            "minutes":
                round(
                    max(
                        2.0,
                        minutes,
                    ),
                    2,
                ),
            "source":
                "거리기반추정",
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

        if (
            self.api_key
            and requests
            is not None
        ):
            try:
                response = requests.get(
                    self.KAKAO_URL,
                    headers={
                        "Authorization":
                            f"KakaoAK "
                            f"{self.api_key}",
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
                payload = (
                    response.json()
                )

                routes = payload.get(
                    "routes",
                    [],
                )

                if routes:
                    summary = (
                        routes[0]
                        .get(
                            "summary",
                            {},
                        )
                    )

                    if (
                        "distance"
                        in summary
                        and "duration"
                        in summary
                    ):
                        result = {
                            "distance_km":
                                summary[
                                    "distance"
                                ]
                                / 1000.0,
                            "minutes":
                                summary[
                                    "duration"
                                ]
                                / 1000.0
                                / 60.0,
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
# 10. 시간/경로 평가
# ============================================================

def parse_hhmm(
    value: str,
) -> int:

    hh, mm = map(
        int,
        str(value).split(":"),
    )

    if not (
        0 <= hh <= 23
        and 0 <= mm <= 59
    ):
        raise ValueError(
            f"잘못된 시간: {value}"
        )

    return (
        hh * 60
        + mm
    )


def minutes_to_hhmm(
    minutes: float,
) -> str:

    total = int(
        round(minutes)
    )

    day = (
        total
        // (24 * 60)
    )

    total %= (
        24 * 60
    )

    hh = total // 60
    mm = total % 60

    if day:
        return (
            f"+{day}일 "
            f"{hh:02d}:"
            f"{mm:02d}"
        )

    return (
        f"{hh:02d}:"
        f"{mm:02d}"
    )


def in_window(
    value: float,
    window: tuple[int, int],
) -> bool:
    return (
        window[0]
        <= value
        <= window[1]
    )


def evaluate_time_fit(
    category: str,
    arrival: float,
) -> float:

    if category == "맛집":

        if in_window(
            arrival,
            LUNCH_WINDOW,
        ):
            return 28.0

        if in_window(
            arrival,
            DINNER_WINDOW,
        ):
            return 24.0

        if arrival < 11 * 60:
            return -32.0

        if (
            14 * 60
            < arrival
            < 17 * 60 + 30
        ):
            return -12.0

        return -8.0

    if (
        category
        == "카페/베이커리"
    ):

        if in_window(
            arrival,
            CAFE_WINDOW,
        ):
            return 18.0

        if arrival < 11 * 60:
            return -12.0

        return 4.0

    if category == "전통시장":

        if in_window(
            arrival,
            MARKET_WINDOW,
        ):
            return 10.0

        return -8.0

    if category == "관광지":

        if (
            9 * 60
            <= arrival
            <= 17 * 60 + 30
        ):
            return 8.0

        return -5.0

    return 0.0


def category_quota_bonus(
    route_categories: list[str],
    selected_categories: list[str],
) -> float:

    covered = len(
        set(route_categories)
        & set(selected_categories)
    )

    return (
        covered
        * 20.0
    )


def get_category_limit(
    category: str,
    selected_categories: list[str],
) -> int:

    if (
        len(
            set(
                selected_categories
            )
        )
        <= 1
    ):
        return 99

    return (
        MULTI_CATEGORY_MAX_VISITS
        .get(
            category,
            99,
        )
    )


def violates_category_limit(
    route: list[int],
    candidates: pd.DataFrame,
    selected_categories: list[str],
) -> bool:

    categories = [
        candidates.loc[
            idx,
            "추천카테고리",
        ]
        for idx in route
    ]

    for category in set(
        categories
    ):
        if (
            categories.count(
                category
            )
            > get_category_limit(
                category,
                selected_categories,
            )
        ):
            return True

    return False


def near_duplicate_tourist_stop(
    new_idx: int,
    route: list[int],
    candidates: pd.DataFrame,
    threshold_km: float
    = NEARBY_TOUR_CLUSTER_KM,
) -> bool:

    new_row = candidates.loc[
        new_idx
    ]

    if (
        new_row["추천카테고리"]
        != "관광지"
    ):
        return False

    for idx in route:

        old_row = candidates.loc[
            idx
        ]

        if (
            old_row[
                "추천카테고리"
            ]
            != "관광지"
        ):
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

    subs = []

    for idx in route:
        row = candidates.loc[
            idx
        ]

        if (
            row[
                "추천카테고리"
            ]
            == "관광지"
        ):
            subs.append(
                str(
                    row[
                        "세부분류"
                    ]
                )
            )

    repeats = (
        len(subs)
        - len(
            set(subs)
        )
    )

    return (
        -repeats
        * SAME_TOUR_SUBCATEGORY_PENALTY
    )


def natural_flow_score(
    schedule: list[dict],
) -> float:

    if not schedule:
        return 0.0

    score = sum(
        stop[
            "time_fit_score"
        ]
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

            elif (
                b
                == "카페/베이커리"
            ):
                score -= 60.0

            elif b == "전통시장":
                score -= 50.0

            else:
                score -= 10.0

        else:
            score += 5.0

            if (
                a == "관광지"
                and b
                in {
                    "맛집",
                    "카페/베이커리",
                    "전통시장",
                }
            ):
                score += 4.0

            if (
                a == "맛집"
                and b
                in {
                    "관광지",
                    "카페/베이커리",
                    "전통시장",
                }
            ):
                score += 4.0

    return score


def simulate_route(
    route: list[int],
    candidates: pd.DataFrame,
    provider: TravelTimeProvider,
    start_lat: float,
    start_lon: float,
    start_minute: int,
) -> tuple[
    list[dict],
    float,
    float,
    float,
]:

    schedule = []

    current_minute = float(
        start_minute
    )

    prev_lat = start_lat
    prev_lon = start_lon

    total_place_score = 0.0
    total_travel_min = 0.0
    total_distance_km = 0.0

    for idx in route:

        row = candidates.loc[
            idx
        ]

        travel = provider.get(
            prev_lat,
            prev_lon,
            row["위도"],
            row["경도"],
        )

        travel_min = float(
            travel["minutes"]
        )

        distance_km = float(
            travel["distance_km"]
        )

        arrival = (
            current_minute
            + travel_min
        )

        category = (
            row[
                "추천카테고리"
            ]
        )

        stay = (
            STAY_MINUTES[
                category
            ]
        )

        depart = (
            arrival
            + stay
        )

        schedule.append(
            {
                "idx": idx,
                "category":
                    category,
                "arrival":
                    arrival,
                "depart":
                    depart,
                "stay":
                    stay,
                "travel_min":
                    travel_min,
                "distance_km":
                    distance_km,
                "travel_source":
                    travel[
                        "source"
                    ],
                "time_fit_score":
                    evaluate_time_fit(
                        category,
                        arrival,
                    ),
            }
        )

        current_minute = (
            depart
        )

        total_place_score += float(
            row[
                "경로선택점수"
            ]
        )

        total_travel_min += (
            travel_min
        )

        total_distance_km += (
            distance_km
        )

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
    start_lat: float,
    start_lon: float,
    start_minute: int,
) -> float:

    schedule, _, _, _ = (
        simulate_route(
            route,
            candidates,
            provider,
            start_lat,
            start_lon,
            start_minute,
        )
    )

    if not schedule:
        return 0.0

    return (
        schedule[-1][
            "depart"
        ]
        - start_minute
    )


# ============================================================
# 11. 경로 최적화
# ============================================================

def optimize_route(
    candidates: pd.DataFrame,
    selected_categories: list[str],
    budget_min: float,
    max_stops: int,
    target_stops: int,
    provider: TravelTimeProvider,
    start_lat: float,
    start_lon: float,
    start_time: str
    = DEFAULT_START_TIME,
) -> list[int]:

    candidates = (
        candidates
        .reset_index(
            drop=True
        )
    )

    if candidates.empty:
        return []

    start_minute = (
        parse_hhmm(
            start_time
        )
    )

    beam = [
        ([], 0.0)
    ]

    finished = []

    for _ in range(
        max_stops
    ):

        next_states = []

        for route, _ in beam:

            used = set(
                route
            )

            for idx, _row in (
                candidates.iterrows()
            ):

                idx = int(idx)

                if idx in used:
                    continue

                if (
                    near_duplicate_tourist_stop(
                        idx,
                        route,
                        candidates,
                    )
                ):
                    continue

                new_route = (
                    route
                    + [idx]
                )

                if (
                    violates_category_limit(
                        new_route,
                        candidates,
                        selected_categories,
                    )
                ):
                    continue

                total_minutes = (
                    route_total_minutes(
                        new_route,
                        candidates,
                        provider,
                        start_lat,
                        start_lon,
                        start_minute,
                    )
                )

                if (
                    total_minutes
                    > budget_min
                ):
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
                    x["category"]
                    for x in schedule
                ]

                coverage_bonus = (
                    category_quota_bonus(
                        categories,
                        selected_categories,
                    )
                )

                flow_score = (
                    natural_flow_score(
                        schedule
                    )
                )

                tour_diversity = (
                    tourist_subcategory_diversity_score(
                        new_route,
                        candidates,
                    )
                )

                travel_penalty = (
                    travel_min
                    * 0.55
                )

                missing_count = len(
                    set(
                        selected_categories
                    )
                    - set(categories)
                )

                missing_penalty = (
                    missing_count
                    * 12.0
                )

                stop_bonus = (
                    min(
                        len(
                            new_route
                        ),
                        target_stops,
                    )
                    * 2.5
                )

                objective = (
                    place_score
                    + coverage_bonus
                    + flow_score
                    + tour_diversity
                    + stop_bonus
                    - travel_penalty
                    - missing_penalty
                )

                state = (
                    new_route,
                    objective,
                )

                next_states.append(
                    state
                )

                finished.append(
                    state
                )

        if not next_states:
            break

        next_states.sort(
            key=lambda x:
                x[1],
            reverse=True,
        )

        unique = []
        seen = set()

        for route, score in (
            next_states
        ):

            signature = (
                frozenset(route),
                route[-1],
            )

            if signature in seen:
                continue

            seen.add(
                signature
            )

            unique.append(
                (
                    route,
                    score,
                )
            )

            if (
                len(unique)
                >= BEAM_WIDTH
            ):
                break

        beam = unique

    if not finished:
        return []

    def final_key(
        state,
    ):
        route, objective = state

        categories = {
            candidates.loc[
                idx,
                "추천카테고리",
            ]
            for idx in route
        }

        coverage = len(
            categories
            & set(
                selected_categories
            )
        )

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
# 12. 최종 경로 상세
# ============================================================

def build_route_detail(
    route: list[int],
    candidates: pd.DataFrame,
    provider: TravelTimeProvider,
    day: int,
    start_lat: float,
    start_lon: float,
    start_time: str
    = DEFAULT_START_TIME,
) -> pd.DataFrame:

    start_minute = (
        parse_hhmm(
            start_time
        )
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
        row = candidates.loc[
            idx
        ]

        rows.append(
            {
                "day":
                    day,
                "order":
                    order,
                "도착예정":
                    minutes_to_hhmm(
                        stop[
                            "arrival"
                        ]
                    ),
                "출발예정":
                    minutes_to_hhmm(
                        stop[
                            "depart"
                        ]
                    ),
                "place_id":
                    row[
                        "place_id"
                    ],
                "장소명":
                    row[
                        "장소명"
                    ],
                "카테고리":
                    row[
                        "추천카테고리"
                    ],
                "세부분류":
                    row[
                        "세부분류"
                    ],
                "시도":
                    row[
                        "시도"
                    ],
                "시군구":
                    row[
                        "시군구"
                    ],
                "장소추천점수":
                    round(
                        float(
                            row[
                                "장소추천점수"
                            ]
                        ),
                        2,
                    ),
                "경로선택점수":
                    round(
                        float(
                            row[
                                "경로선택점수"
                            ]
                        ),
                        2,
                    ),
                "관광유형보정점수":
                    round(
                        float(
                            row[
                                "관광유형보정점수"
                            ]
                        ),
                        2,
                    ),
                "원본추천순위":
                    row[
                        "원본추천순위"
                    ],
                "현지인순위":
                    row[
                        "현지인순위"
                    ],
                "외지인순위":
                    row[
                        "외지인순위"
                    ],
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
                            stop[
                                "travel_min"
                            ]
                        ),
                        1,
                    ),
                "이전장소에서_이동km":
                    round(
                        float(
                            stop[
                                "distance_km"
                            ]
                        ),
                        2,
                    ),
                "이동시간출처":
                    stop[
                        "travel_source"
                    ],
                "체류시간분":
                    stop[
                        "stay"
                    ],
                "누적소요시간분":
                    round(
                        stop[
                            "depart"
                        ]
                        - start_minute,
                        1,
                    ),
                "위도":
                    row[
                        "위도"
                    ],
                "경도":
                    row[
                        "경도"
                    ],
                "주소":
                    row[
                        "주소"
                    ],
                "부가정보":
                    row[
                        "부가정보"
                    ],
                "원본구분":
                    row[
                        "원본구분"
                    ],
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 13. 1박2일
# ============================================================

def optimize_1n2d(
    candidates: pd.DataFrame,
    selected_categories: list[str],
    provider: TravelTimeProvider,
    start_lat: float,
    start_lon: float,
    start_time: str
    = DEFAULT_START_TIME,
) -> tuple[pd.DataFrame, dict]:

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
            "1일차 경로 생성 실패"
        )

    detail1 = (
        build_route_detail(
            route1,
            candidates,
            provider,
            day=1,
            start_lat=start_lat,
            start_lon=start_lon,
            start_time=start_time,
        )
    )

    used_internal_ids = {
        candidates.loc[
            idx,
            "internal_id",
        ]
        for idx in route1
    }

    remaining = (
        candidates[
            ~candidates[
                "internal_id"
            ].isin(
                used_internal_ids
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    last = candidates.loc[
        route1[-1]
    ]

    day2_lat = float(
        last["위도"]
    )
    day2_lon = float(
        last["경도"]
    )

    route2 = optimize_route(
        remaining,
        selected_categories,
        budget_min=480,
        max_stops=6,
        target_stops=5,
        provider=provider,
        start_lat=day2_lat,
        start_lon=day2_lon,
        start_time=start_time,
    )

    if route2:
        detail2 = (
            build_route_detail(
                route2,
                remaining,
                provider,
                day=2,
                start_lat=day2_lat,
                start_lon=day2_lon,
                start_time=start_time,
            )
        )

        detail = pd.concat(
            [
                detail1,
                detail2,
            ],
            ignore_index=True,
        )

    else:
        detail = detail1

    return (
        detail,
        {
            "1일차장소수":
                len(route1),
            "2일차장소수":
                len(route2),
            "숙박가정":
                (
                    "1일차 마지막 장소 "
                    f"'{last['장소명']}' "
                    "인근 숙박"
                ),
        },
    )


# ============================================================
# 14. 메인 추천 함수
# ============================================================

def recommend_local_on_trip(
    selected_sido: str,
    selected_categories: list[str],
    trip_type: str = "day",
    selected_sigungu: Optional[str]
    = None,
    start_lat: Optional[float]
    = None,
    start_lon: Optional[float]
    = None,
    start_time: str
    = DEFAULT_START_TIME,
    base_dir: Path
    = BASE_DIR,
    top_region_n: int
    = 5,
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
            "2h/4h/6h/day/1n2d"
            " 중 하나여야 합니다."
        )

    selected_categories = [
        x.strip()
        for x
        in selected_categories
        if x.strip()
    ]

    invalid = (
        set(
            selected_categories
        )
        - VALID_CATEGORIES
    )

    if invalid:
        raise ValueError(
            "지원하지 않는 카테고리: "
            f"{sorted(invalid)}"
        )

    output_dir = (
        base_dir
        / OUTPUT_DIR_NAME
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1) 지역 추천
    # --------------------------------------------------------
    region_df = (
        load_region_scores(
            base_dir
        )
    )

    region_ranking = (
        get_region_ranking(
            region_df,
            selected_sido,
        )
    )

    formal = region_ranking[
        region_ranking[
            "지역추천구분"
        ]
        == "정식_12개월"
    ]

    if selected_sigungu is None:

        if not formal.empty:
            selected_sigungu = (
                formal.iloc[0][
                    "시군구"
                ]
            )
        else:
            usable = region_ranking[
                region_ranking[
                    "지역추천점수"
                ].notna()
            ]

            if usable.empty:
                raise RuntimeError(
                    "추천 가능한 지역이 없습니다."
                )

            selected_sigungu = (
                usable.iloc[0][
                    "시군구"
                ]
            )

    if not (
        region_ranking[
            "시군구"
        ]
        == selected_sigungu
    ).any():
        raise ValueError(
            f"{selected_sido}에 "
            f"'{selected_sigungu}'가 없습니다."
        )

    region_path = (
        output_dir
        / (
            f"01_{selected_sido}"
            "_지역추천순위.csv"
        )
    )

    region_ranking.to_csv(
        region_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 2) 모든 장소 데이터 생성
    # --------------------------------------------------------
    all_places, qa = (
        build_all_places(
            base_dir,
            region_df,
        )
    )

    qa_path = (
        output_dir
        / "00_통합데이터_QA.csv"
    )

    qa.to_csv(
        qa_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 3) 선택 지역 + 카테고리 후보
    # --------------------------------------------------------
    (
        candidates,
        market_duplicate_records,
    ) = get_place_candidates(
        all_places,
        selected_sido,
        selected_sigungu,
        selected_categories,
    )

    category_counts = (
        candidates[
            "추천카테고리"
        ]
        .value_counts()
        .to_dict()
    )

    candidate_path = (
        output_dir
        / (
            f"02_{selected_sido}_"
            f"{selected_sigungu}_"
            "장소후보.csv"
        )
    )

    candidates.to_csv(
        candidate_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 4) 출발점
    # --------------------------------------------------------
    auto_start_used = False

    if (
        start_lat is None
        or start_lon is None
    ):
        (
            start_lat,
            start_lon,
        ) = infer_region_center(
            candidates
        )

        auto_start_used = True

    # --------------------------------------------------------
    # 5) 이동시간 provider
    # --------------------------------------------------------
    api_key = os.getenv(
        "KAKAO_REST_API_KEY",
        "",
    )

    provider = (
        TravelTimeProvider(
            cache_file=(
                output_dir
                / "travel_time_cache.json"
            ),
            api_key=api_key,
        )
    )

    route_meta = {}

    # --------------------------------------------------------
    # 6) 경로 최적화
    # --------------------------------------------------------
    if trip_type == "1n2d":

        (
            route_detail,
            route_meta,
        ) = optimize_1n2d(
            candidates,
            selected_categories,
            provider,
            start_lat,
            start_lon,
            start_time,
        )

    else:
        cfg = (
            TRIP_CONFIG[
                trip_type
            ]
        )

        route = optimize_route(
            candidates,
            selected_categories,
            budget_min=(
                cfg[
                    "budget_min"
                ]
            ),
            max_stops=(
                cfg[
                    "max_stops"
                ]
            ),
            target_stops=(
                cfg[
                    "target_stops"
                ]
            ),
            provider=provider,
            start_lat=start_lat,
            start_lon=start_lon,
            start_time=start_time,
        )

        if not route:
            raise RuntimeError(
                "주어진 시간 안에 "
                "경로를 생성하지 못했습니다."
            )

        route_detail = (
            build_route_detail(
                route,
                candidates,
                provider,
                day=1,
                start_lat=start_lat,
                start_lon=start_lon,
                start_time=start_time,
            )
        )

    provider.save()

    actual_categories = set(
        route_detail[
            "카테고리"
        ]
    )

    available_categories = set(
        candidates[
            "추천카테고리"
        ]
    )

    expected_categories = (
        set(
            selected_categories
        )
        & available_categories
    )

    missing_categories = (
        expected_categories
        - actual_categories
    )

    route_path = (
        output_dir
        / (
            f"03_{selected_sido}_"
            f"{selected_sigungu}_"
            f"{trip_type}_"
            "추천경로.csv"
        )
    )

    route_detail.to_csv(
        route_path,
        index=False,
        encoding="utf-8-sig",
    )

    selected_region_row = (
        region_ranking[
            region_ranking[
                "시군구"
            ]
            == selected_sigungu
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
        "후보카테고리별개수":
            {
                str(k): int(v)
                for k, v
                in category_counts.items()
            },
        "시장관광지의미중복제거건수":
            len(
                market_duplicate_records
            ),
        "시장관광지의미중복제거내역":
            market_duplicate_records,
        "점수정책":
            {
                "장소추천점수":
                    "카테고리 내부 후보 선정용 원본 점수",
                "경로선택점수":
                    (
                        "카테고리내순위를 공통 100~55점 척도로 변환한 "
                        "경로 최적화용 점수"
                    ),
                "관광지유형보정":
                    (
                        "일반 관광에서 자연·문화·생태 계열 소폭 우대, "
                        "레저·웰니스 계열 과다선택 완화"
                    ),
            },
        "여행유형":
            trip_type,
        "일정시작시간":
            start_time,
        "출발지자동설정":
            auto_start_used,
        "출발위도":
            round(
                float(
                    start_lat
                ),
                6,
            ),
        "출발경도":
            round(
                float(
                    start_lon
                ),
                6,
            ),
        "경로장소수":
            int(
                len(
                    route_detail
                )
            ),
        "경로포함카테고리":
            sorted(
                actual_categories
            ),
        "포함하지못한카테고리":
            sorted(
                missing_categories
            ),
        "관광지세부분류":
            route_detail.loc[
                route_detail[
                    "카테고리"
                ]
                == "관광지",
                "세부분류",
            ]
            .astype(str)
            .tolist(),
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
                if api_key
                else "위경도 기반 추정"
            ),
        "데이터구조":
            {
                "음식점": [
                    RESTAURANT_PLACE_NAME,
                    RESTAURANT_SCORE_NAME,
                    RESTAURANT_LOCATION_NAME,
                    PLACE_RESTAURANT_NAME,
                ],
                "전통시장": [
                    TRADITIONAL_MARKET_NAME,
                    TRADITIONAL_MARKET_FACILITY_NAME,
                    PLACE_MARKET_NAME,
                ],
                "관광지": [
                    TOUR_SCORE_FINAL_NAME,
                    TOUR_GEO_FINAL_NAME,
                ],
                "관광지_QA원본": [
                    TOUR_SCORE_AUDIT_NAME,
                ],
            },
        "지역추천TOP":
            (
                region_ranking
                .head(
                    top_region_n
                )[
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
        output_dir
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
    print(
        "=" * 72
    )
    print(
        "LOCAL:ON v5.1 추천 완료"
    )
    print(
        "=" * 72
    )
    print(
        f"시도: {selected_sido}"
    )
    print(
        f"시군구: {selected_sigungu}"
    )
    print(
        "카테고리:",
        ", ".join(
            selected_categories
        ),
    )
    print(
        f"후보 개수: {category_counts}"
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
                "세부분류",
                "장소추천점수",
                "경로선택점수",
                "이전장소에서_이동분",
            ]
        ].to_string(
            index=False
        )
    )
    print()
    print(
        "생성 파일:"
    )
    print(
        " -",
        qa_path,
    )
    print(
        " -",
        region_path,
    )
    print(
        " -",
        candidate_path,
    )
    print(
        " -",
        route_path,
    )
    print(
        " -",
        summary_path,
    )

    return {
        "region_ranking":
            region_ranking,
        "all_places":
            all_places,
        "candidates":
            candidates,
        "route":
            route_detail,
        "summary":
            summary,
        "qa":
            qa,
    }


# ============================================================
# 15. CLI
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser(
            description=(
                "LOCAL:ON v5.1 "
                "지역/장소/경로 추천"
            )
        )
    )

    parser.add_argument(
        "--base-dir",
        default=str(
            BASE_DIR
        ),
        help=(
            "LOCAL:ON 데이터 폴더"
        ),
    )

    parser.add_argument(
        "--sido",
        default="충청남도",
    )

    parser.add_argument(
        "--sigungu",
        default=None,
        help=(
            "생략 시 선택 시도에서 "
            "추천점수가 가장 높은 지역 사용"
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
            "맛집,관광지,"
            "카페/베이커리,"
            "전통시장"
        ),
        help=(
            "쉼표 구분: "
            "맛집,관광지,"
            "카페/베이커리,"
            "전통시장"
        ),
    )

    parser.add_argument(
        "--start-time",
        default=(
            DEFAULT_START_TIME
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

    args = (
        parser.parse_args()
    )

    categories = [
        x.strip()
        for x
        in args.categories.split(
            ","
        )
        if x.strip()
    ]

    recommend_local_on_trip(
        selected_sido=(
            args.sido
        ),
        selected_sigungu=(
            args.sigungu
        ),
        selected_categories=(
            categories
        ),
        trip_type=(
            args.trip
        ),
        start_lat=(
            args.start_lat
        ),
        start_lon=(
            args.start_lon
        ),
        start_time=(
            args.start_time
        ),
        base_dir=Path(
            args.base_dir
        ),
    )


if __name__ == "__main__":
    main()
