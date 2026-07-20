"""
작성자: 박이완
작성 목적: 3개 공개 API를 비동기로 수집하고 Pydantic v2로 검증한 뒤,
          CSV와 Parquet으로 저장하여 읽기·쓰기 성능을 비교한다.

실행:
    python main.py

품질 검사:
    python -m pytest main.py -q
    python -m ruff check main.py --select E,F,I,UP
"""

import asyncio
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, Literal

import httpx
import pandas as pd
import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    IPvAnyAddress,
    ValidationError,
    model_validator,
)

# 실행 위치와 관계없이 현재 main.py를 기준으로 output 경로를 계산한다.
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

# Open-Meteo: 서울의 3일간 시간별 기온과 강수확률을 요청한다.
WEATHER_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=37.5665&longitude=126.9780"
    "&hourly=temperature_2m,precipitation_probability"
    "&forecast_days=3&timezone=Asia/Seoul"
)
COUNTRY_URL = "https://countries.dev/alpha/KOR"

# ip-api 무료 엔드포인트는 HTTPS가 아닌 HTTP를 사용한다.
IP_URL = (
    "http://ip-api.com/json/8.8.8.8"
    "?fields=status,message,query,country,city,lat,lon,timezone,isp"
)


# ============================================================
# 1) 비동기 수집
# ============================================================
async def fetch_json(
    client: httpx.AsyncClient,
    api_name: str,
    url: str,
) -> dict[str, Any]:
    """API 하나를 호출하고 HTTP 오류 확인 후 JSON을 반환한다."""
    response = await client.get(url)
    print(f"{api_name:<14}: HTTP {response.status_code}")

    # 4xx·5xx 응답은 정상 JSON으로 처리하지 않고 즉시 예외로 전환한다.
    response.raise_for_status()
    return response.json()


async def collect_all() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """asyncio.gather()로 세 API 요청을 동시에 실행한다."""
    async with httpx.AsyncClient(
        timeout=15.0,
        follow_redirects=True,
    ) as client:
        return await asyncio.gather(
            fetch_json(client, "Weather API", WEATHER_URL),
            fetch_json(client, "Country API", COUNTRY_URL),
            fetch_json(client, "IP API", IP_URL),
        )


# ============================================================
# 2) Pydantic v2 스키마 검증
# ============================================================
# 여러 모델에서 반복해서 사용하는 범위 제한 타입을 별칭으로 정의한다.
Latitude = Annotated[float, Field(ge=-90, le=90)]
Longitude = Annotated[float, Field(ge=-180, le=180)]
Temperature = Annotated[float, Field(ge=-90, le=60)]
Probability = Annotated[int, Field(ge=0, le=100)]


class ApiModel(BaseModel):
    """API가 추가 필드를 반환해도 필요한 필드만 검증하도록 하는 공통 모델."""
    model_config = ConfigDict(extra="ignore")


class HourlyUnits(ApiModel):
    """Open-Meteo 시간별 데이터의 단위를 검증한다."""
    time: Literal["iso8601"]
    temperature_2m: Literal["°C"]
    precipitation_probability: Literal["%"]


class HourlyWeather(ApiModel):
    """시간·기온·강수확률 배열의 타입, 범위, 길이를 검증한다."""
    time: list[datetime] = Field(min_length=1)
    temperature_2m: list[Temperature] = Field(min_length=1)
    precipitation_probability: list[Probability] = Field(min_length=1)

    @model_validator(mode="after")
    def check_same_length(self) -> "HourlyWeather":
        """같은 시각의 값끼리 대응하도록 세 배열의 길이를 확인한다."""
        lengths = {
            len(self.time),
            len(self.temperature_2m),
            len(self.precipitation_probability),
        }

        if len(lengths) != 1:
            raise ValueError(
                "time, temperature_2m, precipitation_probability 배열 길이가 다릅니다."
            )

        return self


class WeatherSchema(ApiModel):
    """서울 3일 시간대별 날씨 응답에서 필요한 필드를 검증한다."""
    latitude: Latitude
    longitude: Longitude
    generationtime_ms: float = Field(ge=0)
    utc_offset_seconds: int = Field(ge=-86_400, le=86_400)
    timezone: str = Field(min_length=1)
    timezone_abbreviation: str = Field(min_length=1)
    elevation: float = Field(ge=-500, le=9_000)
    hourly_units: HourlyUnits
    hourly: HourlyWeather


class FlagUrls(ApiModel):
    """국기 이미지 URL을 검증한다."""
    png: HttpUrl
    svg: HttpUrl


class Language(ApiModel):
    """국가 언어 코드와 명칭을 검증한다."""
    name: str = Field(min_length=1)
    iso639_1: str = Field(pattern=r"^[a-z]{2}$")
    iso639_2: str = Field(pattern=r"^[a-z]{3}$")
    nativeName: str = Field(min_length=1)


class Currency(ApiModel):
    """통화 코드, 통화명, 기호를 검증한다."""
    code: str = Field(pattern=r"^[A-Z]{3}$")
    name: str = Field(min_length=1)
    symbol: str = Field(min_length=1)


class CountrySchema(ApiModel):
    """대한민국 국가 응답의 타입과 합리적인 값 범위를 검증한다."""
    area: float = Field(gt=0)
    cioc: str = Field(pattern=r"^[A-Z]{3}$")
    flag: str = Field(min_length=1)
    gini: float = Field(ge=0, le=100)
    name: str = Field(min_length=1)
    flags: FlagUrls
    latlng: tuple[Latitude, Longitude]
    region: str = Field(min_length=1)
    borders: list[str]
    capital: str = Field(min_length=1)
    demonym: str = Field(min_length=1)
    languages: list[Language] = Field(min_length=1)
    subregion: str = Field(min_length=1)
    timezones: list[str] = Field(min_length=1)
    alpha2Code: str = Field(pattern=r"^[A-Z]{2}$")
    alpha3Code: str = Field(pattern=r"^[A-Z]{3}$")
    currencies: list[Currency] = Field(min_length=1)
    nativeName: str = Field(min_length=1)
    population: int = Field(gt=0)
    independent: bool
    numericCode: str = Field(pattern=r"^\d{3}$")
    callingCodes: list[str] = Field(min_length=1)
    topLevelDomain: list[str] = Field(min_length=1)
    populationDensity: float = Field(ge=0)


class IpSuccessSchema(ApiModel):
    """ip-api 정상 응답의 주소와 위치 범위를 검증한다."""
    status: Literal["success"]
    query: IPvAnyAddress
    country: str = Field(min_length=1)
    city: str = Field(min_length=1)
    lat: Latitude
    lon: Longitude
    timezone: str = Field(min_length=1)
    isp: str = Field(min_length=1)


class IpFailSchema(ApiModel):
    """ip-api가 HTTP 200과 함께 반환할 수 있는 실패 응답을 검증한다."""
    status: Literal["fail"]
    message: str = Field(min_length=1)


CollectedRecord = WeatherSchema | CountrySchema | IpSuccessSchema


def validate_data(
    weather_json: dict[str, Any],
    country_json: dict[str, Any],
    ip_json: dict[str, Any],
) -> list[CollectedRecord]:
    """세 JSON 응답을 각 Pydantic 모델로 변환하고 검증"""
    weather_record = WeatherSchema.model_validate(weather_json)
    country_record = CountrySchema.model_validate(country_json)

    # ip-api는 요청 자체가 성공해도 JSON 내부 status가 fail일 수 있다.
    if ip_json.get("status") == "fail":
        failed_ip = IpFailSchema.model_validate(ip_json)
        raise ValueError(f"IP API 실패: {failed_ip.message}")

    ip_record = IpSuccessSchema.model_validate(ip_json)
    records: list[CollectedRecord] = [
        weather_record,
        country_record,
        ip_record,
    ]

    print(f"검증 완료: {len(records)}개 레코드")
    return records


# ============================================================
# 3) CSV와 Parquet 저장 성능 비교
# ============================================================
def save_and_compare(records: list[CollectedRecord]) -> None:
    """검증 데이터를 저장·재로딩하고 형식별 처리 시간을 비교"""
    OUTPUT_DIR.mkdir(exist_ok=True)

    csv_path = OUTPUT_DIR / "collected_data.csv"
    parquet_path = OUTPUT_DIR / "collected_data.parquet"

    # model_dump()를 사용해 Pydantic 객체를 직렬화 가능한 딕셔너리로 변환한다.
    rows = [record.model_dump(mode="json") for record in records]
    dataframe = pd.DataFrame(rows)

    # 쓰기 시간은 파일 저장 호출 직전부터 완료 직후까지 측정한다.
    start = perf_counter()
    dataframe.to_csv(csv_path, index=False, encoding="utf-8-sig")
    csv_write_time = perf_counter() - start

    start = perf_counter()
    dataframe.to_parquet(
        parquet_path,
        engine="pyarrow",
        index=False,
    )
    parquet_write_time = perf_counter() - start

    # 다시 읽은 행 수를 확인하여 두 파일이 정상적으로 저장되었는지 검증한다.
    start = perf_counter()
    csv_rows = len(pd.read_csv(csv_path))
    csv_read_time = perf_counter() - start

    start = perf_counter()
    parquet_rows = len(
        pd.read_parquet(
            parquet_path,
            engine="pyarrow",
        )
    )
    parquet_read_time = perf_counter() - start

    if not csv_rows == parquet_rows == len(records):
        raise ValueError(
            f"CSV({csv_rows})와 Parquet({parquet_rows}) 레코드 수가 "
            f"검증된 레코드 수({len(records)})와 다릅니다."
        )

    print("\n형식       쓰기 시간     읽기 시간")
    print("-" * 40)
    print(
        f"CSV      {csv_write_time:>10.4f}초  "
        f"{csv_read_time:>10.4f}초  "
    )
    print(
        f"Parquet  {parquet_write_time:>10.4f}초  "
        f"{parquet_read_time:>10.4f}초  "
    )
    print(f"\n파일 저장 완료: {csv_path.name}, {parquet_path.name}")


# ============================================================
# 4) pytest 스키마 테스트 + Ruff 검사
# 실행: python -m pytest main.py -q
# 실행: python -m ruff check main.py --select E,F,I,UP
# ============================================================
def make_weather_data() -> dict[str, Any]:
    """정상·비정상 테스트가 공유할 기준 날씨 데이터를 반환"""
    return {
        "latitude": 37.5665,
        "longitude": 126.9780,
        "generationtime_ms": 1.23,
        "utc_offset_seconds": 32_400,
        "timezone": "Asia/Seoul",
        "timezone_abbreviation": "KST",
        "elevation": 38.0,
        "hourly_units": {
            "time": "iso8601",
            "temperature_2m": "°C",
            "precipitation_probability": "%",
        },
        "hourly": {
            "time": [
                "2024-06-01T00:00",
                "2024-06-01T01:00",
            ],
            "temperature_2m": [20.5, 19.8],
            "precipitation_probability": [10, 20],
        },
    }


def test_valid_weather_schema() -> None:
    """정상 날씨 데이터가 모델 검증을 통과하는지 확인"""
    valid_data = WeatherSchema.model_validate(make_weather_data())

    assert isinstance(valid_data, WeatherSchema)
    assert valid_data.latitude == 37.5665
    assert len(valid_data.hourly.time) == 2


def test_invalid_weather_latitude() -> None:
    """범위를 벗어난 위도가 ValidationError를 발생시키는지 확인"""
    invalid_data = make_weather_data()
    invalid_data["latitude"] = 9_999.0

    with pytest.raises(ValidationError) as error:
        WeatherSchema.model_validate(invalid_data)

    assert "latitude" in str(error.value)


# ============================================================
# 메인 실행
# ============================================================
def main() -> None:
    """수집 -> 검증 -> 저장·성능 비교 순서로 전체 파이프라인을 실행"""
    try:
        print("\n--------- 1) 비동기 수집 ---------")
        collected_json = asyncio.run(collect_all())
        print(f"수집 완료: {len(collected_json)}개 API")

        print("\n--------- 2) Pydantic v2 스키마 검증 ---------")
        validated_data = validate_data(*collected_json)

        for record in validated_data:
            print(f"- {record.__class__.__name__}: 검증 성공")

        print("\n--------- 3) CSV와 Parquet 저장 성능 비교 ---------")
        save_and_compare(validated_data)

    except httpx.HTTPError as error:
        print(f"HTTP 요청 실패: {error}")
        raise SystemExit(1) from error

    except ValidationError as error:
        print(f"Pydantic 검증 실패:\n{error}")
        raise SystemExit(1) from error

    except ImportError as error:
        print(
            "Parquet 처리에 필요한 pyarrow가 설치되지 않았습니다.\n"
            "설치 명령: python -m pip install pyarrow"
        )
        raise SystemExit(1) from error

    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"데이터 처리 실패: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()