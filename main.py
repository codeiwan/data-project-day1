"""
작성자: 박이완
작성 목적: 3개 API를 비동기로 수집하고, Pydantic으로 검증한 뒤
          CSV와 Parquet 저장 성능을 비교한다.

실행:
    python main.py

품질 검사:
    pytest main.py -q
    ruff check main.py --select E,F,I,UP
"""

import asyncio
from datetime import datetime
from time import perf_counter
from pathlib import Path
from typing import Any, Annotated, Literal

import httpx
import pandas as pd
from pydantic import BaseModel, Field, model_validator, HttpUrl, ValidationError, IPvAnyAddress


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

WEATHER_URL = ("https://api.open-meteo.com/v1/forecast?latitude=37.5665&"
               "longitude=126.9780&hourly=temperature_2m,precipitation_probability&"
               "forecast_days=3&timezone=Asia/Seoul")
COUNTRY_URL = "https://countries.dev/alpha/KOR"
IP_URL = ("http://ip-api.com/json/8.8.8.8"
          "?fields=status,message,query,country,city,lat,lon,timezone,isp")

# ============================================================
# 1) 비동기 수집
# ============================================================
async def fetch_json(client: httpx.AsyncClient, api_name: str, url: str) -> dict[str, Any]:
    """API를 호출하고 HTTP 응답 상태를 확인"""
    response = await client.get(url)
    print(f"{api_name:<14}: HTTP {response.status_code}")
    response.raise_for_status()
    return response.json()


async def collect_all() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """asyncio.gether()로 3개 API를 비동기로 호출하고 결과 수집"""
    async with httpx.AsyncClient() as client:
        return await asyncio.gather(
            fetch_json(client, "Weather API", WEATHER_URL),
            fetch_json(client, "Country API", COUNTRY_URL),
            fetch_json(client, "IP API", IP_URL),
        )


# ============================================================
# 2) Pydantic v2 스키마 검증
# ============================================================
Latitude = Annotated[float, Field(ge=-90, le=90)]
Longitude = Annotated[float, Field(ge=-180, le=180)]
Temperature = Annotated[float, Field(ge=-90, le=60)]
Probability = Annotated[int, Field(ge=0, le=100)]


class HourlyUnits(BaseModel):
    """Open-Meteo 시간별 데이터 단위."""

    time: Literal["iso8601"]
    temperature_2m: Literal["°C"]
    precipitation_probability: Literal["%"]


class HourlyWeather(BaseModel):
    """시간·기온·강수확률 배열."""

    time: list[datetime] = Field(min_length=1)
    temperature_2m: list[Temperature] = Field(min_length=1)
    precipitation_probability: list[Probability] = Field(min_length=1)

    @model_validator(mode="after")
    def check_same_length(self) -> "HourlyWeather":
        lengths = {
            len(self.time),
            len(self.temperature_2m),
            len(self.precipitation_probability),
        }
        if len(lengths) != 1:
            raise ValueError(
                "time, temperature_2m, precipitation_probability "
                "배열 길이가 다릅니다."
            )
        return self


class WeatherSchema(BaseModel):
    """서울 3일 시간대별 기온·강수확률"""
    latitude: Latitude
    longitude: Longitude
    generationtime_ms: float = Field(ge=0)
    utc_offset_seconds: int = Field(ge=-86_400, le=86_400)
    timezone: str = Field(min_length=1)
    timezone_abbreviation: str = Field(min_length=1)
    elevation: float = Field(ge=-500, le=9_000)
    hourly_units: HourlyUnits
    hourly: HourlyWeather


class FlagUrls(BaseModel):
    png: HttpUrl
    svg: HttpUrl


class Language(BaseModel):
    name: str = Field(min_length=1)
    iso639_1: str = Field(pattern=r"^[a-z]{2}$")
    iso639_2: str = Field(pattern=r"^[a-z]{3}$")
    nativeName: str = Field(min_length=1)


class Currency(BaseModel):
    code: str = Field(pattern=r"^[A-Z]{3}$")
    name: str = Field(min_length=1)
    symbol: str = Field(min_length=1)


class CountrySchema(BaseModel):
    """대한민국 국가 정보."""

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

class IpSuccessSchema(BaseModel):
    """ip-api 정상 응답."""

    status: Literal["success"]
    query: IPvAnyAddress
    country: str = Field(min_length=1)
    city: str = Field(min_length=1)
    lat: Latitude
    lon: Longitude
    timezone: str = Field(min_length=1)
    isp: str = Field(min_length=1)


class IpFailSchema(BaseModel):
    """ip-api 실패 응답."""

    status: Literal["fail"]
    message: str = Field(min_length=1)

CollectedRecord = WeatherSchema | CountrySchema | IpSuccessSchema

def validate_data(
        weather_json: dict[str, Any],
        country_json: dict[str, Any],
        ip_json: dict[str, Any],
) -> list[CollectedRecord]:
    """API JSON에서 필요한 필드만 추출하여 Pydantic v2 모델로 타입·범위 검증"""
    weather_record = WeatherSchema.model_validate(weather_json)
    country_record = CountrySchema.model_validate(country_json)

    if ip_json.get("status") == "fail":
        failed_ip = IpFailSchema.model_validate(ip_json)
        raise ValueError(f"IP API 실패: {failed_ip.message}")

    ip_record = IpSuccessSchema.model_validate(ip_json)

    print(f"검증 완료: {len([weather_record, country_record, ip_record])}개 레코드")

    return [weather_record, country_record, ip_record]


# ============================================================
# 3) CSV와 Parquet 저장 성능 비교
# ============================================================
def save_and_compare(records: list[CollectedRecord]) -> None:
    """검증 데이터를 CSV와 Parquet로 저장하고, 저장 속도를 비교한다."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # CSV, Parquet 저장 경로
    csv_path = OUTPUT_DIR / "collected_data.csv"
    parquet_path = OUTPUT_DIR / "collected_data.parquet"

    # CSV와 Parquet 저장 속도 비교를 위해 각 저장 시간을 측정
    rows = [record.model_dump(mode="json") for record in records]
    dataframe = pd.DataFrame(rows)

    # 쓰기 속도 측정
    start = perf_counter()
    dataframe.to_csv(csv_path, index=False, encoding="utf-8-sig")
    csv_write_time = perf_counter() - start

    start = perf_counter()
    dataframe.to_parquet(parquet_path, index=False)
    parquet_write_time = perf_counter() - start

    # 읽기 속도 측정
    start = perf_counter()
    csv_rows = len(pd.read_csv(csv_path))
    csv_read_time = perf_counter() - start

    start = perf_counter()
    parquet_rows = len(pd.read_parquet(parquet_path))
    parquet_read_time = perf_counter() - start

    try:
        assert csv_rows == parquet_rows == len(records)
    except AssertionError:
        raise ValueError(
            f"CSV({csv_rows})와 Parquet({parquet_rows}) 레코드 수가 "
            f"검증된 레코드 수({len(records)})와 다릅니다."
        )
    
    print(f"CSV 쓰기: {csv_write_time:.4f}s, 읽기: {csv_read_time:.4f}s")
    print(f"Parquet 쓰기: {parquet_write_time:.4f}s, 읽기: {parquet_read_time:.4f}s")

    print(f"\n파일 저장 완료: {csv_path.name}, {parquet_path.name}")


# ============================================================
# 메인 실행
# ============================================================
def main() -> None:
    """3개 API를 비동기로 수집하고, CSV와 Parquet 저장 성능을 비교한다."""
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

    except (KeyError, TypeError, ValueError) as error:
        print(f"데이터 처리 실패: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()