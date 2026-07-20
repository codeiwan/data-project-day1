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
from time import perf_counter
from pathlib import Path
from typing import Any
import httpx

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
        weather_json, country_json, ip_json = await asyncio.gather(
            fetch_json(client, "Weather API", WEATHER_URL),
            fetch_json(client, "Country API", COUNTRY_URL),
            fetch_json(client, "IP API", IP_URL),
        )
    
    return weather_json, country_json, ip_json


# ============================================================
# 메인 실행
# ============================================================
def main() -> None:
    """3개 API를 비동기로 수집하고, CSV와 Parquet 저장 성능을 비교한다."""
    try:
        print("--------- 1) 비동기 수집 ---------")
        collected_json = asyncio.run(collect_all())
        print(f"수집 완료: {len(collected_json)}개 API")
    except (
        httpx.HTTPError,
    ) as error:
        print(f"파이프라인 실행 실패: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()