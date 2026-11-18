#!/usr/bin/env python3
"""Замер скорости HTTP-загрузки ресурса: 10 последовательных GET-запросов."""
from __future__ import annotations

import argparse
import http.client
import socket
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Iterator
from urllib.parse import urlparse

ATTEMPTS = 10
TIMEOUT_SECONDS = 10.0       # таймаут на подключение и на каждое чтение из сокета
MAX_REQUEST_SECONDS = 120.0  # общий лимит на попытку: защита от сервера, отдающего байты по капле
CHUNK_SIZE = 64 * 1024
MIB = 1024 * 1024

Fetch = Callable[[str], tuple[int, int]]


@dataclass
class Attempt:
    number: int
    seconds: float
    bytes_received: int = 0
    status: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class Summary:
    succeeded: int
    failed: int
    total_bytes: int
    avg_seconds: float | None       # None, если успешных попыток нет
    bytes_per_second: float | None  # None, если успешных попыток нет


# --- Измерение ---------------------------------------------------------------

def download(
    url: str,
    timeout: float = TIMEOUT_SECONDS,
    max_seconds: float = MAX_REQUEST_SECONDS,
) -> tuple[int, int]:
    """Скачивает тело ответа целиком и возвращает (HTTP-статус, фактически полученные байты).

    Тело читается чанками и сразу отбрасывается, в памяти не накапливается.
    Ответы 4xx/5xx urllib превращает в HTTPError.
    """
    # Свой User-Agent: многие CDN отвечают 403 на стандартный "Python-urllib".
    headers = {"User-Agent": "http-speedtest/1.0", "Cache-Control": "no-cache"}
    request = urllib.request.Request(url, headers=headers)
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        received = 0
        # read1 отдаёт то, что уже пришло, а не ждёт полный чанк — иначе лимит времени
        # проверялся бы слишком редко на медленном соединении.
        while chunk := response.read1(CHUNK_SIZE):
            received += len(chunk)
            if time.perf_counter() - started > max_seconds:
                raise TimeoutError(f"загрузка не уложилась в {max_seconds:g} с")

        # http.client молча принимает оборванное соединение при чтении по частям,
        # поэтому сверяем фактически полученные байты с заявленными.
        expected = response.headers.get("Content-Length", "")
        if expected.isdigit() and received != int(expected):
            raise http.client.HTTPException(f"получено {received} байт, а Content-Length = {expected}")
        return response.status, received


def describe_error(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code} {exc.reason}"
    if isinstance(exc, urllib.error.URLError):
        if isinstance(exc.reason, socket.timeout):
            return "таймаут подключения"
        return f"сетевая ошибка: {exc.reason}"
    if isinstance(exc, (socket.timeout, TimeoutError)):  # в 3.10+ это один и тот же класс
        return f"таймаут: {exc}"
    if isinstance(exc, http.client.HTTPException):
        return f"некорректный ответ сервера: {type(exc).__name__} {exc}"
    if isinstance(exc, OSError):
        return f"сетевая ошибка: {exc}"
    return f"ошибка запроса: {exc}"


def run_attempt(number: int, url: str, fetch: Fetch = download) -> Attempt:
    started = time.perf_counter()
    try:
        status, received = fetch(url)
    except urllib.error.HTTPError as exc:
        exc.close()
        return Attempt(number, time.perf_counter() - started, status=exc.code, error=describe_error(exc))
    except (OSError, http.client.HTTPException, ValueError) as exc:
        return Attempt(number, time.perf_counter() - started, error=describe_error(exc))
    return Attempt(number, time.perf_counter() - started, bytes_received=received, status=status)


def run_attempts(url: str, count: int = ATTEMPTS, fetch: Fetch = download) -> Iterator[Attempt]:
    """Строго последовательно: следующая попытка начинается после завершения предыдущей."""
    for number in range(1, count + 1):
        yield run_attempt(number, url, fetch)


# --- Статистика --------------------------------------------------------------

def summarize(attempts: list[Attempt]) -> Summary:
    successful = [a for a in attempts if a.ok]
    total_bytes = sum(a.bytes_received for a in successful)
    total_seconds = sum(a.seconds for a in successful)
    if not successful or total_seconds <= 0:
        return Summary(len(successful), len(attempts) - len(successful), total_bytes, None, None)
    return Summary(
        succeeded=len(successful),
        failed=len(attempts) - len(successful),
        total_bytes=total_bytes,
        avg_seconds=total_seconds / len(successful),
        bytes_per_second=total_bytes / total_seconds,
    )


def to_mib_per_second(bytes_per_second: float) -> float:
    return bytes_per_second / MIB


def to_mbps(bytes_per_second: float) -> float:
    return bytes_per_second * 8 / 1_000_000


# --- Вывод -------------------------------------------------------------------

def format_attempt(attempt: Attempt) -> str:
    prefix = f"#{attempt.number:02d}"
    if attempt.ok:
        return (
            f"{prefix}  HTTP {attempt.status}  {attempt.bytes_received:>13,} байт"
            f" ({attempt.bytes_received / MIB:8.2f} MiB)  {attempt.seconds:8.3f} с"
        )
    return f"{prefix}  ОШИБКА    {attempt.error}  ({attempt.seconds:.3f} с)"


def format_summary(summary: Summary) -> str:
    total = summary.succeeded + summary.failed
    lines = [
        "",
        "Итог:",
        f"  Успешных попыток:      {summary.succeeded} из {total}",
        f"  Неуспешных попыток:    {summary.failed}",
    ]
    if summary.bytes_per_second is None:
        lines.append("  Нет успешных запросов — скорость рассчитать невозможно.")
        return "\n".join(lines)
    lines += [
        f"  Среднее время запроса: {summary.avg_seconds:.3f} с",
        f"  Всего скачано:         {summary.total_bytes / MIB:.2f} MiB ({summary.total_bytes:,} байт)",
        f"  Средняя скорость:      {to_mib_per_second(summary.bytes_per_second):.2f} MiB/s"
        f" ({to_mbps(summary.bytes_per_second):.2f} Mbps)",
    ]
    return "\n".join(lines)


# --- CLI ---------------------------------------------------------------------

def parse_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise argparse.ArgumentTypeError(f"нужен абсолютный URL со схемой http:// или https://, получено: {value!r}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"Замер скорости HTTP-загрузки: {ATTEMPTS} последовательных GET-запросов к URL."
    )
    parser.add_argument("url", type=parse_url, help="адрес ресурса, например большой картинки")
    args = parser.parse_args(argv)

    print(f"URL: {args.url}")
    print(f"Попыток: {ATTEMPTS}, таймаут сокета: {TIMEOUT_SECONDS:g} с, лимит на попытку: {MAX_REQUEST_SECONDS:g} с\n")

    attempts: list[Attempt] = []
    try:
        for attempt in run_attempts(args.url):
            print(format_attempt(attempt), flush=True)
            attempts.append(attempt)
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        return 130

    summary = summarize(attempts)
    print(format_summary(summary))
    return 0 if summary.succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
