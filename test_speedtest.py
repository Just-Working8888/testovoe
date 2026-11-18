import contextlib
import functools
import io
import socket
import threading
import time
import unittest
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import speedtest
from speedtest import MIB, Attempt, download, main, run_attempts, summarize


class Handler(BaseHTTPRequestHandler):
    """Тестовый сервер. Протокол HTTP/1.0: соединение закрывается после каждого ответа."""

    requests = Counter()
    active = 0
    max_active = 0
    lock = threading.Lock()

    def do_GET(self):
        cls = type(self)
        with cls.lock:
            cls.requests[self.path] += 1
            cls.active += 1
            cls.max_active = max(cls.max_active, cls.active)
            request_number = cls.requests[self.path]
        try:
            self.route(request_number)
        except (BrokenPipeError, ConnectionResetError):
            pass  # клиент оборвал соединение (таймаут) — для тестов это нормально
        finally:
            with cls.lock:
                cls.active -= 1

    def route(self, request_number):
        kind, _, arg = self.path.strip("/").partition("/")
        if kind == "bytes":  # /bytes/<n> — тело из n байт с Content-Length
            self.send_body(int(arg))
        elif kind == "bytes-no-length":  # /bytes-no-length/<n> — без Content-Length
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"x" * int(arg))
        elif kind == "chunked":  # /chunked/<n> — Transfer-Encoding: chunked, куски по 1000 байт
            self.protocol_version = "HTTP/1.1"
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
            self.end_headers()
            remaining = int(arg)
            while remaining:
                part = min(1000, remaining)
                self.wfile.write(f"{part:x}\r\n".encode() + b"x" * part + b"\r\n")
                remaining -= part
            self.wfile.write(b"0\r\n\r\n")
        elif kind == "status":  # /status/<code>
            self.send_body(16, status=int(arg))
        elif kind == "flaky":  # нечётные запросы — 500, чётные — 1000 байт
            self.send_body(1000, status=500 if request_number % 2 else 200)
        elif kind == "slow":  # ответ позже, чем клиентский таймаут
            time.sleep(1)
            self.send_body(10)
        elif kind == "trickle":  # байты по капле: каждое чтение укладывается в таймаут, вся загрузка — нет
            self.send_response(200)
            self.end_headers()
            for _ in range(100):
                self.wfile.write(b"x")
                self.wfile.flush()
                time.sleep(0.02)
        elif kind == "truncated":  # обещаем 1000 байт, отдаём 10
            self.send_response(200)
            self.send_header("Content-Length", "1000")
            self.end_headers()
            self.wfile.write(b"x" * 10)

    def send_body(self, size, status=200):
        self.send_response(status)
        self.send_header("Content-Length", str(size))
        self.end_headers()
        self.wfile.write(b"x" * size)

    def log_message(self, *args):
        pass


class LocalServerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.server.daemon_threads = True
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        Handler.requests.clear()
        Handler.max_active = 0


class SummaryTest(unittest.TestCase):
    def test_average_time_and_speed(self):
        attempts = [
            Attempt(1, seconds=0.5, bytes_received=1 * MIB, status=200),
            Attempt(2, seconds=1.5, bytes_received=3 * MIB, status=200),
        ]
        summary = summarize(attempts)
        self.assertEqual(summary.succeeded, 2)
        self.assertEqual(summary.failed, 0)
        self.assertEqual(summary.total_bytes, 4 * MIB)
        self.assertAlmostEqual(summary.avg_seconds, 1.0)
        # 4 MiB за 2 с суммарно → 2 MiB/s; 4·1048576·8 / 2 / 1e6 Mbps
        self.assertAlmostEqual(speedtest.to_mib_per_second(summary.bytes_per_second), 2.0)
        self.assertAlmostEqual(speedtest.to_mbps(summary.bytes_per_second), 16.777216)

    def test_failed_attempts_are_excluded(self):
        attempts = [
            Attempt(1, seconds=1.0, bytes_received=2 * MIB, status=200),
            Attempt(2, seconds=9.0, status=500, error="HTTP 500"),
            Attempt(3, seconds=3.0, error="таймаут"),
        ]
        summary = summarize(attempts)
        self.assertEqual((summary.succeeded, summary.failed), (1, 2))
        self.assertAlmostEqual(summary.avg_seconds, 1.0)
        self.assertAlmostEqual(summary.bytes_per_second, 2 * MIB)

    def test_no_successful_attempts(self):
        summary = summarize([Attempt(n, seconds=0.1, error="x") for n in range(1, 11)])
        self.assertEqual((summary.succeeded, summary.failed, summary.total_bytes), (0, 10, 0))
        self.assertIsNone(summary.avg_seconds)
        self.assertIsNone(summary.bytes_per_second)
        self.assertIn("невозможно", speedtest.format_summary(summary))


class DownloadTest(LocalServerTestCase):
    def test_counts_bytes_with_content_length(self):
        size = 3 * speedtest.CHUNK_SIZE + 123  # несколько чанков и неполный последний
        self.assertEqual(download(f"{self.base_url}/bytes/{size}"), (200, size))

    def test_counts_bytes_without_content_length(self):
        size = MIB + 7
        self.assertEqual(download(f"{self.base_url}/bytes-no-length/{size}"), (200, size))

    def test_counts_bytes_with_chunked_encoding(self):
        self.assertEqual(download(f"{self.base_url}/chunked/123456"), (200, 123456))

    def test_http_error_is_failure_with_status(self):
        attempt = speedtest.run_attempt(1, f"{self.base_url}/status/404")
        self.assertFalse(attempt.ok)
        self.assertEqual(attempt.status, 404)
        self.assertEqual(attempt.bytes_received, 0)

    def test_socket_timeout(self):
        fetch = functools.partial(download, timeout=0.2)
        attempt = speedtest.run_attempt(1, f"{self.base_url}/slow", fetch)
        self.assertFalse(attempt.ok)
        self.assertIn("таймаут", attempt.error)
        self.assertLess(attempt.seconds, 1.0)

    def test_total_request_limit(self):
        fetch = functools.partial(download, timeout=1.0, max_seconds=0.3)
        attempt = speedtest.run_attempt(1, f"{self.base_url}/trickle", fetch)
        self.assertFalse(attempt.ok)
        self.assertIn("таймаут", attempt.error)
        self.assertLess(attempt.seconds, 1.0)

    def test_truncated_body(self):
        attempt = speedtest.run_attempt(1, f"{self.base_url}/truncated")
        self.assertFalse(attempt.ok)
        self.assertIn("некорректный ответ", attempt.error)

    def test_connection_refused(self):
        with socket.socket() as sock:  # берём заведомо свободный порт
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        attempt = speedtest.run_attempt(1, f"http://127.0.0.1:{port}/")
        self.assertFalse(attempt.ok)
        self.assertIn("сетевая ошибка", attempt.error)


class RunAttemptsTest(LocalServerTestCase):
    def test_requests_are_sequential(self):
        events = []

        def fake_fetch(url):
            events.append("start")
            time.sleep(0.001)
            events.append("end")
            return 200, 1

        attempts = list(run_attempts("http://example.test/", fetch=fake_fetch))
        self.assertEqual([a.number for a in attempts], list(range(1, 11)))
        self.assertEqual(events, ["start", "end"] * 10)

    def test_server_never_sees_parallel_requests(self):
        attempts = list(run_attempts(f"{self.base_url}/bytes/{MIB}"))
        self.assertTrue(all(a.ok for a in attempts))
        self.assertEqual(Handler.requests[f"/bytes/{MIB}"], 10)
        self.assertEqual(Handler.max_active, 1)

    def test_partial_success(self):
        attempts = list(run_attempts(f"{self.base_url}/flaky"))
        summary = summarize(attempts)
        self.assertEqual((summary.succeeded, summary.failed), (5, 5))
        self.assertEqual(summary.total_bytes, 5 * 1000)
        self.assertEqual([a.status for a in attempts], [500, 200] * 5)

    def test_exactly_ten_attempts_without_retries(self):
        attempts = list(run_attempts(f"{self.base_url}/status/503"))
        self.assertEqual(len(attempts), 10)
        self.assertEqual(Handler.requests["/status/503"], 10)


class CliTest(LocalServerTestCase):
    def run_main(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = main(list(argv))
        return code, out.getvalue()

    def test_success_exit_code_and_output(self):
        code, output = self.run_main(f"{self.base_url}/bytes/{MIB}")
        self.assertEqual(code, 0)
        self.assertIn("Успешных попыток:      10 из 10", output)
        self.assertIn("Всего скачано:         10.00 MiB", output)
        self.assertIn("Mbps", output)

    def test_all_failed_gives_nonzero_exit_code(self):
        code, output = self.run_main(f"{self.base_url}/status/500")
        self.assertEqual(code, 1)
        self.assertIn("Неуспешных попыток:    10", output)
        self.assertIn("невозможно", output)

    def test_invalid_url_is_rejected(self):
        for bad in ("example.com/image.jpg", "ftp://example.com/file", "http://"):
            with self.subTest(url=bad), self.assertRaises(SystemExit) as ctx:
                self.run_main(bad)
            self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
