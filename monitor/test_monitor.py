from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from server import Handler, active_api_model, api_base_url, model_catalog
from telemetry import Telemetry, llama_processes


class MonitorTest(unittest.TestCase):
    def test_api_model_discovery_keeps_the_configured_url_prefix(self):
        base_url = api_base_url("https://node.example.ts.net/local-llm/v1/")
        with patch("server.HTTPConnection") as connect:
            http = connect.return_value
            http.getresponse.return_value.status = 200
            http.getresponse.return_value.read.return_value = b'{"data":[null,{"id":"demo"}]}'
            self.assertEqual(active_api_model("http://100.64.0.10:10444/v1"), "demo")
            http.request.assert_called_once_with("GET", "/v1/models")
            http.close.assert_called_once()
        with patch("server.HTTPSConnection") as connect:
            http = connect.return_value
            http.getresponse.return_value.status = 200
            http.getresponse.return_value.read.return_value = b'{"data":[{"id":"demo"}]}'
            self.assertEqual(active_api_model(base_url), "demo")
            http.request.assert_called_once_with("GET", "/local-llm/v1/models")

    def test_tailnet_model_origin_requires_the_matching_https_port(self):
        handler = Handler.__new__(Handler)
        model = "node.example.ts.net:8791"
        handler.server = SimpleNamespace(origins={model: "https://" + model})
        handler.json = Mock()
        handler.headers = {
            "Host": model,
            "Origin": "https://" + model,
        }
        self.assertFalse(handler.check_origin())
        handler.headers["X-Forwarded-Proto"] = "https"
        self.assertTrue(handler.check_origin())
        handler.headers["Origin"] = "https://node.example.ts.net:8790"
        self.assertFalse(handler.check_origin())

    def test_direct_tailnet_origin_allows_the_configured_ip_and_port(self):
        handler = Handler.__new__(Handler)
        host = "100.64.0.10:8789"
        handler.server = SimpleNamespace(origins={host: "http://" + host})
        handler.json = Mock()
        handler.headers = {"Host": host, "Origin": "http://" + host}
        self.assertTrue(handler.check_origin())
        handler.headers["Origin"] = "http://100.64.0.10:8788"
        self.assertFalse(handler.check_origin())

    def test_library_lists_complete_split_models_once(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            library = root / "models"
            library.mkdir()
            outside = root / "outside.gguf"
            outside.write_bytes(b"data")
            (library / "outside.gguf").symlink_to(outside)
            for filename in (
                "regular.gguf",
                "split-00001-of-00002.gguf", "split-00002-of-00002.gguf",
                "partial-00001-of-00003.gguf", "partial-00003-of-00003.gguf",
                "mmproj-example.gguf", "downloading.gguf.partial",
            ):
                (library / filename).write_bytes(b"data")
            models = model_catalog([library])
            self.assertEqual([(m["name"], m["bytes"]) for m in models], [("regular", 4), ("split", 8)])
            self.assertTrue(models[1]["id"].endswith("split-00001-of-00002.gguf"))

    def test_rapl_power_handles_wrap_and_missing_readings(self):
        with TemporaryDirectory() as temp:
            domain = Path(temp)
            energy = domain / "energy_uj"
            meter = Telemetry()
            (domain / "max_energy_range_uj").write_text("100000000")
            energy.write_text("90000000")
            self.assertIsNone(meter.power([domain], 10))
            energy.write_text("10000000")
            self.assertEqual(meter.power([domain], 12), 10)
            energy.unlink()
            self.assertIsNone(meter.power([domain], 14))
            energy.write_text("20000000")
            self.assertIsNone(meter.power([domain], 16))

    def test_process_list_omits_exited_llama_servers(self):
        with TemporaryDirectory() as temp, patch("telemetry.PROC", Path(temp)):
            for pid, state in ((10, "S (sleeping)"), (11, "Z (zombie)")):
                process = Path(temp) / str(pid)
                process.mkdir()
                (process / "comm").write_text("llama-server\n")
                (process / "status").write_text(f"State:\t{state}\nVmRSS:\t1024 kB\n")
                (process / "cmdline").write_bytes(b"llama-server\0--model\0/tmp/demo.gguf\0")
            self.assertEqual(llama_processes(), [{"pid": 10, "model": "demo.gguf", "rss": 1024 ** 2}])
