"""Exercise each student-test trip through the same HTTP calls as the evaluator UI."""

import argparse
import json
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from time import monotonic, sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def call(base, path, body=None):
    request = Request(base + path, data=None if body is None else json.dumps(body).encode(),
                      headers={"Content-Type": "application/json", "Origin": "http://127.0.0.1:8080"})
    with urlopen(request, timeout=45) as response:
        return json.load(response)


def stop(process):
    process.terminate()
    process.wait(timeout=10)


def run(args):
    out = Path("results")
    logfile = out / "api_smoke.log"
    base = f"http://127.0.0.1:{args.port}"
    with ExitStack() as stack:
        log = stack.enter_context(logfile.open("w", encoding="utf-8"))
        server = subprocess.Popen(
            [sys.executable, "-u", "-m", "pathsnap.api", "--port", str(args.port),
             "--trips", str(args.trips)], stdout=log, stderr=subprocess.STDOUT)
        stack.callback(stop, server)
        deadline = monotonic() + 45
        while True:
            try:
                call(base, "/health")
                break
            except URLError:
                if server.poll() is not None or monotonic() > deadline:
                    raise RuntimeError(f"API startup failed. Read {logfile}.") from None
                sleep(0.25)
        trips = call(base, "/trips")
        assert trips["mode"] == "development" and len(trips["trip_ids"]) == 10
        observed = []
        for trip_id in trips["trip_ids"]:
            result = call(base, "/predict", {"trip_id": trip_id})
            assert result["trip_id"] == trip_id and set(result) == {"trip_id", "geometry", "metrics"}
            assert set(result["geometry"]) == {"gps", "predicted_route"}
            assert result["geometry"]["predicted_route"]
            metrics = result["metrics"]
            assert sum(map(len, result["geometry"]["gps"])) == metrics["gps_observations"]
            assert len(result["geometry"]["predicted_route"]) == metrics["route_parts"]
            assert 0.0 <= metrics["observation_accuracy"] <= 1.0
            observed.append({"trip": trip_id, **metrics})
            print(f"HTTP 200 {trip_id} ({metrics['inference_ms']} ms)", flush=True)
        try:
            call(base, "/predict", {"trip_id": "missing"})
        except HTTPError as error:
            assert error.code == 404
        else:
            raise AssertionError("Unknown trip should return 404")
        if args.node:
            ui = subprocess.Popen([sys.executable, "-m", "http.server", str(args.ui_port),
                                   "--bind", "127.0.0.1", "--directory", "ui"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            stack.callback(stop, ui)
            sleep(0.5)
            subprocess.run([args.node, "scripts/capture_demo.cjs", args.playwright, args.browser,
                            str(args.port), str(args.ui_port)], check=True)
    lines = logfile.read_text(encoding="utf-8").splitlines()
    evidence_index = next(i for i, line in enumerate(lines) if f"trip={observed[0]['trip']}" in line)
    prediction = lines[evidence_index]
    success = next(line for line in lines[evidence_index + 1:] if 'POST /predict HTTP/1.1" 200 OK' in line)
    result = {"successful_predictions": len(observed), "unknown_trip_status": 404,
              "terminal_line": prediction + "\n" + success, "trips": observed}
    (out / "api_smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trips", type=Path, default=Path("Student folder/student_test"))
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--ui-port", type=int, default=8781)
    parser.add_argument("--node")
    parser.add_argument("--playwright")
    parser.add_argument("--browser")
    options = parser.parse_args()
    if options.node and not (options.playwright and options.browser):
        parser.error("--node requires --playwright and --browser")
    run(options)
