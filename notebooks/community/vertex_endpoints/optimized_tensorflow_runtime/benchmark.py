import itertools
import json
import threading
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf

logging = tf.get_logger()
logging.propagate = False
logging.setLevel("INFO")

def benchmark_qps(send_request, requests, qps):
    logging.info("Running benchmark at {} qps".format(qps))
    # List appends are thread safe
    num_requests = len(requests)
    success = []
    error = []
    latency = []

    def _make_call(i):
        """Send a request to using specified method and measure observed latency."""
        start_time = time.time()
        try:
            _ = send_request(requests[i])
            success.append(1)
        except Exception as e:
            print(e)
            error.append(1)

        latency.append(time.time() - start_time)
        if len(latency) % (qps * 10) == 0:
            logging.info("received {} responses.".format(len(latency)))

    thread_lst = []
    miss_rate_percent = []
    start_time = time.time()
    previous_worker_start = start_time
    for i in range(num_requests):
        thread = threading.Thread(target=_make_call, args=(i,))
        thread_lst.append(thread)
        thread.start()
        if i % (qps * 10) == 0 and i != 0:
            logging.info("sent {} requests.".format(i))

        # send requests at a constant rate and adjust for the time it took to send previous request
        pause = 1.0 / qps - (time.time() - previous_worker_start)
        if pause > 0:
            time.sleep(pause)
        else:
            missed_delay = (
                100 * ((time.time() - previous_worker_start) - 1.0 / qps) / (1.0 / qps)
            )
            miss_rate_percent.append(missed_delay)
        previous_worker_start = time.time()

    for thread in thread_lst:
        thread.join()

    acc_time = time.time() - start_time

    avg_miss_rate_percent = 0
    if len(miss_rate_percent) > 0:
        avg_miss_rate_percent = np.average(miss_rate_percent)
        logging.warning(
            "couldn't keep up at current QPS rate, average miss rate:{:.2f}%".format(
                avg_miss_rate_percent
            )
        )

    logging.info(
        "num_qps:{} requests/second: {:.2f} #success:{} #error:{} "
        "latencies: [avg:{:.2f}ms p50:{:.2f}ms p90:{:.2f}ms p99:{:.2f}ms]".format(
            qps,
            num_requests / acc_time,
            sum(success),
            sum(error),
            np.average(latency) * 1000,
            np.percentile(latency, 50) * 1000,
            np.percentile(latency, 90) * 1000,
            np.percentile(latency, 99) * 1000,
        )
    )
    return {
        "reqested_qps": qps,
        "actual_qps": num_requests / acc_time,
        "success": sum(success),
        "error": sum(error),
        "time": acc_time,
        "avg_latency": np.average(latency) * 1000,
        "p50": np.percentile(latency, 50) * 1000,
        "p90": np.percentile(latency, 90) * 1000,
        "p99": np.percentile(latency, 99) * 1000,
        "avg_miss_rate_percent": avg_miss_rate_percent,
    }


def benchmark(
    send_request,
    build_request,
    request_file_path,
    qps_list,
    duration_sec,
    model_name="default",
):
    requests = []
    with tf.io.gfile.GFile(request_file_path, "r") as f:
        for line in f:
            row_dict = json.loads(line)
            requests.append(build_request(row_dict, model_name))

    results = []
    for qps in qps_list:
        num_requests = max(qps * duration_sec, 10)
        requests_for_qps = list(
            itertools.islice(itertools.cycle(requests), num_requests)
        )
        logging.info(f"benchmarking at {qps} QPS, sending {num_requests} requests")
        results.append(benchmark_qps(send_request, requests_for_qps, qps))

    columns = [
        "reqested_qps",
        "actual_qps",
        "avg_latency",
        "p50",
        "p90",
        "p99",
        "success",
        "error",
    ]
    merged_results = defaultdict(list)
    for result in results:
        for column in columns:
            merged_results[column].append(result[column])

    return merged_results