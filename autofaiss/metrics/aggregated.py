"""Aggregated metrics"""

import json
import pandas as pd
import graphyte


def maxd(l):
    if len(l) == 0:
        return -1
    return max(l)


def mind(l):
    if len(l) == 0:
        return -1
    return min(l)


def avg(l):
    if len(l) == 0:
        return -1
    return sum(l) / len(l)


def compute_aggregated_metrics(indices_data):
    """Compute aggregated metrics from indices df"""
    max_reconstruction_error = maxd(
        [json.loads(index_data["metrics"])["reconstruction error %"] for index_data in indices_data]
    )
    min_1_recall_40 = mind([json.loads(index_data["metrics"])["1-recall@40"] for index_data in indices_data])
    avg_1_recall_40 = avg([json.loads(index_data["metrics"])["1-recall@40"] for index_data in indices_data])
    min_40_recall_40 = mind([json.loads(index_data["metrics"])["40-recall@40"] for index_data in indices_data])
    avg_40_recall_40 = avg([json.loads(index_data["metrics"])["40-recall@40"] for index_data in indices_data])
    max_avg_search_speed_ms = maxd(
        [json.loads(index_data["metrics"])["avg_search_speed_ms"] for index_data in indices_data]
    )
    avg_avg_search_speed_ms = avg(
        [json.loads(index_data["metrics"])["avg_search_speed_ms"] for index_data in indices_data]
    )
    max_99p_search_speed = maxd(
        [json.loads(index_data["metrics"])["99p_search_speed_ms"] for index_data in indices_data]
    )
    avg_99p_search_speed = avg(
        [json.loads(index_data["metrics"])["99p_search_speed_ms"] for index_data in indices_data]
    )
    total_compressed_size = sum([json.loads(index_data["metrics"])["size in bytes"] for index_data in indices_data])
    embedding_count = sum([json.loads(index_data["metrics"])["nb vectors"] for index_data in indices_data])
    dimension = indices_data[0]["dimension"] if len(indices_data) > 0 else -1
    initial_size = 4 * dimension * embedding_count
    total_compression = initial_size / total_compressed_size if total_compressed_size != 0 else -1
    return {
        "count": len(indices_data),
        "dimension": dimension,
        "embedding_count": embedding_count,
        "initial_size": initial_size,
        "total_compressed_size": total_compressed_size,
        "total_compression_ratio": total_compression,
        "max_avg_search_speed_ms": max_avg_search_speed_ms,
        "avg_avg_search_speed_ms": avg_avg_search_speed_ms,
        "max_99p_search_speed": max_99p_search_speed,
        "avg_99p_search_speed": avg_99p_search_speed,
        "min_1_recall_40": min_1_recall_40,
        "avg_1_recall_40": avg_1_recall_40,
        "min_40_recall_40": min_40_recall_40,
        "avg_40_recall_40": avg_40_recall_40,
        "max_reconstruction_error": max_reconstruction_error,
    }


def metrics_per_subsets(indices_data):
    """Compute aggregated metrics per subset"""

    subsets = {
        "all": lambda index_data: True,
        "flat": lambda index_data: json.loads(index_data["indexParams"])["indexKey"] == "Flat",
        "hnsw": lambda index_data: json.loads(index_data["indexParams"])["indexKey"] == "HNSW15",
        "quantized": lambda index_data: "IVF" in json.loads(index_data["indexParams"])["indexKey"],
        "recommendable": lambda index_data: index_data["isRecommendable"],
        "non_recommendable": lambda index_data: not index_data["isRecommendable"],
    }
    return {
        subset_name: compute_aggregated_metrics([index_data for index_data in indices_data if subset_pred(index_data)])
        for subset_name, subset_pred in subsets.items()
    }


def compute_aggregated_metrics_all(data):
    """Compute aggregated metrics after reading from hdfs"""

    metrics = metrics_per_subsets(data)

    pd.set_option("precision", 2)
    pd.set_option("display.float_format", lambda x: "%.2f" % x)
    return pd.DataFrame(metrics), metrics


def send_metrics(
    raw,
    platform,
    country=None,
    graphite_prefix="criteo.deepr.python.indices",
    graphite_host="graphite-relay.storage.criteo.preprod",
):
    """Send metrics to graphite"""
    host = graphite_host
    port = 3341
    timeout = 5
    interval = 60
    queue_size = None
    log_sends = False
    protocol = "tcp"
    batch_size = 1000
    graphite_prefix = (
        f"{graphite_prefix}.perCountry.{platform}.{country}"
        if country is not None
        else f"{graphite_prefix}.perPlatform.{platform}"
    )
    metrics = {}
    for group, v1 in raw.items():
        for metric, v2 in v1.items():
            metrics[group + "." + metric] = v2

    sender = graphyte.Sender(
        host=host,
        port=port,
        prefix=graphite_prefix,
        timeout=timeout,
        interval=interval,
        queue_size=queue_size,
        log_sends=log_sends,
        protocol=protocol,
        batch_size=batch_size,
    )
    for metric, value in metrics.items():
        sender.send(metric, value)
