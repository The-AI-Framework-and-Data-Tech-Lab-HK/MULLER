#!/usr/bin/env python3
import gc
import sys
import muller
import time
import numpy as np
import argparse

# Constants
COLUMN_NAMES = {
    "vectorized_filter": "l_linenumber",
    "inverted_index": "text",
}
SEPARATE_NUM = 10
QUERY_NUM = 10
TOTAL_VECTORS = 1000000
VECTOR_DIMENSION = 960


def inverted_index_search(dataset, keyword):
    """Execute inverted index search for a single keyword"""
    col_name = COLUMN_NAMES["inverted_index"]

    try:
        start_time = time.perf_counter()
        result = dataset.filter_vectorized([(col_name, "CONTAINS", keyword)])
        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000
        result_count = len(result) if hasattr(result, '__len__') else 0

        print(f"RESULT: inverted_index,{keyword},{elapsed_ms:.3f},{result_count}")

        del result
        gc.collect()

        return elapsed_ms, result_count

    except Exception as e:
        print(f"ERROR: Inverted index search failed for keyword '{keyword}': {e}", file=sys.stderr)
        return None, None


def generate_run_hnsw(dataset_path):
    """Generate HNSW index and run vector search testing."""
    print(f"Generating {TOTAL_VECTORS:,} random vectors...")

    gist_data = np.random.rand(TOTAL_VECTORS, VECTOR_DIMENSION).astype(np.float32)

    ds = muller.dataset(dataset_path, overwrite=True)

    try:
        with ds:
            ds.create_tensor(
                name="gist", htype="vector", dtype="float32", dimension=VECTOR_DIMENSION
            )
            ds.gist.extend(gist_data)

            del gist_data
            gc.collect()

            ds.commit("save original data of gist.")

            print("Building HNSW index...")
            ds.create_vector_index(
                tensor_name="gist",
                index_name="hnsw",
                index_type="HNSWFLAT",
                metric="l2",
                nlist=1000,
                m=96,
            )
            print("HNSW index generated successfully")

    except Exception as e:
        print(f"ERROR: HNSW index generation failed: {e}", file=sys.stderr)
        raise

    # Run performance tests
    query_counts = np.linspace(1, QUERY_NUM, SEPARATE_NUM, dtype=int)

    for idx, q_num in enumerate(query_counts, 1):
        try:
            query_vectors = np.random.rand(q_num, VECTOR_DIMENSION).astype(np.float32)

            start_time = time.perf_counter()
            results = ds.vector_search(
                query_vector=query_vectors,
                tensor_name="gist",
                index_name="hnsw",
                topk=10,
                nprobe=10,
                refine_factor=2,
            )
            end_time = time.perf_counter()

            elapsed_ms = (end_time - start_time) * 1000
            qps = q_num / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
            avg_latency = elapsed_ms / q_num

            print(f"RESULT: hnsw,{q_num},{elapsed_ms:.3f},{qps:.2f},{avg_latency:.3f}")

            del query_vectors
            del results
            gc.collect()

        except Exception as e:
            print(f"ERROR: HNSW search failed with {q_num} queries: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Filter and Vector Search Benchmark")
    parser.add_argument("test_type", choices=["inv", "hnsw"],
                        help="Type of test to run")
    parser.add_argument("-d", "--dataset-path", required=True,
                        help="Path to dataset")
    parser.add_argument("-k", "--keyword",
                        help="Keyword for inverted index search")

    args = parser.parse_args()

    if args.test_type == "inv":
        if not args.keyword:
            print("ERROR: --keyword is required for inverted index test", file=sys.stderr)
            sys.exit(1)

        try:
            dataset = muller.load(args.dataset_path)
            elapsed_ms, result_count = inverted_index_search(dataset, args.keyword)

            if elapsed_ms is None:
                sys.exit(1)

        except Exception as e:
            print(f"ERROR: Failed to load dataset: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.test_type == "hnsw":
        try:
            generate_run_hnsw(args.dataset_path)
        except Exception as e:
            print(f"ERROR: Failed to run HNSW test: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()