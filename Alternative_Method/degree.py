import os
import pandas as pd
import networkx as nx
import argparse
import re
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time
import numpy as np

def compute_average(metrics):
    """Compute the average of metrics, handling empty dictionaries."""
    if not metrics:
        return 0
    return sum(metrics.values()) / len(metrics)

def compute_network_metrics(G):
    """Compute various network metrics and return as a dictionary. Handle empty graphs."""
    if len(G) == 0:
        return {
            'avg_degree_centrality': 0,
            'avg_pagerank': 0,
            'avg_closeness_centrality': 0,
            'density': 0,
            'avg_eigenvector_centrality': 0,
            'avg_betweenness_centrality': 0,
        }

    try:
        degree_centrality = nx.degree_centrality(G)
        pagerank = nx.pagerank(G)
        closeness_centrality = nx.closeness_centrality(G)
        density = nx.density(G)
        eigenvector_centrality = nx.eigenvector_centrality_numpy(G)
        betweenness_centrality = nx.betweenness_centrality(G)

        # Replace nan values with 0
        degree_centrality = {k: (v if not np.isnan(v) else 0) for k, v in degree_centrality.items()}
        pagerank = {k: (v if not np.isnan(v) else 0) for k, v in pagerank.items()}
        closeness_centrality = {k: (v if not np.isnan(v) else 0) for k, v in closeness_centrality.items()}
        eigenvector_centrality = {k: (v if not np.isnan(v) else 0) for k, v in eigenvector_centrality.items()}
        betweenness_centrality = {k: (v if not np.isnan(v) else 0) for k, v in betweenness_centrality.items()}

        return {
            'avg_degree_centrality': compute_average(degree_centrality),
            'avg_pagerank': compute_average(pagerank),
            'avg_closeness_centrality': compute_average(closeness_centrality),
            'density': density,
            'avg_eigenvector_centrality': compute_average(eigenvector_centrality),
            'avg_betweenness_centrality': compute_average(betweenness_centrality),
        }
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return {
            'avg_degree_centrality': 0,
            'avg_pagerank': 0,
            'avg_closeness_centrality': 0,
            'density': 0,
            'avg_eigenvector_centrality': 0,
            'avg_betweenness_centrality': 0,
        }

def process_file(args):
    """Process a single file."""
    arc_file, df_file, fellow_ids = args
    try:
        start_time = time.time()
        arc = pd.read_csv(arc_file, header=None, delimiter='\t', usecols=[0, 1], names=['source', 'target'])
        arc_read_time = time.time()
        df = pd.read_csv(df_file, header=None)
        df_read_time = time.time()
        G = nx.from_pandas_edgelist(arc, 'source', 'target', create_using=nx.Graph())
        graph_construct_time = time.time()
        
        if len(G) == 0:
            print(f"Warning: No data in graph for file {arc_file}")
            return None

        author_id = re.findall(r'\d+', os.path.basename(arc_file))[0]
        base_metrics = {
            'author_id': author_id,
            'counts_links': len(arc),
            'counts_papers': len(df)
        }

        metrics = compute_network_metrics(G)
        metrics_time = time.time()
        base_metrics.update(metrics)
        base_metrics['is_awarded'] = 1 if author_id in fellow_ids else 0

        total_time = time.time() - start_time
        print(f"Processed file {arc_file} in {total_time:.2f} seconds (arc_read: {arc_read_time - start_time:.2f}, df_read: {df_read_time - arc_read_time:.2f}, graph_construct: {graph_construct_time - df_read_time:.2f}, metrics: {metrics_time - graph_construct_time:.2f})")

        return base_metrics
    except Exception as e:
        print(f"Error processing file {arc_file}: {e}")
        return None

def process_files_parallel(directory, prefix, fellow_df):
    """Process files in the directory using parallel processing."""
    fellow_ids = set(fellow_df[0].astype(str).tolist())
    files = [
        (os.path.join(root, name), os.path.join(root, "papers" + name[5:]), fellow_ids)
            for root, _, filenames in os.walk(directory)
            for name in filenames if name.startswith(prefix)]

    print(f"Total files to process: {len(files)}")

    with Pool(cpu_count() // 2) as pool:  # Reduce the number of parallel processes
        results = list(tqdm(pool.imap(process_file, files), total=len(files), desc="Processing files", unit="file"))

    return [res for res in results if res is not None]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--field', type=str, default='IR', help='specify the research field')
    parser.add_argument('--PATH', type=str, default="data_origin", help='specify the research field')
    args = parser.parse_args()

    print("Loading fellow data...")
    fellow_path = f"../Process_data/{args.field}_data/fellow.csv"
    fellow_df = pd.read_csv(fellow_path, header=None)
    print(f"Loaded fellow data: {len(fellow_df)} rows")

    directory = f"../Process_data/{args.field}_data/data/{args.PATH}"
    print(f"Processing directory: {directory}")
    network_features = process_files_parallel(directory, 'l', fellow_df)

    if not network_features:
        print("No network features processed.")
        return

    network_df = pd.DataFrame(network_features)
    os.makedirs("field", exist_ok=True)
    csv_filename = os.path.join("field", f"{args.field}_{args.PATH}_network_features.csv")
    network_df.to_csv(csv_filename, index=False)
    print(f"Saved network features to {csv_filename}")

if __name__ == "__main__":
    main()
