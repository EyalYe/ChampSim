#parse_runs.py

import os
import json
import matplotlib.pyplot as plt
import numpy as np

RUNS_DIR = 'sim_output'

def load_runs_file_list(runs_dir):
    """Load the list of run files from the specified directory."""
    run_files = []
    for filename in os.listdir(runs_dir):
        if filename.endswith('txt'):
            run_files.append(os.path.join(runs_dir, filename))
    return run_files

def parse_run_file(file_list, runs_dir=RUNS_DIR):
    """Parse a single run file and return its content as a dictionary."""
    report = {}
    l1d_report = {}
    l2c_report = {}
    for file in file_list:
        file_path = file
        file = file.replace(runs_dir+'/', '').replace('.txt', '').strip().split('_')
        L1D = ''
        L2C = ''
        L1D_parsed = False
        for i in file:
            if i in ['L1D', 'L2C']:
                if i == 'L2C':
                    L1D_parsed = True
            else:
                if not L1D_parsed:
                    L1D = L1D + i if L1D == '' else L1D + '_' + i
                else:
                    L2C = L2C + i if L2C == '' else L2C + '_' + i
        print(f"L1D: {L1D}, L2C: {L2C}")
        if L2C == 'no':
            relavent_line_header = 'cpu0->cpu0_L1D TOTAL'
        else:
            relavent_line_header = 'cpu0->cpu0_L2C TOTAL'
        with open(file_path, 'r') as f:
            report_key = f"L1D-{L1D}_L2C-{L2C}"
            for line in f:
                if relavent_line_header in line:
                    parts = line.replace(relavent_line_header, '').strip().split()
                    try:
                        acesses, miss, hit, mshr = parts[1], parts[3], parts[5], parts[7]
                        miss_rate = (int(miss) / int(acesses)) * 100 if int(acesses) > 0 else 0
                        hit_rate = (int(hit) / int(acesses)) * 100 if int(acesses) > 0 else 0
                        report[report_key] = {
                            'accesses': int(acesses),
                            'misses': int(miss),
                            'hits': int(hit),
                            'mshr': int(mshr),
                            'miss_rate': miss_rate,
                            'hit_rate': hit_rate
                        }
                        if L1D == 'no':
                            l2c_report[L2C] = report[report_key]
                        else:
                            l1d_report[L1D] = report[report_key]
                        break  # Exit after finding the relevant line
                    except IndexError as e:
                        print(f"Error parsing line: {line.strip()} - {e}")
                        continue
            else:
                print(f"No relevant line found in file: {file_path}")
                report[report_key] = {
                    'accesses': 0,
                    'misses': 0,
                    'hits': 0,
                    'mshr': 0,
                    'miss_rate': 0,
                    'hit_rate': 0
                }
                if L1D == 'no':
                    l2c_report[L2C] = report[report_key]
                else:
                    l1d_report[L1D] = report[report_key]
    return report, l1d_report, l2c_report

def output_report(report, output_file='report.json', output_dir='reports'):
    os.makedirs(output_dir, exist_ok=True)
    """Output the report dictionary to a JSON file."""
    with open(os.path.join(output_dir, output_file), 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Report written to {output_file}")


def output_graph(report, output_file='report_graph.jpg', output_dir='reports/graphs', sort_by=None):
    """
    Render a grouped bar chart: x = prefetcher name, y = miss/hit rate side-by-side.

    Args:
        report: dict like {"name": {"miss_rate": <float>, "hit_rate": <float>}, ...}
                rates can be in [0,1] or [0,100]
        output_file: filename to save (e.g., 'report_graph.jpg')
        output_dir: directory to save into
        sort_by: None | 'miss_rate' | 'hit_rate' to sort descending for readability
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect items; keep insertion order unless sorting requested
    items = [(k, v.get('miss_rate', 0.0), v.get('hit_rate', 0.0)) for k, v in report.items()]
    if sort_by in ('miss_rate', 'hit_rate'):
        idx = 1 if sort_by == 'miss_rate' else 2
        items.sort(key=lambda t: t[idx], reverse=True)

    labels = [k for k, _, _ in items]
    miss = np.array([m for _, m, _ in items], dtype=float)
    hit  = np.array([h for _, _, h in items], dtype=float)

    # If values look like fractions, convert to percentages
    max_val = float(np.nanmax([miss.max() if miss.size else 0, hit.max() if hit.size else 0]))
    as_percent = max_val <= 1.0
    if as_percent:
        miss *= 100.0
        hit  *= 100.0

    x = np.arange(len(labels))
    width = 0.42  # bar width within each group

    # Scale figure width with number of configs for readability
    fig_w = max(10, 0.7 * len(labels) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    r1 = ax.bar(x - width/2, miss, width, label='Miss rate')
    r2 = ax.bar(x + width/2, hit,  width, label='Hit rate')

    ax.set_xlabel('Prefetcher')
    ax.set_ylabel('Rate (%)')
    ax.set_title('Cache Miss vs Hit Rates by Prefetcher')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right')

    # Helpful grid & bounds
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    if as_percent or (miss.max() <= 100 and hit.max() <= 100):
        ax.set_ylim(0, 100)

    ax.legend()

    # Add small value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}%',
                        xy=(rect.get_x() + rect.get_width()/2, h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)
    autolabel(r1)
    autolabel(r2)

    fig.tight_layout()
    out_path = os.path.join(output_dir, output_file)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Graph saved to {out_path}")

def main():
    run_files = load_runs_file_list(RUNS_DIR)
    report, l1d_report, l2c_report = parse_run_file(run_files)
    output_report(report)
    output_report(l1d_report, output_file='l1d_report.json')
    output_report(l2c_report, output_file='l2c_report.json')
    output_graph(report)
    output_graph(l1d_report, output_file='l1d_report_graph.jpg', sort_by='miss_rate')
    output_graph(l2c_report, output_file='l2c_report_graph.jpg', sort_by='miss_rate')

def parse_runs():
    run_files = load_runs_file_list(RUNS_DIR)
    report, l1d_report, l2c_report = parse_run_file(run_files)
    output_report(report)
    output_report(l1d_report, output_file='l1d_report.json')
    output_report(l2c_report, output_file='l2c_report.json')
    output_graph(report)
    output_graph(l1d_report, output_file='l1d_report_graph.jpg', sort_by='miss_rate')
    output_graph(l2c_report, output_file='l2c_report_graph.jpg', sort_by='miss_rate')

if __name__ == "__main__":
    main()
