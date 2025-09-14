# run_options.py 

# This script takes a list of all available prefetchers and runs them accross L1D and L2C

import os
import subprocess
import sys
import time 
import json
import argparse
from parse_runs import parse_runs

# List of all available prefetchers
PREFETCHERS = [
    "no",
    "next_line",
    "ip_stride",
    "spp_dev",
    "va_ampm_lite"
]

# List of cache levels
CACHE_LEVELS = [
    "L1D",
    "L2C"
]

# file containing list of prefetchers
PREFETCHERS_FILE = "prefetchers.txt"

# output directory
OUTPUT_DIR = "sim_output"

# import json file
CONFIG_FILE = "champsim_config.json"

# load config file to read
def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

# write config file
def write_config(config_file, config):
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)

# itierate through all combinations of prefetchers on L1D only

def run_sim(is_l1d, is_l2c, config):
    for prefetcher in PREFETCHERS:
        if is_l1d and not is_l2c:
            l1d_prefetcher = prefetcher
            config["L1D"]["prefetcher"] = l1d_prefetcher
            l2c_prefetcher = "no"
            config["L2C"]["prefetcher"] = l2c_prefetcher
        elif not is_l1d and is_l2c:
            l1d_prefetcher = "no"
            config["L1D"]["prefetcher"] = l1d_prefetcher
            l2c_prefetcher = prefetcher
            config["L2C"]["prefetcher"] = l2c_prefetcher
        else:
            continue
        write_config(CONFIG_FILE, config)
        print(f"Running simulation with L1D prefetcher: {l1d_prefetcher}, L2C prefetcher: {l2c_prefetcher}")
        # Run the simulation
        start_time = time.time()
        subprocess.run([f"./run_champsim.sh {OUTPUT_DIR}/L1D_{l1d_prefetcher}_L2C_{l2c_prefetcher}.txt"], shell=True)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Simulation completed in {elapsed_time:.2f} seconds\n")


def main():
    global CONFIG_FILE, OUTPUT_DIR, PREFETCHERS


    parser = argparse.ArgumentParser(description="Run simulations with different prefetchers on L1D and L2C caches.")
    parser.add_argument('--config', type=str, default=CONFIG_FILE, help='Path to the configuration JSON file.')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR, help='Directory to store simulation outputs.')
    parser.add_argument('--all_prefetchers', action='store_true', help='Run simulations with all prefetchers on both L1D and L2C.')
    parser.add_argument('--l1d_only', action='store_true', help='Run simulations with different prefetchers on L1D only.')
    parser.add_argument('--l2c_only', action='store_true', help='Run simulations with different prefetchers on L2C only.')
    parser.add_argument('--prefetchers', type=str, nargs='+', default=PREFETCHERS, help='List of prefetchers to use.')
    args = parser.parse_args()

    CONFIG_FILE, OUTPUT_DIR = args.config, args.output

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if not os.path.exists(CONFIG_FILE):
        print(f"Configuration file {CONFIG_FILE} does not exist.")
        sys.exit(1)

    config = load_config(CONFIG_FILE)

    if args.prefetchers and not args.all_prefetchers:
        print(f"Using specified prefetchers: {args.prefetchers}")
        PREFETCHERS = args.prefetchers
    
    if args.l1d_only and args.l2c_only:
        print("Cannot specify both --l1d_only and --l2c_only. Please choose one.")
        sys.exit(1)

    if args.l1d_only:
        print("Running simulations with different prefetchers on L1D only:")
        config = load_config(CONFIG_FILE)
        run_sim(is_l1d=True, is_l2c=False, config=config)
        parse_runs()
        sys.exit(0)
    elif args.l2c_only:
        print("Running simulations with different prefetchers on L2C only:")
        config = load_config(CONFIG_FILE)
        run_sim(is_l1d=False, is_l2c=True, config=config)
        parse_runs()
        sys.exit(0)
    else:
        print("Running simulations with all prefetchers on both L1D and L2C:")
        config = load_config(CONFIG_FILE)
        run_sim(is_l1d=True, is_l2c=False, config=config)
        config = load_config(CONFIG_FILE)
        run_sim(is_l1d=False, is_l2c=True, config=config)
        parse_runs()
        sys.exit(0)




if __name__ == "__main__":
    main()
