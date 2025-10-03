# run_champsim.sh
# This script automates the process of running ChampSim simulations with various configurations.

# Usage: ./run_champsim.sh <output_file_name>
# !/bin/bash

./config.sh champsim_config.json
make
bin/champsim --warmup-instructions 200000000 --simulation-instructions 500000000 traces/benchbase-tpcc.champsim.trace.gz | tee $1
echo "Simulation complete. Output saved to $1"

