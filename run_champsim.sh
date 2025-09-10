# run_champsim.sh
# This script automates the process of running ChampSim simulations with various configurations.

# Usage: ./run_champsim.sh <output_file_name>
# !/bin/bash

cd /home/eyal/ChampSim
./config.sh champsim_config.json
make
bin/champsim --warmup-instructions 200000000 --simulation-instructions 500000000 ~/trace_ex/benchbase-tpcc.champsim.trace.gz > $1
echo "Simulation complete. Output saved to $1"

