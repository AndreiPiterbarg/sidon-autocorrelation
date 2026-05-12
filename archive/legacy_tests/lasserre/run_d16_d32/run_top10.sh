#!/bin/bash
# Top 10 cascade experiments using run_cascade.py (Gray code + incremental conv)
# All use coarse grid mode (--S flag)
set -e
cd /workspace/sidon-autocorrelation

echo "============================================="
echo "TOP 10 CASCADE EXPERIMENTS — c_target=1.30"
echo "Using: run_cascade.py with Gray code kernel"
echo "============================================="

# Exp 1: d0=2, S=30 — quick baseline
echo -e "\n\n########## EXPERIMENT 1: d0=2, S=30 ##########"
python3.13 -u -m cloninger_steinerberger.cpu.run_cascade --d0 2 --S 30 --c_target 1.30 --max_levels 10

# Exp 2: d0=2, S=50
echo -e "\n\n########## EXPERIMENT 2: d0=2, S=50 ##########"
python3.13 -u -m cloninger_steinerberger.cpu.run_cascade --d0 2 --S 50 --c_target 1.30 --max_levels 10

# Exp 3: d0=4, S=20
echo -e "\n\n########## EXPERIMENT 3: d0=4, S=20 ##########"
python3.13 -u -m cloninger_steinerberger.cpu.run_cascade --d0 4 --S 20 --c_target 1.30 --max_levels 8

# Exp 4: d0=4, S=30
echo -e "\n\n########## EXPERIMENT 4: d0=4, S=30 ##########"
python3.13 -u -m cloninger_steinerberger.cpu.run_cascade --d0 4 --S 30 --c_target 1.30 --max_levels 8

# Exp 5: d0=2, S=75
echo -e "\n\n########## EXPERIMENT 5: d0=2, S=75 ##########"
python3.13 -u -m cloninger_steinerberger.cpu.run_cascade --d0 2 --S 75 --c_target 1.30 --max_levels 10

# Exp 6: d0=2, S=100
echo -e "\n\n########## EXPERIMENT 6: d0=2, S=100 ##########"
python3.13 -u -m cloninger_steinerberger.cpu.run_cascade --d0 2 --S 100 --c_target 1.30 --max_levels 10

# Exp 7: d0=4, S=50
echo -e "\n\n########## EXPERIMENT 7: d0=4, S=50 ##########"
python3.13 -u -m cloninger_steinerberger.cpu.run_cascade --d0 4 --S 50 --c_target 1.30 --max_levels 8

# Exp 8: d0=6, S=30
echo -e "\n\n########## EXPERIMENT 8: d0=6, S=30 ##########"
python3.13 -u -m cloninger_steinerberger.cpu.run_cascade --d0 6 --S 30 --c_target 1.30 --max_levels 6

# Exp 9: d0=6, S=50
echo -e "\n\n########## EXPERIMENT 9: d0=6, S=50 ##########"
python3.13 -u -m cloninger_steinerberger.cpu.run_cascade --d0 6 --S 50 --c_target 1.30 --max_levels 6

# Exp 10: d0=2, S=150
echo -e "\n\n########## EXPERIMENT 10: d0=2, S=150 ##########"
python3.13 -u -m cloninger_steinerberger.cpu.run_cascade --d0 2 --S 150 --c_target 1.30 --max_levels 10

echo -e "\n\n============================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "============================================="
