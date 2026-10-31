#!/usr/bin/env bash
# Step 1 on the supercomputer. Run on the LOGIN node:   bash scripts/cluster/01_check_cluster.sh
# It only reads information; it changes nothing. Paste the whole output back to Claude so the
# job scripts can be tailored (partition names, GPU type, modules, internet access, disk).
echo "===== host / OS"; hostname; uname -srm; cat /etc/os-release 2>/dev/null | grep PRETTY
echo; echo "===== scheduler"
if command -v sinfo >/dev/null; then
  echo "SLURM found"; sinfo -o "%P %a %l %D %G %c %m" 2>/dev/null | head -25
  echo "--- my accounts / QOS"; sacctmgr -nP show assoc user=$USER format=account,partition,qos 2>/dev/null | head -10
elif command -v qstat >/dev/null; then echo "PBS/Torque found"; qstat -Q 2>/dev/null | head -20
else echo "no SLURM or PBS command found"; fi
echo; echo "===== GPUs visible on this node (often none on login nodes)"; nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv 2>/dev/null || echo "nvidia-smi not available here"
echo; echo "===== module system"
if command -v module >/dev/null 2>&1 || [ -n "$MODULESHOME" ]; then
  for m in python anaconda miniconda conda cuda java jdk openjdk gcc; do
    module -t avail $m 2>&1 | grep -i "^$m\|/$m" | head -4
  done
else echo "no 'module' command"; fi
echo; echo "===== tools"; for c in python3 conda mamba pip java git git-lfs huggingface-cli; do printf "%-16s " $c; command -v $c || echo "-"; done
python3 --version 2>&1; java -version 2>&1 | head -1
echo; echo "===== internet from login node"
for u in https://huggingface.co https://pypi.org/simple/ https://github.com; do
  printf "%-32s " $u; curl -s -o /dev/null -m 10 -w "%{http_code}\n" $u || echo "no access"
done
echo; echo "===== disk space and quotas"
echo "HOME=$HOME"; df -h "$HOME" 2>/dev/null | tail -1
for d in /scratch /scratch/$USER $SCRATCH /work /work/$USER /data /lustre; do [ -d "$d" ] && { printf "%-24s " "$d"; df -h "$d" | tail -1; }; done
(quota -s 2>/dev/null || lfs quota -hu $USER /scratch 2>/dev/null) | head -8
echo; echo "===== done"
