#!/usr/bin/env bash

curl micro.mamba.pm/install.sh | bash
micromamba shell completion

echo "+-----------------------------------------------+"
echo "| Done installing mamba                         |"
echo "| Now run: mamba env create -f environment.yml  |"
echo "| This step may take several minutes            |"
echo "+-----------------------------------------------+"
