#!/bin/bash -eu

allowed_shots=(1e7 2e7 3e7 4e7 5e7 6e7 8e7 9e7 1e8)
methods=('heat-annealing' 'BFGS' 'Powell' 'Nelder-Mead')
count=1

for method in ${methods[@]}
do
  for shots in ${allowed_shots[@]}
  do
    echo $method $shots
    sed -i -e "s/int(1e8)/int(${shots})/g" ./qparcchallenge2022/qparc.py
    poetry run python evaluation.py $method $shots > evaluate_data/${method}_${shots}.txt
    sed -i -e "s/int(${shots})/int(1e8)/g" ./qparcchallenge2022/qparc.py
  done
done
