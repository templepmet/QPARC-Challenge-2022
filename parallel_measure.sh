#!/bin/bash -eu

# allowed_shots=(1e7 2e7 3e7 4e7 5e7 6e7 7e7 8e7 9e7 1e8)
allowed_shots=(2e7 3e7 4e7 5e7 6e7 7e7 8e7 9e7 1e8)
methods=('heat-annealing' 'BFGS' 'Powell' 'Nelder-Mead')
count=(2 3 4 5)

for shots in ${allowed_shots[@]}
do
	sed -i -e "s/int(1e8)/int(${shots})/g" ./qparcchallenge2022/qparc.py
	parallel --bar "pipenv run python evaluation.py {1} > evaluate_data/{1}_${shots}_{2}.txt" ::: ${methods[@]} ::: ${count[@]}
	sed -i -e "s/int(${shots})/int(1e8)/g" ./qparcchallenge2022/qparc.py
done
