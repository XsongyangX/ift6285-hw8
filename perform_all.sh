#!/bin/bash

jump=2
begin=(0 2 4 6 8)

for i in "${begin[@]}"; do 
    let j=$i+$jump
    pkscreen ssh ens -J arcade "cd ift6285/hw8; python performance.py $i $j > performance_$i-$j.log"
done