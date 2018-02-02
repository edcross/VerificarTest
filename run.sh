#!/bin/bash
for number in 1 2 3
do


echo "script" + $number

python test$number.py > log$number.txt

done
echo "Finish"
exit 0