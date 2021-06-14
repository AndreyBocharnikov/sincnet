#!/bin/bash

mv datasets/timit/TEST datasets/timit/test
mv datasets/timit/TRAIN datasets/timit/train

for st in "train" "test"
do
	for i in $( find datasets/timit/"$st" -maxdepth 1 -type d ); do mv -i $i `echo $i | tr 'A-Z' 'a-z'`; done

	for dr in "dr1" "dr2" "dr3" "dr4" "dr5" "dr6" "dr7" "dr8"
	do
		for i in $( find datasets/timit/"$st"/"$dr" -maxdepth 1 -type d ); do mv -i $i `echo $i | tr 'A-Z' 'a-z'`; done
		for j in $( find datasets/timit/"$st"/"$dr" -type f ); do mv -i $j `echo $j | tr 'A-Z' 'a-z'`; done
	done
done

#for j in $(find datasets/timit/test/dr1 -type f)
#do
#	mv -i $j `echo $j | tr 'A-Z' 'a-z'`
#done
