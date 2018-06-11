#!/bin/bash
workspace=`pwd`
datadir=$workspace/../ctb_training_data/
$workspace/../scripts/EVALB/evalb -p $workspace/../scripts/EVALB/COLLINS_ch.prm $datadir/test.goldpos.tree $workspace/$1 1>$workspace/$1.evalfile.log 2>&1
python $workspace/../scripts/get_fmeasure2.py $workspace/$1.evalfile.log

