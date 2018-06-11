#!/bin/bash
workspace=`pwd`
datadir=$workspace/../ctb_training_data
tooldir=$workspace/../binary_span_clas_chn/bilstm_cky/
model=models/neural.cky.parser._80_200_2-pid12567.params
model=$1
function run()
{
    $tooldir/$1 --embedding $datadir/zzgiga.sskip.80.vectors.nofirstline --unkmap_file ${datadir}/ctb.all.unk.dict\
        --train ${datadir}/train.autopos.tree.bin \
        --dev ${datadir}/dev.autopos.tree.bin \
        --test ${datadir}/test.autopos.tree.bin \
        --dynet-mem 2200 \
        --dynet-seed 123456789 \
        --trainer sgd -lr 0.1 -l 2 --eta_decay 0.05\
        --token_input_dim 80\
        --output_hidden_dim 200\
        --hidden_dim 200\
        --model $model\
        --istest 1\
        --use_char 1\
        --tree_dropout 0.0\
        --idbuilder ntparser\
        --init_from_pretrained 1\
        --dev_output_file dev.clas.notreedrop.tensor.l32.output\
        --test_output_file test.clas.notreedrop.tensor.l32.output
}

run BILSTM-CKY-PARSER
