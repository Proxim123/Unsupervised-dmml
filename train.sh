#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2,3,4
delta=3.5
lambda=0.55

#########duke 2 market transfer
mu=0.4
xi=0.6
sourceset='duke'
targetset='market'

#############my experiments
dropout=0.3
lr=0.01
momentum=0.09

<<COMMENT
#########market 2 duke transfer
mu=0.5
xi=0.6
sourceset='market'
targetset='duke'
COMMENT

<<COMMENT
#########duke 2 market target only
mu=0.4
xi=1.0
sourceset='duke'
targetset='market'
COMMENT

<<COMMENT
#########market 2 duke target only
mu=0.5
xi=1.0
sourceset='market'
targetset='duke'
COMMENT


python main.py -s ${sourceset} -t ${targetset} --logs-dir checkpoint/AE_${sourceset}2${targetset}_xi_${xi} --data-dir  data/ --lambda $lambda --delta $delta --xi $xi --mu $mu
#python main.py -s ${sourceset} -t ${targetset} --logs-dir checkpoint/AE_${sourceset}2${targetset}_xi_${xi} --data-dir  data/ --lambda $lambda --delta $delta --xi $xi --mu $mu --cuda
