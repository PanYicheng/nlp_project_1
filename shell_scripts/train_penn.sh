#!/bin/bash
source ~/torchenv/bin/activate
python main.py --data ./data/penn/ --save ./data/penn/model.pt \
	--model GRU --emsize 800 --nhid 800 --nlayers 3 \
	--lr 10 --clip 0.25 --epochs 500 --batch_size 80 --bptt 70 \
	--alpha 0.1 --wdrop 0.5 \
	--dropoute 0.5 --dropouti 0.5 --dropout 0.5 --dropoutrnn 0.5 \
	--seed 42 --nonmono 5 --cuda --log-interval 200 \
	--when 300 400
