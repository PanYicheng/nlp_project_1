#!/bin/bash
source ~/torchenv/bin/activate
python main.py --data ./rocstory_data/ --save ./storymodels/model.pt \
	--model GRU --emsize 800 --nhid 800 --nlayers 3 \
	--lr 10 --clip 0.25 --epochs 500 --batch_size 80 --bptt 70 \
	--alpha 0.1 \
	--dropoute 0.1 --dropouti 0.1 --dropout 0.1 --dropoutrnn 0.1 \
	--seed 42 --nonmono 5 --cuda --log-interval 200 \
	--when 300 400
