#!/bin/bash
source ~/torchenv/bin/activate
python generate.py --data ./rocstory_data/ \
	--conditional_data ./rocstory_data/test.txt.generateinput \
	--print_cond_data --checkpoint ./storymodels/model.pt \
    --cuda --nsents 1000 --words 100 \
	--temperature 0.5
