#!/bin/bash

docker run -it \
    --detach \
	--name gpu \
	--gpus all \
    sincnet
