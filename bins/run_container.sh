#!/bin/bash

docker run -it \
  --detach \
	--name cuda0 \
	--gpus device=0
	sincnet
