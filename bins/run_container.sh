#!/bin/bash

image_name=sincnet
project_name=sincnet
username=$(whoami)
container_name=${username}-${image_name}

docker stop "${container_name}"
docker rm "${container_name}"

docker run -it \
    --detach \
    086bb8a82fdf
