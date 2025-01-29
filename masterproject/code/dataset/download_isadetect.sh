#!/bin/bash

mkdir -p ISAdetect/ISAdetect_full_dataset

cd ISAdetect/ISAdetect_full_dataset

curl 'https://etsin.fairdata.fi/api/download/authorize' -H 'Content-Type: application/json' -d '{"cr_id":"9f6203f5-2360-426f-b9df-052f3f936ed2","file":"/new_new_dataset/ISAdetect_full_dataset.tar.gz"}' | jq -r '.url' | xargs curl -fo ISAdetect_full_dataset.tar.gz

tar -xzf ISAdetect_full_dataset.tar.gz

rm ISAdetect_full_dataset.tar.gz