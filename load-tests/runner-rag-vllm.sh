#!/bin/bash

K6_DURATION="1m"

mkdir -p reports/rag-vllm

for K6_VUS in 1 2 4 8 12 24 32 48 64 84
do
  K6_DURATION=$K6_DURATION K6_VUS=$K6_VUS npm run rag-vllm
done
