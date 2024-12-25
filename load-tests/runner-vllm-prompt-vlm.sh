#!/bin/bash

K6_DURATION="3m"

mkdir -p reports/vllm-prompt-vlm

for SLEEP_TIMEOUT in 15 30 45
do
  for K6_VUS in 12 24 26 28 30 32
  do
    K6_DURATION=$K6_DURATION K6_VUS=$K6_VUS SLEEP_TIMEOUT=$SLEEP_TIMEOUT npm run vllm-prompt-vlm
  done
done
