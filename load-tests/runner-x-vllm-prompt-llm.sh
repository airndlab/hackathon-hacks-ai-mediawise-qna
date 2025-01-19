#!/bin/bash

K6_DURATION="5m"

mkdir -p reports/x-vllm-prompt-llm

K6_DURATION="1m" K6_VUS=14 SLEEP_TIMEOUT=30 npm run vllm-prompt-llm
sleep 30

for SLEEP_TIMEOUT in 15
do
  for K6_VUS in 1 2 4 8 10 12 14 16 20 28 44 76
  do
    K6_DURATION=$K6_DURATION K6_VUS=$K6_VUS SLEEP_TIMEOUT=$SLEEP_TIMEOUT npm run vllm-prompt-llm
    sleep 30
  done
done
