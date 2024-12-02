#!/bin/bash

K6_DURATION="1m"

mkdir -p reports

for K6_VUS in 1 2 4 8 12 24 32
do
  K6_DURATION=$K6_DURATION K6_VUS=$K6_VUS npm run test
done
