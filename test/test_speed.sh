#!/bin/bash

N=20

for ((i=1; i<=N; i++)); do
    python test/test_http_server.py &
    sleep 1
done
