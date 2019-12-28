#!/bin/bash
#g++ -pthread -fPIC -o3 -std=c++11 ./src/*.cpp index_main.cpp -I./include -o libhnsw.so -shared 
g++ -pthread -fPIC -o3 -std=c++11 ./src/*.cpp -I./include -o libhnsw.so -shared 
