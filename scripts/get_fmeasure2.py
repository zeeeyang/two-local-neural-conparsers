#!/usr/bin/env python
# coding=utf-8
import sys
import traceback

if len(sys.argv) != 2:
    print 'Usage: %s input' % sys.argv[0]
    sys.exit(0)

pattern = "Bracketing FMeasure       ="
input_lines = [ x.strip() for x in open(sys.argv[1], "r") ]
num_lines = len(input_lines)
for i in range(num_lines):
    line = input_lines[i]
    if line.find(pattern) != -1:
        value = float(line.replace(pattern, "").strip())
        print value
        break
