#!/bin/bash

cat summary.txt |tr '\n' '|'|sed -e 's%||%@@%g'|tr '@' '\n'|tr '|' ' '
