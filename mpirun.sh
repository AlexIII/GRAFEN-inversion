#!/bin/bash
mpiexec.mpich -l -machinefile hosts.txt ${@:1}
