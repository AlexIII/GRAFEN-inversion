#!/bin/bash
mpiexec.mpich -l -machinefile `dirname "$0"`/hosts.txt ${@:1}
