#! /bin/env python3

from dataclasses import dataclass
import sys, re, os, shutil
from typing import Literal
import Aop, cg
import numpy as np

PRECISE_INIT = False

@dataclass
class Config:
    l0: float
    pprr: float
    n_layers: int
    workload_path: str
    target_field_grd: str
    topo_hieghtmap_grd: str
    modelMeshBaseDir = "v_rightSide"
    mode: Literal['init', 'solve', 'continue']
    def pathOf(self, file_name: str):
        return os.path.join(self.workload_path, file_name)

def parseInput() -> Config:
    if len(sys.argv) < 2:
        print(f"Usage: ./{sys.argv[0]} workload_dir [init | solve | continue]")
        print("\t workload_dir/")
        print("\t\t target_field*.grd")
        print("\t\t topo_hieghtmap*.grd")
        print("\t\t config.txt")
        print("\t\t\t n_layers = 81")
        print("\t\t\t [l0 = 60]")
        print("\t\t\t [pprr = 100] # Point Potential Replace Radius, -1 for no PPRR")
        exit(1)

    TARGET_FIELD_REGEXP = re.compile(".*target_field.*\.grd$")
    TOPO_HEIGHTMAP_REGEXP = re.compile(".*topo_hieghtmap.*\.grd$")

    # get file names
    workloadPath = sys.argv[1]
    workloadPathFiles = Aop.files(workloadPath)
    target_field_grd = os.path.basename(next(fname for fname in workloadPathFiles if TARGET_FIELD_REGEXP.match(fname)))
    topo_hieghtmap_grd = os.path.basename(next(fname for fname in workloadPathFiles if TOPO_HEIGHTMAP_REGEXP.match(fname)))

    # parse config
    config = {}
    with open(workloadPath + '/config.txt', 'r') as f:
        config = { k : v for [k, v] in ( [ token.strip() for token in line.split('#')[0].split('=')] for line in f.readlines() ) }

    mode = sys.argv[2] if len(sys.argv) > 2 else 'solve'

    return Config(
        workload_path = workloadPath,
        target_field_grd = target_field_grd,
        topo_hieghtmap_grd = topo_hieghtmap_grd,
        l0 = float(config.get('l0', 60)),
        pprr = float(config.get('pprr', 100)),
        n_layers = int(config['n_layers']),
        mode = mode
    )

def solverInit(config: Config):
    # Create self.vRightDir - right-side vector = B^T(f), B - forward problem operator
    # Make model_mesh dir with layer files
    modelMeshDir_path = config.pathOf(config.modelMeshBaseDir)
    target_field_path = config.pathOf(config.target_field_grd)
    os.makedirs(modelMeshDir_path, exist_ok=True)
    for l in range(config.n_layers):
        shutil.copyfile(target_field_path, os.path.join(modelMeshDir_path, f"layer_{l:04d}.grd"))
    shutil.copyfile(target_field_path, os.path.join(modelMeshDir_path, f"zzz_topo_dens.grd"))
    # Compute values for v_rightSide
    Aop.SolveTrans(config.workload_path, config.target_field_grd, config.modelMeshBaseDir, config.topo_hieghtmap_grd, config.l0, -1 if PRECISE_INIT else config.pprr)

config = parseInput()

if config.mode == 'init':
    solverInit(config)
    print("Workload init complete")
    exit(0)
elif config.mode == 'solve' or config.mode == 'continue':
    print("Starting solver")
    # gamma = np.interp(range(config.n_layers + 1), [0, config.n_layers + 1], [1, 100]).tolist()
    gamma = np.interp(range(config.n_layers + 1), [0, config.n_layers * 0.8 , config.n_layers + 1], [1, 50, 200]).tolist()

    cg.attempt(
        config.workload_path, config.target_field_grd, config.topo_hieghtmap_grd,
        config.l0, config.pprr, config.modelMeshBaseDir,
        useOldx0 = (config.mode == 'continue'), gamma = gamma
    )
    exit(0)
else:
    print(f'Unknown command "{config.mode}"')
    exit(1)
