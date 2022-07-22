# -*- coding: utf-8 -*-

# Copyright (c) 2019 Alexander Chernoskutov <endoftheworld@bk.ru>
# https://github.com/AlexIII
# MIT License

from typing import *
import subprocess
from shutil import copytree, rmtree
from pygrid import Grid
import sys, os
import numpy as np
import numpy.typing as npt

dStream = sys.stdout

def __log(msg, noNl = False) -> None:
	if dStream is not None:
		dStream.write(msg + ("" if noNl else "\r\n"))
		dStream.flush()

def waitForCompletion(proc: subprocess.Popen) -> None:
	while True:
		rCode = proc.poll()
		if rCode != None:
			if rCode == 0:
				break
			raise ChildProcessError("process " + str(proc.pid) + " exited with code " + str(rCode))
		while True:
			# stdout
			stdoutLine = proc.stdout.readline()		# blocking
			line = stdoutLine.strip('\r\n')
			if line: __log("P: " + line)
			if not stdoutLine: break
			# stderr
			# stderrLine = proc.stderr.readline()
			# line = stderrLine.strip('\r\n')
			# if line: __log("E: " + line)
			# if not stdoutLine and not stderrLine: break

# Run GRAFEN
def runSphSolver(path: str, params: List[str]) -> None:
	print(' '.join(["../mpirun.sh", "../../GRAFEN/src/grafen_rocm"] + params))
	proc = subprocess.Popen(["../mpirun.sh", "../../GRAFEN/src/grafen_rocm"] + params,
		cwd=path,
		stdin=subprocess.PIPE,
		stdout=subprocess.PIPE,
 		stderr=subprocess.PIPE,
 		universal_newlines=True,
		bufsize=0)
	waitForCompletion(proc)

# Run 'flat' forward gravity solver 
def runFlatSolver(path: str, params: List[str]) -> None:
	proc = subprocess.Popen([path+"gravcalcN_cuda92.exe"]+params,	
		cwd=path,
		stdin=subprocess.PIPE,
		stdout=subprocess.PIPE,
 		stderr=subprocess.PIPE,
 		universal_newlines=True,
		bufsize=0)
	waitForCompletion(proc)

def files(path: str, extention: str = None, reverseSort: bool = False) -> List[str]:
    fs = [str(os.path.join(path, f)) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and (not extention or os.path.splitext(f)[1] == extention)]
    fs.sort(reverse=reverseSort)
    return fs

#dir1 = alpha*dir1 + betta*dir2, betta may be a list
def sumGridsInDir(dir1: str, dir2: str, betta: Union[float, List[float]] = 1, alpha = 1) -> None:
	f1 = files(dir1, '.grd')
	f2 = files(dir2, '.grd')
	if not isinstance(betta, list):
		betta = np.ones(len(f1))*betta
	assert len(f1) == len(f2) and len(f1) == len(betta)
	for i in range(len(f1)):
		g = Grid().read_grd7(f1[i])
		g.data = alpha*g.data + betta[i]*Grid().read_grd7(f2[i]).data
		g.write_grd7(f1[i])

def mulGrids(dir: str, gamma: float) -> None:
	for f in files(dir, '.grd'):
		g = Grid().read_grd7(f)
		g.data *= gamma
		g.write_grd7(f)

def setGrids(dir: str, v: float) -> None:
	for f in files(dir, '.grd'):
		g = Grid().read_grd7(f)
		g.data.fill(v)
		g.write_grd7(f)

def mapGridsIndexed(dir: str, fun: Callable[[npt.NDArray, int], npt.NDArray]) -> List[npt.NDArray]:
	res = []
	i = 0
	for f in files(dir, '.grd'):
		g = Grid().read_grd7(f)
		res.append(fun(g.data, i))
		i += 1
	return res

def mapGrids(dir: str, fun: Callable[[npt.NDArray], npt.NDArray]) -> List[npt.NDArray]:
	return mapGridsIndexed(dir, lambda g, _: fun(g))

def transformGridsInDir(dir1: str, dir2: str, fun: Callable[[npt.NDArray, npt.NDArray], npt.NDArray], noWrite = False) -> None:
	f1 = files(dir1, '.grd')
	f2 = files(dir2, '.grd')
	assert len(f1) == len(f2)
	for i in range(len(f1)):
		g1 = Grid().read_grd7(f1[i])
		g2 = Grid().read_grd7(f2[i])
		g1.data = fun(g1.data, g2.data)
		if not noWrite:
			g1.write_grd7(f1[i])

def dotGridsInDir(dir1: str, dir2: str) -> float:
	f1 = files(dir1, '.grd')
	f2 = files(dir2, '.grd')
	assert len(f1) == len(f2)
	sum = 0
	for i in range(len(f1)):
		g1 = Grid().read_grd7(f1[i])
		g2 = Grid().read_grd7(f2[i])
		sum += np.dot(g1.data.flatten(), g2.data.flatten())
	return sum

def SolveFwd(path: str, fieldFname: str, modelDir: str, topoGrd: Optional[str] = None, l0: float = 60, pprr: float = 100):	# pprr - Point Potential Replace Radius
	layers = len(files(os.path.join(path, modelDir), '.grd')) - (1 if topoGrd is not None else 0)
	runSphSolver(path, 
		["-grd7", fieldFname, "-Hf", "0.00001", "-Hfrom", str(-layers), "-Hto", "0", "-Hn", str(layers), "-l0", str(l0), "-dens", modelDir, "-DPR", str(pprr)] + 
		(["-fieldOnTopo", "-topoHeightGrd7", topoGrd] if topoGrd is not None else [])
	)

def SolveTrans(path: str, fieldFname: str, modelDir: str, topoGrd: Optional[str] = None, l0: float = 60, pprr: float = 100):	# pprr - Point Potential Replace Radius
	layers = len(files(os.path.join(path, modelDir), '.grd')) - (1 if topoGrd is not None else 0)
	runSphSolver(path,
		["-grd7", fieldFname, "-Hf", "0.00001", "-Hfrom", str(-layers), "-Hto", "0", "-Hn", str(layers), "-l0", str(l0), "-dens", modelDir, "-DPR", str(pprr), "-transposeSolver"] + 
		(["-fieldOnTopo", "-topoHeightGrd7", topoGrd] if topoGrd is not None else [])
	)

def Aop(path: str, outFwdFieldGrd: str, modelDir: str, topoGrd: str, resultDir: str, layerWeights: List[float], l0: float, pprr: float):
	rmtree(os.path.join(path, resultDir), ignore_errors = True)
	copytree(os.path.join(path, modelDir), os.path.join(path, resultDir))

	#fwd
	SolveFwd(path, outFwdFieldGrd, modelDir, topoGrd, l0, pprr)
	#trans
	SolveTrans(path, outFwdFieldGrd, resultDir, topoGrd, l0, pprr)
	# A(A^T(x)) + wb
	sumGridsInDir(os.path.join(path, resultDir), os.path.join(path, modelDir), layerWeights)
	