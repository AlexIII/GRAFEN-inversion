# -*- coding: utf-8 -*-

# Copyright (c) 2019 Alexander Chernoskutov <endoftheworld@bk.ru>
# https://github.com/AlexIII
# MIT License

import subprocess
from shutil import copyfile
from shutil import copytree
from shutil import rmtree
from pygrid import Grid
import sys
import os
import numpy as np

dStream = sys.stdout

def __log(msg, noNl = False):
	if dStream is not None:
		dStream.write("prog: " + msg + ("" if noNl else "\r\n"))
		dStream.flush()

def waitForCompletion(proc):
	while True:
		rCode = proc.poll()
		if rCode != None:
			if rCode == 0:
				break
			raise ChildProcessError("process " + str(proc.pid) + " exited with code " + str(rCode))
		line = proc.stdout.readline().strip('\r\n')
		if line:
			__log("proc: "+line)

# Run GRAFEN
def runSphSolver(path, params):
	proc = subprocess.Popen([path+"mpirun.cmd", "elFieldCU_GKed_het_91_df.exe"]+params,
		cwd=path,
		stdin=subprocess.PIPE,
		stdout=subprocess.PIPE,
 		stderr=subprocess.PIPE,
 		universal_newlines=True,
		bufsize=0)
	waitForCompletion(proc)

# Run 'flat' forward gravity solver 
def runFlatSolver(path, params):
	proc = subprocess.Popen([path+"gravcalcN_cuda92.exe"]+params,	
		cwd=path,
		stdin=subprocess.PIPE,
		stdout=subprocess.PIPE,
 		stderr=subprocess.PIPE,
 		universal_newlines=True,
		bufsize=0)
	waitForCompletion(proc)

def files(mypath, ext = None, revSort = False):
    fs = [str(os.path.join(mypath, f)) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f)) and (not ext or os.path.splitext(f)[1] == ext)]
    fs.sort(reverse=revSort)
    return fs

#dir1 = alpha*dir1 + betta*dir2, betta may be an array
def sumGridsInDir(dir1, dir2, betta = 1, alpha = 1):
	f1 = files(dir1, '.grd')
	f2 = files(dir2, '.grd')
	if not isinstance(betta, list):
		betta = np.ones(len(f1))*betta
	assert len(f1) == len(f2) and len(f1) == len(betta)
	for i in range(len(f1)):
		g = Grid().read_grd7(f1[i])
		g.data = alpha*g.data + betta[i]*Grid().read_grd7(f2[i]).data
		g.write_grd7(f1[i])

def sumGridsInDirWeighted(dir1, dir2, w):
	f1 = files(dir1, '.grd')
	f2 = files(dir2, '.grd')
	assert len(f1) == len(f2) and len(f1) == len(w)
	for i in range(len(f1)):
		g = Grid().read_grd7(f1[i])
		g.data = g.data + w[i]*Grid().read_grd7(f2[i]).data
		g.write_grd7(f1[i])

def mulGrids(dir, gamma):
	for f in files(dir, '.grd'):
		g = Grid().read_grd7(f)
		g.data *= gamma
		g.write_grd7(f)

def setGrids(dir, v):
	for f in files(dir, '.grd'):
		g = Grid().read_grd7(f)
		g.data.fill(v)
		g.write_grd7(f)

def mapGridsIndexed(dir, fun):
	res = []
	i = 0
	for f in files(dir, '.grd'):
		g = Grid().read_grd7(f)
		res.append(fun(g.data, i))
		i += 1
	return res

def mapGrids(dir, fun):
	mapGridsIndexed(dir, lambda g, i: fun(g))

def scalGridsWf(dir1, dir2, fun, noWrite = False):
	f1 = files(dir1, '.grd')
	f2 = files(dir2, '.grd')
	assert len(f1) == len(f2)
	for i in range(len(f1)):
		g1 = Grid().read_grd7(f1[i])
		g2 = Grid().read_grd7(f2[i])
		g1.data = fun(g1.data, g2.data)
		if not noWrite:
			g1.write_grd7(f1[i])

def scalGridsInDir(dir1, dir2):
	f1 = files(dir1, '.grd')
	f2 = files(dir2, '.grd')
	assert len(f1) == len(f2)
	sum = 0
	for i in range(len(f1)):
		g1 = Grid().read_grd7(f1[i])
		g2 = Grid().read_grd7(f2[i])
		sum += np.dot(g1.data.flatten(), g2.data.flatten())
	return sum

def Aop(path, modelDir, resultDir, gamma = None, w = None):
	tempFieldDir = "tmpField/"
	tempFieldFname = tempFieldDir+"fieldTemp.grd"
	layers = len(files(modelDir, '.grd'))
	params = (lambda isTrans:
		["-grd7", tempFieldFname, "-Hf", "0.00001", "-Hfrom", str(-layers), "-Hto", "0", "-Hn", str(layers), "-l0", "57", "-dens", resultDir, "-DPR", "180", "-transposeSolver"]
		if isTrans else 
		["-grd7", tempFieldFname, "-Hf", "0.00001", "-Hfrom", str(-layers), "-Hto", "0", "-Hn", str(layers), "-l0", "57", "-dens", modelDir, "-DPR", "180"]
		)

	rmtree(path+resultDir, ignore_errors = True)
	copytree(path+modelDir, path+resultDir)
	try:
		os.mkdir(path+resultDir)
	except OSError:
		None

	#fwd
	runSphSolver(path, params(False))
	#trans
	runSphSolver(path, params(True))
	#FFtx + gm*x
	if gamma != None:
		sumGridsInDir(path+resultDir, path+modelDir, gamma)
	else:
		sumGridsInDirWeighted(path+resultDir, path+modelDir, w)

def AopFlat(path, modelDir, resultDir, gamma = None, w = None):
	topLayerZ = 0
	tempFieldDir = "tmpField/"
	tempFieldFname = tempFieldDir+"fieldTemp.grd"
	layers = len(files(modelDir, '.grd'))
	params = (lambda isTrans, gpu = 1:
		[tempFieldDir, "0", str(-layers+1+topLayerZ), "1", "*", "*", str(gpu), resultDir+"\\f", "invField", str(layers)]
		if isTrans else 
		[modelDir, "0", str(-layers+1+topLayerZ), "1", tempFieldFname, "*", str(gpu), "*", "invField"])

	#prepare dirs
	try:
		os.mkdir(tempFieldDir)
	except OSError:
		None		
	rmtree(path+resultDir, ignore_errors = True)
	try:
		os.mkdir(path+resultDir)
	except OSError:
		None

	#fwd
	runFlatSolver(path, params(False))
	#trans
	runFlatSolver(path, params(True))
	#FFtx + gm*x
	if gamma != None:
		sumGridsInDir(path+resultDir, path+modelDir, gamma)
	else:
		sumGridsInDirWeighted(path+resultDir, path+modelDir, w)

def test():
	path = "X:\\elFieldCU_GKed_het_91_df_hex_2phase_fix_pool\\x64\\minRefine_TimanHD_ds_flat\\"
	x = "inc_model"
	#Aop test
	AresDir = "Aop"
	Aop(path, x, AresDir, 100)

#test()
#sumGridsInDir("model0ds", "gm=100-600_e=0.00228838_me=0.0331472_it=36")