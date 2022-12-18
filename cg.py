#!/bin/python3

# -*- coding: utf-8 -*-

# Copyright (c) 2019 Alexander Chernoskutov <endoftheworld@bk.ru>
# https://github.com/AlexIII
# MIT License

import Aop
import math, os, datetime, time
from shutil import copyfile, copytree, rmtree
from pygrid import Grid
import numpy as np
import numpy.typing as npt
from typing import *


class Solver(object):
	gammaWeights: List[npt.NDArray]

	def __init__(self, path: str, ref_field_grd: str, topoGrd: str, l0: float, pprr: float, vRightDir: str, useOldx0 = False, gamma: Union[float, List[float]] = 400):
		#self.ref_field - target field f [to provide]
		#self.vRightDir - right-side vector = B^T(f), B - forward problem operator [to provide]
		#inc_model - reconstructed model (current step) [to provide if useOldx0 == True]
		#tmpField/fieldTemp.grd - field of reconstructed model (current step). The file is generated when running Aop().
		#A = (B^T * B + lambda*E) - main operator of this task
		#rDir, zDir - temporary computations

		self.iter = 0
		self.gamma = gamma
		# self.origGamma = gamma
		self.path = path
		self.l0 = l0
		self.pprr = pprr
		self.topoGrd = topoGrd
		self.eps = 0.001
		
		#input data
		self.ref_field = ref_field_grd
		self.vRightDir = vRightDir
		
		#output
		self.x = "inc_model"	#resulting model

		#set x0 = 0
		if not useOldx0:
			self.copyReplaceDir(self.x, self.vRightDir)
			Aop.setGrids(self.pathOf(self.x), 0)

		self.gammaWeights = (
			Aop.mapGrids(self.pathOf(self.x), (lambda gd: np.ones(gd.data.shape)*self.gamma ))
			if not isinstance(self.gamma, list) else
			Aop.mapGridsIndexed(self.pathOf(self.x), (lambda gd, i: np.ones(gd.data.shape)*self.gamma[i] ))
		)

		#top layer spec. gamma weights (based on the height map)
		# g_spec_top_layer_map = lambda x: (1.4 - (0 if x >= Grid.BlankValue else x))**4*20000 + 50
		# self.gammaWeights[len(self.gammaWeights) - 1] = np.array([g_spec_top_layer_map(v) for v in Grid().read_grd7(self.pathOf(self.topoGrd)).data])

		#process dirs
		self.zDir = "zDir"	#z_k-1
		self.rDir = "rDir"	#r_k-1

		self.curFieldFile = "field_temp.grd"
		self.xNorm = 0

		copyfile(self.pathOf(self.ref_field), self.pathOf(self.curFieldFile))
		self.copyReplaceDir(self.rDir, self.vRightDir)

	def reset(self):
		#z_0 = r_0 = v-Ax_0
		Aop.Aop(self.path, self.curFieldFile, self.x, self.topoGrd, self.rDir, self.gammaWeights, self.l0, self.pprr)

		self.Fx = self.readFlatGridData(self.curFieldFile)
		self.sumGrids(-1, self.rDir, 1, self.vRightDir) #rDir = vRightDir - rDir
		self.copyReplaceDir(self.zDir, self.rDir)
		self.normRight = math.sqrt(self.dotGridsDir(self.vRightDir, self.vRightDir))
		self.recalcGammaEvery = 6

	def lfun1(self, z: npt.NDArray, x: npt.NDArray):
		for i in range(0, x.shape[0]):
			if abs(x[i]) >= self.brd and x[i]*z[i] > 0:
				self.constrVlCnt += 1
				z[i] = 0.
		return z

	def iterate(self):
		if self.iter == 0:
			self.reset()
		#elif self.iter%self.recalcGammaEvery == 0:
		#	self.recalcGammaEvery += 1
		#	self.gammaWeights = Aop.mapGrids(self.x, (lambda gd: np.square(self.origGamma*gd)))
		#	print("new gamma:")
		#	print(np.array([np.mean(i) for i in self.gammaWeights]))
		#	self.reset()

		#calc alpha
		rPrvDot = self.dotGridsDir(self.rDir, self.rDir)
		AopTempRes = "AopRes"

		Aop.Aop(self.path, self.curFieldFile, self.zDir, self.topoGrd, AopTempRes, self.gammaWeights, self.l0, self.pprr)
		psi = 1
		self.alpha = psi * self.dotGridsDir(self.rDir, AopTempRes) / self.dotGridsDir(AopTempRes, AopTempRes)
		alpha = self.alpha

		#update F
		self.Fx += alpha * self.readFlatGridData(self.curFieldFile)

		#clac new x
		self.sumGrids(1, self.x, alpha, self.zDir)
		self.xNorm = self.dotGridsDir(self.x, self.x)

		#calc field error
		b = self.readFlatGridData(self.ref_field)
		self.modelError = np.linalg.norm(self.Fx-b) / np.linalg.norm(b)
		
		#calc new r_k
		self.sumGrids(1, self.rDir, -alpha, AopTempRes)
		rDot = self.dotGridsDir(self.rDir, self.rDir)

		#calc error
		self.error = math.sqrt(rDot) / self.normRight

		#apply constraints
		self.brd = 0.5
		self.constrVlCnt = 0
		Aop.transformGridsInDir(self.pathOf(self.rDir), self.pathOf(self.x), self.lfun1)
		print("Constraint violations: %g" % self.constrVlCnt)

		#calc betta
		betta = rDot / rPrvDot

		#calc new z_k
		self.sumGrids(betta, self.zDir, 1, self.rDir)

		self.iter += 1
		return self.error

	def copyReplaceDir(self, dest, src):
		rmtree(self.pathOf(dest), ignore_errors = True)
		copytree(self.pathOf(src), self.pathOf(dest))
	#dir1 = alpha*dir1 + betta*dir2
	def sumGrids(self, alpha, dir1, betta, dir2):
		Aop.sumGridsInDir(self.pathOf(dir1), self.pathOf(dir2), betta, alpha)
	def dotGridsDir(self, dir1, dir2):
		return Aop.dotGridsInDir(self.pathOf(dir1), self.pathOf(dir2))
	def makeResultCopy(self, dest):
		self.copyReplaceDir(dest, self.x)
	def rmDir(self, dest):
		rmtree(self.pathOf(dest), ignore_errors = True)
	def pathOf(self, fname):
		return os.path.join(self.path, fname)
	def readFlatGridData(self, fname_grd: str):
		return Grid().read_grd7(self.pathOf(fname_grd)).data.flatten()

def makeDistr(vFrom, vTo, size):
	return np.interp(range(size), [0, size-1], [vFrom, vTo]).tolist()

def printStat(dir):
	print("99%:")
	print(np.array(Aop.mapGrids(dir, (lambda gd: np.percentile(gd, 99)))))
	print("1%:")
	print(np.array(Aop.mapGrids(dir, (lambda gd: np.percentile(gd, 1)))))

def attempt(path: str, ref_field_grd: str, topoGrd: str, l0: float, pprr: float, vRightDir = "v_rightSide", useOldx0 = False, gamma = 400):
	cg = Solver(path, ref_field_grd, topoGrd, l0, pprr, vRightDir, useOldx0, gamma)
	minModelErr = minErr = err = 1
	flog = open(os.path.join(path, 'cg_log.txt'), ('a' if useOldx0 else 'w'))
	if useOldx0: flog.write("- continue -\r\n")
	info = "CG attempt start at %s" % datetime.datetime.now()
	print("--- "+info)
	flog.write(info+"\r\n")
	flog.flush()
	lastFname = None
	gammaDecreased = False

	tsStart = time.time()
	while cg.iter < 150 and (err < 100 or cg.iter < 10):
		err = cg.iterate()

		toSave = False
		if minErr > err:
			minErr = err
			toSave = True
		if minModelErr > cg.modelError:
			minModelErr = cg.modelError
			toSave = True

		if toSave:
			if lastFname:
				cg.rmDir(lastFname)
			lastFname = (
				"gm=%g_e=%g_me=%g_it=%i" % (gamma, err, cg.modelError, cg.iter)
				if not isinstance(gamma, list) else
				"gm=%g-%g_e=%g_me=%g_it=%i" % (gamma[0], gamma[len(gamma)-1], err, cg.modelError, cg.iter))
			cg.makeResultCopy(lastFname)

		info = "Iter: %i (%i sec); Error: %g; Model Error: %g; |x| = %g" % (cg.iter, int(time.time() - tsStart), err, cg.modelError, cg.xNorm)
		print("--- "+info)
		flog.write(info+"\r\n")
		
		# if err < 0.25 and not gammaDecreased:
		# 	for g in cg.gammaWeights: g *= 0.5
		# 	gammaDecreased = True
		# 	flog.write("Gamma decreased\r\n")
			
		flog.flush()

	flog.close()
	printStat(cg.x)

def searchGamma():
	gamma = 30
	while gamma < 200:
		attempt(gamma)
		gamma += 10



def sum_res():
	inc_model = "./postprocess-workload/gm=1-98.7778_e=0.000991535_me=0.0405617_it=43"
	base_model = "./postprocess-workload/model0_rffi2022"
	sum_model = "./postprocess-workload/model0_result_sum_rffi2022"
	rmtree(sum_model, ignore_errors = True)
	copytree(base_model, sum_model)
	Aop.sumGridsInDir(sum_model, inc_model)

# sum_res()