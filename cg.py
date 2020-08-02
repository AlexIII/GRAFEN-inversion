# -*- coding: utf-8 -*-

# Copyright (c) 2019 Alexander Chernoskutov <endoftheworld@bk.ru>
# https://github.com/AlexIII
# MIT License

import Aop
import math
from shutil import copyfile
from shutil import copytree
from shutil import rmtree
from pygrid import Grid
import numpy as np


class Solver(object):
	def __init__(self, useOldx0 = False, gamma = 400):
		#diff_ref-m0.grd - target field f [provided]
		#v_rightSide - right-side vector (B^T * f), B - forward problem operator [provided]
		#inc_model - reconstructed model (current step) [empty folder provided]
		#tmpField/fieldTemp.grd - field of reconstructed model (current step). The file is generated when running Aop(). [provided (for point coorditates)]
		#A = (B^T * B + lambda*E) - main operator of this task
		#rDir, zDir - temporary computations [rDir provided (for point coorditates)]

		self.iter = 0
		self.gamma = gamma
		self.origGamma = gamma
		self.path = "X:\\elFieldCU_GKed_het_91_df_hex_2phase_fix_pool\\x64\\minRefine_TimanHD_ds\\"
		self.eps = 0.001
		
		#input data
		self.ref_field = "diff_ref_sph.grd"
		self.vRightDir = "v_rightSide"
		
		#output
		self.x = "inc_model"	#resulting model

		#set x0 = 0
		if not useOldx0:
			self.copyReplace(self.x, self.vRightDir)
			Aop.setGrids(self.path+self.x, 0)

		self.gammaWeights = (
			Aop.mapGrids(self.path+self.x, (lambda gd: np.ones(gd.data.shape)*self.origGamma ))
			if not isinstance(self.origGamma, list) else
			Aop.mapGridsIndexed(self.path+self.x, (lambda gd, i: np.ones(gd.data.shape)*self.origGamma[i] ))
		)

		#process dirs
		self.zDir = "zDir"	#z_k-1
		self.rDir = "rDir"	#r_k-1

		self.curFieldFile = "tmpField/fieldTemp.grd"
		self.xNorm = 0

	def reset(self):
		#z_0 = r_0 = v-Ax_0
		Aop.Aop(self.path, self.x, self.rDir, w = self.gammaWeights)

		self.Fx = Grid().read_grd7(self.curFieldFile).data.flatten()
		self.sumGrids(-1, self.rDir, 1, self.vRightDir) #rDir = vRightDir - rDir
		self.copyReplace(self.zDir, self.rDir)
		self.normRight = math.sqrt(self.dotGrids(self.vRightDir, self.vRightDir))
		self.recalcGammaEvery = 6

	def cont(self, it = 1):
		self.iter = it
		self.Fx = Grid().read_grd7(self.curFieldFile).data.flatten()
		self.normRight = math.sqrt(self.dotGrids(self.vRightDir, self.vRightDir))
		self.error = math.sqrt(self.dotGrids(self.rDir, self.rDir)) / self.normRight
		return self.error

	def lfun1(self, z, x):
		for i in range(0, x.shape[0]):
			if abs(x[i]) >= self.brd and x[i]*z[i] > 0:
				self.constrVlCnt += 1
				z[i] = 0.
		return z

	def lfun2(self, z, x):
		for i in range(0, z.shape[0]):
			if z[i] > 0.0:
				tmpd = self.brd - x[i]
				if tmpd >= 1e-6:
					lt = tmpd/z[i]
					if self.alpha > lt:
						self.alpha = lt
			elif z[i] < 0.0:
				tmpd = -self.brd - x[i]
				if tmpd < 1e-6:
					lt = tmpd/z[i];
					if self.alpha > lt:
						self.alpha = lt

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
		rPrvDot = self.dotGrids(self.rDir, self.rDir)
		AopTempRes = "AopRes"

		Aop.Aop(self.path, self.zDir, AopTempRes, w = self.gammaWeights)
		psi = 1
		self.alpha = psi * self.dotGrids(self.rDir, AopTempRes) / self.dotGrids(AopTempRes, AopTempRes)
		alpha = self.alpha

		#update F
		self.Fx += alpha*Grid().read_grd7(self.curFieldFile).data.flatten()	

		#clac new x
		self.sumGrids(1, self.x, alpha, self.zDir)
		self.xNorm = self.dotGrids(self.x, self.x)

		#calc field error
		b = Grid().read_grd7(self.ref_field).data.flatten()
		self.modelError = np.linalg.norm(self.Fx-b) / np.linalg.norm(b)
		
		#calc new r_k
		self.sumGrids(1, self.rDir, -alpha, AopTempRes)
		rDot = self.dotGrids(self.rDir, self.rDir)

		#calc error
		self.error = math.sqrt(rDot) / self.normRight

		#apply constraints
		self.brd = 0.15
		self.constrVlCnt = 0
		Aop.scalGridsWf(self.rDir, self.x, self.lfun1)
		print("Constraint violations: %g" % self.constrVlCnt)

		#calc betta
		betta = rDot / rPrvDot
		#betta = 0

		#calc new z_k
		self.sumGrids(betta, self.zDir, 1, self.rDir)

		self.iter += 1
		return self.error

	def copyReplace(self, dest, src):
		rmtree(self.path+dest, ignore_errors = True)
		copytree(self.path+src, self.path+dest)
	#dir1 = alpha*dir1 + betta*dir2
	def sumGrids(self, alpha, dir1, betta, dir2):
		Aop.sumGridsInDir(self.path+dir1, self.path+dir2, betta, alpha)
	def dotGrids(self, dir1, dir2):
		return Aop.scalGridsInDir(self.path+dir1, self.path+dir2)
	def copyResult(self, dest):
		self.copyReplace(dest, self.x)
	def rmDir(self, dest):
		rmtree(self.path+dest, ignore_errors = True)

def makeDistr(vFrom, vTo, size):
	return np.interp(range(size), [0, size-1], [vFrom, vTo]).tolist()

def printStat(dir):
	print("99%:")
	print(np.array(Aop.mapGrids(dir, (lambda gd: np.percentile(gd, 99)))))
	print("1%:")
	print(np.array(Aop.mapGrids(dir, (lambda gd: np.percentile(gd, 1)))))

def attempt(gamma, useOldx0 = False):
	cg = Solver(gamma = gamma, useOldx0 = useOldx0)
	minModelErr = minErr = err = 1
	flog = open('cg_log.txt','w')
	info = "initial Error: %g" % err
	print("--- "+info)
	flog.write(info+"\r\n")
	flog.flush()
	lastFname = None
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
			cg.copyResult(lastFname)

		info = "Iter: %i; Error: %g; Model Error: %g; |x| = %g" % (cg.iter, err, cg.modelError, cg.xNorm)
		print("--- "+info)
		flog.write(info+"\r\n")
		flog.flush()
	flog.close()
	printStat(cg.x)

def searchGamma():
	gamma = 30
	while gamma < 200:
		attempt(gamma)
		gamma += 10



#searchGamma()
#attempt(0)

#attempt(makeDistr(50, 800, 81))
#attempt(np.interp(range(81), [0, 75, 80], [100, 400, 1000]).tolist(), True)
attempt(np.interp(range(81), [0, 70, 80], [103, 300, 600], True).tolist()) #the best choice
#printStat("gm=100-600_e=0.00228838_me=0.0331472_it=36")

#g = [0.2619324847124174, 0.27561953400975386, 0.29005070132715327, 0.305351810840637, 0.32175956087870394, 0.33909221539974604, 0.35740615353159466, 0.37676162443291744, 0.39722306170586313, 0.4188589989046868, 0.44174344193647763, 0.4659555605603787, 0.4915801759327107, 0.5187083379127242, 0.5474665631374739, 0.5780197588643246, 0.6104152629290912, 0.6447785027366729, 0.6812449784133526, 0.7199612379810172, 0.7610859625917143, 0.8048919907839428, 0.8521711959427758, 0.9025047293348009, 0.9561210725066372, 1.013267426229828, 1.0742131902701455, 1.1392516147457603, 1.208704170653074, 1.282922423256156, 1.3622913824466416, 1.447233489590814, 1.5382129270625127, 1.6357404887020066, 1.7403790987871601, 1.852750063239467, 1.9735392254974107, 2.1034904403910675, 2.2434589375841947, 2.3943797192364547, 2.557292567836633, 2.733355547001864, 2.9238603986830367, 3.13025011331314, 3.354138985692128, 3.5973318881411385, 3.8618603058592194, 4.150003304712743, 4.464324069771366, 4.807708804695258, 5.183410849823939, 5.5951007659612975, 6.046923022738798, 6.543559918551868, 7.090303290434379, 7.693134425306537, 8.358812335001371, 9.09491886690054, 9.91006809920736, 10.825684660436353, 11.86739304421799, 13.033458892107369, 14.34087582789377, 15.809005812932709, 17.459843383025206, 19.318271096036835, 21.412283180541117, 23.77314256888586, 26.435420279441505, 29.436844430765518, 32.817857977798276, 36.62074904393152, 40.888176461548795, 45.77439103730591, 51.9121258532526, 58.875935662290566, 68.12231335129046, 85.23048072627797, 106.0969931547547, 135.00390676244638, 195.87842149939766]
#g = np.array(g)*100
#attempt(g)

#attempt(np.interp(range(81), [0, 75, 80], [20, 400, 10000]).tolist())



def sum_res():
	inc_model = "./gm=103-600_e=0.0294503_me=0.127686_it=15"
	base_model = "./TimanHD_ds"
	sum_model = "./TimanHD_ds_sph"
	rmtree(sum_model, ignore_errors = True)
	copytree(base_model, sum_model)
	Aop.sumGridsInDir(sum_model, inc_model)

#sum_res()