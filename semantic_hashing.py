#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Semantic Hashing

import sys
import numpy as np
import numpy.matlib
from scipy.special import expit
import math
from sys import stdout

# Generar batchs, ultimo batch reune los casos sobrantes.
def makebatches(tfs,size_batch=100):
	np.random.shuffle(tfs)
	num_batches=len(tfs)/size_batch
	last=len(tfs)%size_batch
	tf_batchs = np.split(tfs[:-last],num_batches)
	tf_batchs[-1]=np.vstack((tf_batchs[-1],tfs[-last:]))
	return tf_batchs

def sigmoid(x):
	z = np.exp(-x)
	if np.any(np.isnan(z)) or np.any(np.isinf(z)):
		z = np.nan_to_num(z)
	return np.divide(1.,1+z)

# Pre-training
# Puede usarse con RBM, CPM y RBMLineal
def pre_training(batchs,num_h=128,lr_w=0.01,lr_vb=0.01,lr_hb=0.01,momentum=0.9,weightcost=0.0002,maxepoch=50,type_pre_training='RBM'):
	num_batches = np.shape(batchs)
	num_v = batchs[0].shape[1]
	vishid     = 0.1*np.random.randn(num_v, num_h) #Generar Matrix Pesos con siguitne distribución N(0,0.01)
	hidbiases  = np.zeros((1,num_h))
	visbiases  = np.zeros((1,num_v))

	batchposhidprobs = []

        for idx,data in enumerate(batchs):
                batchposhidprobs.append(np.zeros(data.shape))
	errsum1 = np.zeros(maxepoch)

	for epoch in xrange(maxepoch):
		for idx, data in enumerate(batchs):
			num_cases = data.shape[0]
			poshidprobs = np.zeros((num_cases,num_h))
			neghidprobs = np.zeros((num_cases,num_h))
			posprods    = np.zeros((num_v,num_h))
			negprods    = np.zeros((num_v,num_h))
			vishidinc    = np.zeros((num_v,num_h))
			hidbiasinc = np.zeros((1,num_h))
			visbiasinc = np.zeros((1,num_v))

			# START POSITIVE PHASE
			stdout.write("\r epoch %s, Batch %s              " %(str(epoch+1),str(idx+1)))
			if type_pre_training == 'RBM' or type_pre_training == 'CPM':
				poshidprobs = expit(np.dot(data,vishid) + np.tile(hidbiases,(num_cases,1)))
			elif type_pre_training == 'RBM_hidlinear':
				poshidprobs =  np.dot(data,vishid) + np.tile(hidbiases,(num_cases,1))
			batchposhidprobs[idx]=poshidprobs
			posprods    = np.dot(np.transpose(data),poshidprobs)
			poshidact   = np.sum(poshidprobs, axis=0)
  			posvisact = np.sum(data, axis=0)
			# END POSITIVE PHASE

			if type_pre_training == 'RBM' or type_pre_training == 'CPM':
				poshidstates = poshidprobs > np.random.rand(num_cases,num_h)
			elif type_pre_training == 'RBM_hidlinear':
				poshidstates = poshidprobs + np.random.rand(num_cases,num_h)

			# START NEGATIVE PHASE
			if type_pre_training == 'RBM' or type_pre_training == 'RBM_hidlinear':
				negdata = expit(np.dot(poshidstates,np.transpose(vishid)) + np.tile(visbiases,(num_cases,1)))
			elif type_pre_training == 'CPM':
				N = np.sum(data,axis=1)
				size_doc = np.tile(N,(num_v,1)).T
				numerator=np.dot(poshidstates,np.transpose(vishid)) + np.tile(visbiases,(num_cases,1))
				lambda_=np.multiply(size_doc,softmax(numerator))
				negdata = np.random.poisson(lambda_)

			if type_pre_training == 'RBM' or type_pre_training == 'CPM':
				neghidprobs = expit(np.dot(negdata,vishid) + np.tile(hidbiases,(num_cases,1)))
			elif type_pre_training == 'RBM_hidlinear':
				neghidprobs = np.dot(negdata,vishid) + np.tile(hidbiases,(num_cases,1))

			negprods  = np.dot(np.transpose(negdata),neghidprobs)
  			neghidact = np.sum(neghidprobs,axis=0)
  			negvisact = np.sum(negdata,axis=0)
			# END NEGATIVE PHASE

			# ERRORES
			err1 = (np.sum(np.square(data-negdata)))/num_cases
			errsum1[epoch] = err1 + errsum1[epoch]

			# START UPDATE W Y BIAS
			vishidinc = momentum*vishidinc + lr_w*( (posprods-negprods)/num_cases - weightcost*vishid)
			visbiasinc = momentum*visbiasinc + (lr_vb/num_cases)*(posvisact-negvisact)
			hidbiasinc = momentum*hidbiasinc + (lr_hb/num_cases)*(poshidact-neghidact)

			vishid = vishid + vishidinc
			visbiases = visbiases + visbiasinc
			hidbiases = hidbiases + hidbiasinc
			# END UPDATE W Y BIAS
			stdout.flush()
		errsum1[epoch] = errsum1[epoch]/(idx+1)
	print errsum1
	return batchposhidprobs, vishid, visbiases, hidbiases

# Une las matrices de pesos en solo un vector.
# Necesario para la librería scipy.optimize
def pack_W(W):
	return np.concatenate(tuple([w.reshape(-1) for w in W]))

# Separa el vector de pesos en las matrices correspondientes.
def unpack_W(W_concat,layers):
	W=[]
	pos=0
	for l in xrange(1,len(layers)):
		W.append(W_concat[pos:pos+((layers[l-1]+1)*layers[l])].reshape(layers[l-1]+1,layers[l]))
		pos = pos + 	(layers[l-1]+1)*layers[l]
	return W

# Realiza softmax para cualquier dimensión evitando overflow.
def softmax(x):
	if x.ndim == 1:
		x = x.reshape([1,x.size])
	softmaxInvx = x - np.max(x,1).reshape([x.shape[0],1])
	maxExp = np.exp(softmaxInvx)
	return maxExp/np.sum(maxExp,axis=1).reshape([maxExp.shape[0],1])

# Mide error en finetuning y ajuste a los pesos.
def CG(W_concat,layers,data):
	batches_prob = []
	dW = []
	num_cases=np.shape(data)[0]
	W = unpack_W(W_concat,layers)
	batch = data.copy()
	for idx,w in enumerate(W[:-1]):
		batch = np.hstack((batch,np.ones((num_cases,1))))
		batches_prob.insert(0,batch)
		batch = expit(np.dot(batch,w))
	batch = np.hstack((batch,np.ones((num_cases,1))))
	batches_prob.insert(0,batch)
	batch = softmax(np.dot(batch,W[-1]))
	cost_Pos = np.nan_to_num((- np.sum(np.multiply(data,np.log(batch)))))
	cost = cost_Pos/num_cases

	I = (batch-data)/num_cases
	for idx,b_p in enumerate(batches_prob,1):
		dw =  np.dot(b_p.conj().T,I)
		dW.insert(0,dw)
		I = np.multiply(np.dot(I,W[len(W)-idx].conj().T),np.multiply(b_p,1-b_p))[:,:-1]
	return cost, pack_W(dW)

# Equivalente a minimize.m
def minim(W_concat,layers,data,n_linesearch=3):
	INT = 0.1
	EXT = 3.0
	MAX = 20
	RATIO = 10
	SIG = 0.1
	RHO = SIG/2
	red = 1
	i = 0
	ls_failed = 0
	X = W_concat
	f0, df0 = CG(X,layers,data)
	fX = f0
	s = -df0
	d0 = -np.dot(s,s.conj().T)
	x3 = red/(1-d0)

	for i in xrange(n_linesearch):
		X0 = X
		F0 = f0
		dF0 = df0
		M = MAX
		while 1:
			x2 = 0
			f2 = f0
			d2 = d0
			f3 = f0
			df3 = df0
			success = 0
			while not success and M > 0:
				M = M - 1
				f3, df3 = CG((X+x3*s),layers,data)
				bool_f3_nan = np.isnan(f3)
				bool_f3_inf = np.isinf(f3)
				bool_df3_nan = np.any(np.isnan(df3))
				bool_df3_inf = np.any(np.isinf(df3))
				if bool_f3_nan or bool_f3_inf or bool_df3_nan or bool_df3_inf:
					x3 = (x2+x3)/2
				else:
					success = 1
			if f3 < F0:
				X0 = X+x3*s
				F0 = f3
				dF0 = df3
			d3 = np.dot(s,df3.conj().T)

			if d3 > SIG*d0 or f3 > f0+x3*RHO*d0 or M == 0:
				break
			x1 = x2
			f1 = f2
			d1 = d2
			x2 = x3
			f2 = f3
			d2 = d3
			A = 6*(f1-f2)+3*(d2+d1)*(x2-x1)
			B = 3*(f2-f1)-(2*d1+d2)*(x2-x1)
			x3 = x1-d1*math.pow((x2-x1),2)/(B+np.nan_to_num(np.sqrt(B*B-A*d1*(x2-x1))))
			if not np.isreal(x3) or np.isnan(x3) or np.isinf(x3) or x3 < 0:
				x3 = x2*EXT
			elif x3 > x2*EXT:
				x3 = x2*EXT
			elif x3 < x2+INT*(x2-x1):
				x3 = x2+INT*(x2-x1)

		while (abs(d3) > -SIG*d0 or f3 > f0 + x3*RHO*d0) and M>0:
			if d3 > 0 or f3 > f0+x3*RHO*d0:
				x4 = x3
				f4 = f3
				d4 = d3
			else:
				x2 = x3
				f2 = f3
				d2 = d3

			if f4 > f0:
				x3 = x2-(0.5*d2*math.pow((x4-x2),2))/(f4-f2-d2*(x4-x2))
			else:
				A = 6*(f2-f4)/(x4-x2)+3*(d4+d2)
				B = 3*(f4-f2)-(2*d2+d4)*(x4-x2)
				x3 = x2+(np.sqrt(B*B-A*d2*math.pow((x4-x2),2))-B)/A

			if np.isnan(x3) or np.isinf(x3):
				x3 = (x2+x4)/2

			x3 = max(min(x3, x4-INT*(x4-x2)),x2+INT*(x4-x2))
			f3, df3 = CG((X+x3*s),layers,data)
			if f3 < F0:
				X0 = X+x3*s
				F0 = f3
				dF0 = df3
			M = M - 1
			d3 = np.dot(s,df3.conj().T)

		if abs(d3) < -SIG*d0 and f3 < f0+x3*RHO*d0:
			X=X+x3*s
			f0 = f3
			fX = np.hstack((fX.conj().T,f0)).conj().T
			print'\r Linesearch %d; Value %f' %(i,f0)
			s = (np.dot(df3,df3.conj().T)-np.dot(df3,df0.conj().T))/np.dot(df0,df0.conj().T)*s - df3
			df0 = df3
			d3 = d0
			d0 = np.dot(s,df0.conj().T)
			if d0 > 0:
				s = -df0
				d0 = -np.dot(s,s.conj().T)
			x3 = x3*min(RATIO,d3/(d0-np.finfo(np.double).tiny))
			ls_failed = 0
		else:
			X = X0
			f0 = F0
			df0 = dF0

			if ls_failed or i > n_linesearch:
				break

			s = -df0
			d0 = -np.dot(s,s.conj().T)
			x3 = 1/(1-d0)
			ls_failed = 1
	return X

def finetuning(batchs,W,layers,maxepoch=50):
	W_concat = pack_W(W)
	for epoch in xrange(maxepoch):
		for idx,data in zip(xrange(len(batchs)),batchs):
			print "epoch %d, batch %d" %(epoch+1,idx+1)
			W_concat = minim(W_concat,layers,data)
			stdout.flush()
	return unpack_W(W_concat,layers)

def fit(tfs, hidden_layer=[500,500], output_layer=128, maxepoch=50,
		lr_w = 0.1, lr_vb = 0.1, lr_hb = 0.1, weightcost = 0.0002, momentum = 0.9,
		pretrain_size_batch=100, finetuning_size_batch=1000):
	tf_batchs = makebatches(tfs,pretrain_size_batch)
	num_batches=len(tf_batchs)
	print "Número Batchs: %d\nTamaño Batch:  %d" % (num_batches, pretrain_size_batch)
	input_layer = np.size(tf_batchs[0][0])
	layers = [input_layer] + hidden_layer + [output_layer]
	print "Pre-training"
	W_rec = []
	W_gen = []
	print "Pre-Training Pesos para entrada con CPM"
	post_batchs, vishid, visbiases,hidrecbiases = pre_training(tf_batchs,hidden_layer[0],lr_w,lr_vb,lr_hb,momentum,weightcost,maxepoch,'CPM')
	W_rec.append(np.append(vishid,hidrecbiases,axis=0))
	W_gen.insert(0,np.append(np.transpose(vishid),visbiases,axis=0))

	for i in xrange(1,len(hidden_layer)):
		print "Pre-Training Pesos para capas oculta %d" %i
		post_batchs, vishid, visbiases, hidrecbiases = pre_training(post_batchs,hidden_layer[i],lr_w,lr_vb,lr_hb,momentum,weightcost,maxepoch,'RBM')
		W_rec.append(np.append(vishid,hidrecbiases,axis=0))
		W_gen.insert(0,np.append(np.transpose(vishid),visbiases,axis=0))

	print "Pre-Training Pesos capa salida"
	post_batchs, vishid, visbiases, hidrecbiases = pre_training(post_batchs,output_layer,lr_w,lr_vb,lr_hb,momentum,weightcost,maxepoch,'RBM')

	W_rec.append(np.append(vishid,hidrecbiases,axis=0))
	W_gen.insert(0,np.append(np.transpose(vishid),visbiases,axis=0))

	W = W_rec + W_gen
	hidden_layer.reverse()
	layers_autoencoder = layers + hidden_layer + [input_layer]
	print "Fine-Tuning"

	tf_batchs = makebatches(tfs,finetuning_size_batch)
	
	num_batches=len(tf_batchs)
	print "Número Batchs: %d\nTamaño Batch:  %d" % (num_batches, finetuning_size_batch)
	for i in xrange(len(tf_batchs)):
		size_doc = np.matlib.repmat(np.sum(tf_batchs[i],axis=1,dtype=float),tf_batchs[i].shape[1],1).T
		tf_batchs[i]=np.divide(tf_batchs[i],size_doc)

	W_f = finetuning(tf_batchs,W,layers_autoencoder,maxepoch)
	return W_f[:len(W_f)/2]

def transform(tfs,W):
	print 'Generar Hash Data'
	hashs = np.zeros((tfs.shape[0],W[-1].shape[1]),dtype=np.bool_)
	N = tfs.shape[0]
	for tf,i in zip(tfs,range(1,N+1)):
		stdout.write("\r%d/%d" % (i,N))
		for w in W:
			tf = np.hstack((tf,np.ones((1))))
			tf = expit(np.dot(tf,w))
		tf = (tf > 0.1).astype(bool)
		hashs[i-1] = tf
		stdout.flush()
	print '\nHash Data generado'
	return hashs
