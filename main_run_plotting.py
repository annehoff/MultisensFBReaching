# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:54:29 2024

@author: anne hoffmannan
"""

# Import python packages: numpy, matplotlib.pyplot, seaborn
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns # plot style

# Import simulation function
from lqgfunctions import runSimulations

# Run simulations --------------------------------------------------------------
with_sdn = True  # set to True for model with signal-dependent noise
change_duration = False  # set to True to display visual feedback fro differnt durations
fully_observable = False  # set to True to simulate fully observable model
move_times = [0.7,0.4]  # movement times [slow,fast]

Data, kalmanGains, controlGains, timeVisOn = runSimulations(G=0.1,m=1,tau=0.066,
                                                            delta=0.01,nState=12,
                                                            nCtrl=2,r=10**-4,
                                                            wposh=0,wposc=1000,
                                                            wvel=0,tarycoor=0.2,
                                                            nSimu=25,load=9,
                                                            stime=0.5,
                                                            mtimes=move_times,
                                                            vsigmas=[0.1,1,10],
                                                            cshifts=[-0.02,0,0.02],
                                                            multnoise=with_sdn,
                                                            changedur=change_duration,
                                                            fullobs=fully_observable)

# Figure 1: Plot x-positions, x-errors, & x-delta force ------------------------

#sns.set_theme()
#sns.set_style("ticks")
fig = plt.figure(figsize=(12,5))
timevec = [i/1000 for i in range(-50,610,10)]

for iTime in range(0,2):

  if move_times[iTime] == 0.7:
    timeveclong = [i/1000 for i in range(0,1200,10)]
    nStep = 120
  elif move_times[iTime] == 0.4:
    timeveclong = [i/1000 for i in range(0,900,10)]
    nStep = 90

  ax1 = plt.subplot2grid((2,3),(iTime,0),rowspan=1,colspan=1)
  ax2 = plt.subplot2grid((2,3),(iTime,1),rowspan=1,colspan=1)
  ax3 = plt.subplot2grid((2,3),(iTime,2),rowspan=1,colspan=1)
  plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.2,
                    hspace=0.2)

  # positions:
  pos_sigma0 = np.nanmean(Data['posXall'][iTime,0,:,:,:],axis=1)

  # errors:
  err_sigma0 = np.nanmean(Data['errorX'][iTime,0,:,:,:],axis=1)
  err_sigma1 = np.nanmean(Data['errorX'][iTime,1,:,:,:],axis=1)
  err_sigma2 = np.nanmean(Data['errorX'][iTime,2,:,:,:],axis=1)

  # forces:
  # compute means to plot
  for_sigma0 = np.nanmean(Data['forceX'][iTime,0,:,:,:],axis=1)
  for_sigma1 = np.nanmean(Data['forceX'][iTime,1,:,:,:],axis=1)
  for_sigma2 = np.nanmean(Data['forceX'][iTime,2,:,:,:],axis=1)
  # subtract shift-no shift
  for_sigma0_left = for_sigma0[0,:]-for_sigma0[1,:]
  for_sigma0_right = for_sigma0[2,:]-for_sigma0[1,:]
  for_sigma1_left = for_sigma1[0,:]-for_sigma1[1,:]
  for_sigma1_right = for_sigma1[2,:]-for_sigma1[1,:]
  for_sigma2_left = for_sigma2[0,:]-for_sigma2[1,:]
  for_sigma2_right = for_sigma2[2,:]-for_sigma2[1,:]

  # make plots
  # 1: positions
  ax1.plot(timeveclong,pos_sigma0[0,:nStep],color=(0.75,0.75,0.75))
  ax1.plot(timeveclong,pos_sigma0[1,:nStep],color=(0.5,0.5,0.5))
  ax1.plot(timeveclong,pos_sigma0[2,:nStep],color=(0,0,0))

  ax1.spines.right.set_visible(False)
  ax1.spines.top.set_visible(False)
  if iTime == 0:
    if change_duration == True:
      ax1.set_ylabel("slow (duration: 100ms)")
    elif change_duration == False:
      ax1.set_ylabel("slow")
    ax1.set_title("Position X (sigma=low) [cm]")
    ax1.legend(['Left Shift','No Shift','Right Shift'],prop={'size': 8})
  elif iTime == 1:
    ax1.set_xlabel("Time [ms]")
    if change_duration == True:
      ax1.set_ylabel("slow (duration: 170ms)")
    elif change_duration == False:
      ax1.set_ylabel("fast")
  ax1.set_ylim([-2.5,10])

  # 2: estimation errors
  left, = ax2.plot(timevec,err_sigma0[0,:],':',color=(20/255,52/255,164/255),
                   label="Left")
  right, = ax2.plot(timevec,err_sigma0[2,:],color=(20/255,52/255,164/255),
                    label="Right")

  ax2.plot(timevec,err_sigma1[0,:],':',color=(70/255,130/255,180/255))
  ax2.plot(timevec,err_sigma1[2,:],color=(70/255,130/255,180/255))

  ax2.plot(timevec,err_sigma2[0,:],':',color=(42/255,170/255,138/255))
  ax2.plot(timevec,err_sigma2[2,:],color=(42/255,170/255,138/255))

  ax2.spines.right.set_visible(False)
  ax2.spines.top.set_visible(False)
  if iTime == 0:
    ax2.set_title("Estimation Error [cm]")
    ax2.legend([right, left], ['Right', 'Left'],prop={'size': 8})
    if change_duration == True:
      ax2.axvline(x=0.1,color='r',linestyle='dashed')
  elif iTime == 1:
    ax2.set_xlabel("Time from Cursor Jump [ms]")
    if change_duration == True:
      ax2.axvline(x=0.17,color='r',linestyle='dashed')
  ax2.axvline(x=0,color='r',linestyle='dashed')
  ax2.set_ylim([-3,3])

  # 3: delta forces
  ax3.plot(timevec,for_sigma0_right-for_sigma0_left,color=(20/255,52/255,164/255))
  ax3.plot(timevec,for_sigma1_right-for_sigma1_left,color=(70/255,130/255,180/255))
  ax3.plot(timevec,for_sigma2_right-for_sigma2_left,color=(42/255,170/255,138/255))

  ax3.spines.right.set_visible(False)
  ax3.spines.top.set_visible(False)
  if iTime == 0:
    ax3.set_title("Delta Force (Right-Left) [N]")
    ax3.legend(['low','med','high'],prop={'size': 8})
  elif iTime == 1:
    ax3.set_xlabel("Time from Cursor Jump [ms]")
  ax3.axvline(x=0,color='r',linestyle='dashed')
  ax3.set_ylim([-5.5,5.5])

# Figure 2: Plot slopes -------------------------------------------------------

#sns.set_theme()
#sns.set_style("ticks")
fig = plt.figure(figsize=(10,2))

# create subplots
ax1 = plt.subplot2grid((1,4),(0,0),rowspan=1,colspan=1)
ax2 = plt.subplot2grid((1,4),(0,1),rowspan=1,colspan=1)
ax3 = plt.subplot2grid((1,4),(0,2),rowspan=1,colspan=1)
ax4 = plt.subplot2grid((1,4),(0,3),rowspan=1,colspan=1)
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.6,
                    hspace=0.4)

nSimu=25  # number of simulations
axes = [ax1,ax2,ax3,ax4]
idx = [15,25,35,45]  # Vis. Pert. + 100, 200, 300, 400ms
tpnts = ["100ms","200ms","300ms","400ms"]
colors = [(.5,.5,.5),(0,0,0)]

for iTime in range(0,2):

  for iPnt in range(0,4):

    betas = np.empty(shape=(3,2))

    for iVsig in range(0,3):
      pos = Data['posX'][iTime,iVsig,:,:,idx[iPnt]]

      # create arrays to compute slopes
      y = np.concatenate((pos[0,:],pos[1,:],pos[2,:]),axis=0)
      x1 = np.ones(shape=(nSimu*3,1))
      x2 = np.concatenate((np.ones(shape=(nSimu,1))*-2,\
                           np.zeros(shape=(nSimu,1)),\
                           np.ones(shape=(nSimu,1))*2))
      X = np.concatenate((x1,x2),axis=1)

      # compute slopes
      betas[iVsig,:] = np.linalg.pinv(X) @ y

    axes[iPnt].axhline(y=0,color='r',linestyle='dashed',label='_nolegend_')
    axes[iPnt].axhline(y=-1,color='r',linestyle='dashed',label='_nolegend_')
    axes[iPnt].plot(range(0,3),betas[:,1],color=colors[iTime])
    axes[iPnt].set_xticks([0,1,2],labels=['low','med','high'])
    axes[iPnt].set_yticks([-1,-0.75,-0.5,-0.25,0],
                          labels=['-1','','-0.5','','0'])
    axes[iPnt].set_ylim([-1.1,0.1])
    axes[iPnt].spines.right.set_visible(False)
    axes[iPnt].spines.top.set_visible(False)
    axes[iPnt].set_xlabel("Visual Uncertainty")
    axes[iPnt].set_title("Vis Pert. + {}".format(tpnts[iPnt]))
    if iPnt == 0:
      axes[iPnt].set_ylabel("Slope [-]")
      if change_duration == True:
        axes[iPnt].legend(['slow-100ms','slow-170ms'],loc='lower left',prop={'size': 8})
      elif change_duration == False:
        axes[iPnt].legend(['slow','fast'],loc='lower left',prop={'size': 8})
        
# Figure 3: Plot Kalman & control gains ---------------------------------------

# make figure
fig = plt.figure(figsize=(20,3))
labels = ['slow','fast']
linestyles = [':','-']

# plot norm of Kalman gains influencing estimation of cursor x-pos (entry 4 in state vector)
ax0 = plt.subplot2grid((1, 3), (0, 0), rowspan = 1, colspan=1)

for d in range(2):
  KwP_low = [np.linalg.norm(kalmanGains[d,0,2,i,0:6,4]) for i in range(66)]
  KwP_med = [np.linalg.norm(kalmanGains[d,1,2,i,0:6,4]) for i in range(66)]
  KwP_high = [np.linalg.norm(kalmanGains[d,2,2,i,0:6,4]) for i in range(66)]
  ax0.plot(timevec,KwP_low,linestyles[d],color=(20/255,52/255,164/255),linewidth=1.5)
  ax0.plot(timevec,KwP_med,linestyles[d],color=(70/255,130/255,180/255),linewidth=1.5)
  ax0.plot(timevec,KwP_high,linestyles[d],color=(42/255,170/255,138/255),linewidth=1.5)

ax0.set_ylabel('Norm of Kalman gain (Cur. Pos. X)')
ax0.set_xlabel('Time from Cursor Jump [s]')
ax0.legend(['s-low','s-med','s-high','f-low','f-med','f-high'],loc='center right')

# plot norm of Kalman gains corresponding to x-dimension (entries 0-5 in state vector)
ax1 = plt.subplot2grid((1, 3), (0, 1), rowspan = 1, colspan=1)

# dimensions of Kalman gain matrix: shape=(nTimes,nVsigmas,nCshifts,nMax,nAug,nAug)
KwP_low = [np.linalg.norm(kalmanGains[i,0,2,50,0:6,0:6]) for i in range(2)]
KwP_med = [np.linalg.norm(kalmanGains[i,1,2,50,0:6,0:6]) for i in range(2)]
KwP_high = [np.linalg.norm(kalmanGains[i,2,2,50,0:6,0:6]) for i in range(2)]

ax1.set_xticks([0,1],labels)
ax1.plot([0,1],KwP_low,color=(20/255,52/255,164/255))
ax1.plot([0,1],KwP_med,color=(70/255,130/255,180/255))
ax1.plot([0,1],KwP_high,color=(42/255,170/255,138/255))
ax1.set_ylabel('Norm of Kalman gains (x-dim)')
ax1.set_xlabel('Movement Time')
ax1.legend(['low','med','high'],loc='center right')

# plot control gains corresponding to cursor x offset (entry 5 in state vector)
ax2 = plt.subplot2grid((1, 3), (0, 2), rowspan = 1, colspan=1)

for d in range(2):
  ax2.plot(timevec,controlGains[d,0,2,:,0,5],linestyles[d],color=(20/255,52/255,164/255),linewidth=1.5)
  ax2.plot(timevec,controlGains[d,1,2,:,0,5],linestyles[d],color=(70/255,130/255,180/255),linewidth=1.5,label='_nolegend_')
  ax2.plot(timevec,controlGains[d,2,2,:,0,5],linestyles[d],color=(42/255,170/255,138/255),linewidth=1.5,label='_nolegend_')

ax2.set_ylabel('Control gain (Cursor Offset X)')
ax2.set_xlabel('Time from Cursor Jump [s]')
ax2.legend(labels,loc='lower right')