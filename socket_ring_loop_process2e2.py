#!/usr/bin/env python
#
# example1.py: simplest possible example.
#
#   - `config_params` constructed "by hand" in python (rather than reading from file)
#
#   - dedisperser run on simulated noise-only data
#
#   - coarse-grained triggers postprocessed "by hand", and global maximum trigger value reported
#     (just as a toy example to illustrate the API for postprocessing the triggers)
#  e.g. python2 socket_ring_loop_process2d.py 172.17.31.236 8000 16
#  using multiprocessing
#  read 20sec data from one all data
#  ring buffer/fp16
#  take too long copy ring buffer to sav_intensity
#   sav_intensity[:,:,it]=ring[it*nbeam+id-1]
#  fork for sav_intensity=ring and np.save
#
#  saving the trigger ring buffer to npy file
#      data=np.load('20230427-231237_beam2.npy')
#      data.shape=[2*Nx,Ny,t_total]
#  saving socket buffer to a binary file by schedule file
#      f=open('file.bin','rb')
#      dd=f.read()
#      data=np.ones((len(dd)/1024/16/2,1024,16),dtype=float32)
#      data[:,:,:]=np.frombuffer(dd,dtype='<i2').reshape(len(dd)/1024/16/2,1024,16)
#      
#  schedule_start format
#  ON  
#  20230501-1730
#  120    in second

#from multiprocessing import Process, Pool
import sys
import socket
import numpy as np
import numpy.random as rand
import collections
import os

import bonsai
import time, resource

#from threading import Thread
from multiprocessing import Pool

#
############################################################
############################################################
HOST='172.19.51.103'
PORT=5001
Nx=1000        # time=1s
Ny=1024        # channel
Nz=16          # beam
nbeam=Nz       # number of beam in received data
beams=Nz       #int(sys.argv[3]), number of threads for each beam
t_total=1800    # ringbuffer time = t_total*NBLK  sec (e.g. 1000*1024*3600*4=14GB for 1 hour, take 70s to save)
itime = 0
dtime = 10000  # save ringbuffer for every dtime sec.
sn=10.         # triggering SN

BLK_SIZE = Nx*Ny*Nz   #1000*1024*16
NBLK = 2        # NBLK*NX is power of 16 for bonsai only
njump = 80
HDR_SIZE = 64
nchunk=Nx*NBLK

# sending trigger to ip/port 
if len(sys.argv) == 4:
    ip = sys.argv[1]
    port = int(sys.argv[2])
else:
    print("Run like : python2 bonsai_trigger_column_bin_tcp.py <arg1 server ip 192.168.1.102> <arg2 server port 4444 >")
    exit(1)


config = { 'nfreq': 1024,
       'freq_lo_MHz': 400.0,
       'freq_hi_MHz': 800.0,
       'dt_sample': 1.024e-3,
       'nt_chunk': nchunk,
       'ntrees': 1,
       'tree_size': 1024,
       'dm_coarse_graining_factor': 1,
       'time_coarse_graining_factor': 1 }

# Construct the dedisperser object from the config_params.
#
dedisp = bonsai.Dedisperser(config, use_analytic_normalization=True)
ring = collections.deque(maxlen=nbeam*t_total)
data1 = np.zeros((Nx*NBLK, Ny, Nz), dtype='float32')
for it in range(int(t_total)):
    for ibeams in range(int(nbeam)):
        ring.append(data1[:,:,ibeams])

#print('len of ring, nbeam*t_total =', len(ring),nbeam*t_total)


def multi_run_wrapper(args):
   return main_map(*args)

def main_map(id,itime):

    #id, itime = args
#   t0=time.time()
    data=data1[:,:,id]      # data1.shape = (Nx*NBLK, channel, beams)
    data=data.transpose()   # data.shape = (channel, Nx*NBLK)
    #print('data shape,data1.shape =',data.shape,data1.shape)
    #data=np.float32(data)
#   print(data[0,id])
    
    nchunks = 1

    weights = np.ones((dedisp.nfreq, dedisp.nt_chunk), dtype=np.float32)

    global_max_trigger = 0.0;
    #print >>sys.stderr, 'example1: running dedisperser..',


    #if bonsai._have_hdf5():
    # This built-in trigger_processor writes the coarse-grained triggers to an HDF5 file.
    # The bonsai._have_hdf5() function returns True if bonsai was compiled with HDF5 support.
        #hdf5_writer = bonsai.trigger_hdf5_writer('test_triggers.h5')
        #dedisp.add_processor(hdf5_writer)
        #print('no hdf')

    #if bonsai._have_png():
    # This built-in trigger_processor writes the coarse-grained triggers to a sequence of png files.
    # The plotter is very primitive and not suitable for production use, but I sometimes
    # run it as a simple visual sanity check.
        #plot_writer = bonsai.trigger_plot_writer('test_trigger_plot', Ny) # length of x, 1024
        #dedisp.add_processor(plot_writer)
        #print('no png')

    for ichunk in xrange(nchunks):
    # Dedisperse one chunk.
        dedisp.run(data, weights)

    # The output of the dedisperser is a set of "coarse-grained triggers".  After each chunk of input
    # data is processed, the corresponding set of coarse-grained triggers is available for processing.  
    #

        all_triggers = dedisp.get_triggers()
        #print('all_triggers=',all_triggers)

    # Here we just do some trivial "processing" of the coarse-grained triggers, by looping over
    # all the trigger arrays, and keeping track of the global maximum.
    
        for tp in all_triggers:
        # 'tp' is a 4D numpy array
            global_max_trigger = max(global_max_trigger, np.max(tp))

    dedisp.end_dedispersion();

    #print('spending %f sec') %(time.time()-t0)
    max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6 #KB to GB
    #mem_cost.append(max_mem)
#   print('max_mem (GB): ',max_mem)

#   print 'example1:', (nchunks * dedisp.nt_chunk), 'time samples successfully processed'
#   print 'example1: The global maximum trigger value was', global_max_trigger, ' sigma'

# sending trigger signal via tcp
#   if len(sys.argv) == 4:
#       ip = sys.argv[1]
#       port = int(sys.argv[2])
#   else:
#       print("Run like : python2 bonsai_trigger_column_bin_tcp.py <arg1 server ip 192.168.1.102> <arg2 server port 4444 >")
#       exit(1)

    send_data = "2"
    #print('global_max_trigger =',global_max_trigger)
    if global_max_trigger > sn:
        #print("send trigger to server \n")
        #s.send(send_data.encode('utf-8'))
        #os.system("./trigger.out 172.17.31.236 8000 1")
        cmd = "./trigger.out %s %s %s"%(ip, port, send_data)
        #os.system(cmd)
        #s.close()
        # copy data[] to npy file
        ts=time.time()
        pid = os.fork()
        if pid == 0:
           timestr=time.strftime("%Y%m%d-%H%M%S")
           sav_intensity=np.zeros((Nx*NBLK,Ny,t_total),dtype='float32')
        #print('sav_intensity.shape,len of ring = ',sav_intensity.shape,len(ring))
           ts=time.time()
           for it in range(int(t_total)):
           #print('trigger sn=',global_max_trigger)
               sav_intensity[:,:,it]=ring[it*nbeam+id-1]
           print('ProcessID:%s, t_total=%f sec, copy from ringbuffer takes %f sec') %(os.getpid(),t_total,time.time()-ts)    
           ts=time.time()
           path='./'   #'/mnt/ramdisk/'
           np.save(path+timestr+'_beam'+str(id)+'.npy',sav_intensity)
           print('Child ProcessID:%s, id=%s, np.save takes %f sec') %(os.getpid(),id,time.time()-ts)
    else:
        #print("SN is not large enough \n")
        print('Parent ProcessID:%s, id=%s, np.save takes %f sec') %(os.getpid(),id,time.time()-ts)
#   time.sleep(3)
    # save npy from ringbuffer every dtime for test
    if id == 2 and (itime % dtime) == 0 :
        ts=time.time()
        pid = os.fork()
        if pid == 0:
           timestr=time.strftime("%Y%m%d-%H%M%S")
           sav_intensity=np.zeros((Nx*NBLK,Ny,t_total),dtype='float32')
        #print('sav_intensity.shape,len of ring = ',sav_intensity.shape,len(ring))
           ts=time.time()
           for it in range(int(t_total)):
           #print('trigger sn=',global_max_trigger)
               sav_intensity[:,:,it]=ring[it*nbeam+id-1]
           print('ProcessID:%s, t_total=%f sec, copy from ringbuffer takes %f sec') %(os.getpid(),t_total,time.time()-ts)    
           ts=time.time()
           path='./'    #'/mnt/ramdisk/'
           np.save(path+timestr+'_beam'+str(id)+'.npy',sav_intensity)
           print('Child ProcessID:%s, id=%s, np.save takes %f sec') %(os.getpid(),id,time.time()-ts)
        else:
           print('Parent ProcessID:%s, id=%s, np.save takes %f sec') %(os.getpid(),id,time.time()-ts)
        #frh=open(timestr+"_beam"+str(id)+"_header.bin","ab")
        #frh.write(header)
        #frh.close()
    
# 
def readBlock(s):
    recv_size = s.recv_into(memoryview(header), HDR_SIZE)
    if recv_size != 64:
        print('Error receiving header, size=', recv_size)
        raise
    
    view = memoryview(buffer)
    next_offset = 0
    while BLK_SIZE*2 - next_offset > 0:
        recv_size = s.recv_into(view[next_offset:], BLK_SIZE*2 - next_offset)
        next_offset += recv_size


header = bytearray(HDR_SIZE)
buffer = bytearray(BLK_SIZE*2)
#data1 = np.zeros((1000*NBLK, 1024, 16), dtype='float32')
#beams=Nz        #int(sys.argv[3])

s=socket.socket()
s.connect((HOST, PORT))
ind = 0
write_status="off"
start_time="20000101-000000"
end_time="20000102-000000"
duration = 0
rtime = 0

while(True):
  t0=time.time()
  #s=socket.socket()
  #s.connect((HOST, PORT))
  #now=time.strftime("%Y%m%d-%H%M%S")
  #print(now)
  rq_path='./'
  f2=open(rq_path+'schedule_start','r')
  status=f2.read().split('\n')
  f2.close()
  if len(status[0]) != 2:
      print('schedule_start file error in the first line !')
  if status[0] == "ON":
    start_time = status[1]
    duration = int(status[2])
    f=open(start_time+"_buffer.bin","ab")
    fh=open(start_time+"_header.bin","ab")
    f2=open('schedule_start','w')
    f2.write('OX\n')
    f2.write(status[1]+'\n')
    f2.write(status[2]+'\n')
    f2.close()
  now=time.strftime("%Y%m%d-%H%M")
  #print(now)
  if start_time == now:
    write_status = "on"
    #rtime = 0
  if write_status == "on" and rtime >= duration:
    write_status = "off"
    rtime = 0
    f.close()
    fh.close()

  #print(status)
  for nb in range(NBLK):
    readBlock(s)
# check write buffer
    if write_status == "on":
      f.write(buffer)
      fh.write(header)
      rtime = rtime + 1
    #if write_status == "off" and rtime >= duration:
    #  f.close()
    #  fh.close()

    data1[nb*Nx:(nb+1)*Nx,:,:] = np.frombuffer(buffer, dtype='<i2').reshape((Nx, Ny, Nz))  
  #data1=np.reshape(data1,[subNx,Ny,Nz])

  # copy to ring_buffer
  if np.any(data1):
      print(' not all zero in data1')
  for ibeams in range(int(nbeam)):
      ring.append(data1[:,:,ibeams])

  #print('transfer time: %f sec') %(time.time()-t0)
  #s.close()

  t1 = time.time()
  if __name__ == '__main__':
      #with Pool() as pool:
          pool = Pool(nbeam)
          args = [(i, itime) for i in range(nbeam)]
          #print(args)
          pool_outputs = pool.map(multi_run_wrapper, args)
          
      #inputs = range(beams)
      ##inputs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
      #pool = Pool(beams)

      #pool_outputs = pool.map(main_map, inputs)

  print('searching %f sec, final spending %f sec') %(time.time()-t1, time.time()-t0)
  itime = itime + 1
  ind = ind + 1

  pool.close()


