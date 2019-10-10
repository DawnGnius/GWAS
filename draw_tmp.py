import pylab as pl
pl.figure(figsize=(10, 10))
pl.title("SOM Node")
pl.subplot(3,6,4)
pl.title("Cluster 0 (n=50)")
pl.ylim(-1,25)
pl.plot([1,2],self.W[0,:], 'ko-')
pl.subplot(3,6,5)
pl.title("Cluster 1 (n=0)")
pl.ylim(-1,25)

pl.subplot(3,6,6)
pl.title("Cluster 2 (n=50)")
pl.ylim(-1,25)
pl.plot([1,2],self.W[2,:], 'ko-')
pl.subplot(3,6,10)
pl.title("Cluster 3 (n=50)")
pl.ylim(-1,25)
pl.plot([1,2],self.W[3,:], 'ko-')
pl.subplot(3,6,11)
pl.title("Cluster 4 (n=3)")
pl.ylim(-1,25)
pl.plot([1,2],self.W[4,:], 'ko-')
pl.subplot(3,6,12)
pl.title("Cluster 5 (n=0)")
pl.ylim(-1,25)

pl.subplot(3,6,16)
pl.title("Cluster 6 (n=50)")
pl.ylim(-1,25)
pl.plot([1,2],self.W[6,:], 'ko-')
pl.subplot(3,6,17)
pl.title("Cluster 7 (n=50)")
pl.ylim(-1,25)
pl.plot([1,2],self.W[7,:], 'ko-')
pl.subplot(3,6,18)
pl.title("Cluster 8 (n=50)")
pl.ylim(-1,25)
pl.plot([1,2],self.W[8,:], 'ko-')

