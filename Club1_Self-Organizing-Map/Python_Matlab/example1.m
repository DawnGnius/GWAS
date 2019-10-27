net = selforgmap([2 3],100, 3, 'gridtop', 'linkdist');
P = [.1 .3 1.2 1.1 1.8 1.7 .1 .3 1.2 1.1 1.8 1.7;...
0.2 0.1 0.3 0.1 0.3 0.2 1.8 1.8 1.9 1.9 1.7 1.8];

net = configure(net,P);
plotsompos(net,P)

net.trainParam.epochs = 50000;
net = train(net,P);
plotsompos(net,P)