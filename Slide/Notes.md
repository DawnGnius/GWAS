# Notes

## Section page

Hey guys, in this part, I will show you some visualized simulation, which will then lead to the conclusion. This part would be much simple than the former 2.
Just looking at the picture, you will understand som.

## Model and Setting

Before the simulation, let's focus on the model and setting.

We generate a 2-D r.v. from a mixtrue distribution with K compoents.
Each compoent is a normal distribution with mean $ \mu_i $ and covariance matrix I, which is a identity matrix.

And, our data has 300 observations, 6 classes, each class has 50 observations. Those are their means.

Further, as introduced in section 2, we initialize SOM with a retangular grid and Batch size, neighborhood function and learning rate function, as following.

## Visualization or initialization

Ok, we prepare a short gif to illustrate the process of SOM algorithem. You can open that Website to have a look.

Let's look at the pictures. The dark points in the left panel are data points, and the red lines and points form a 3*3 grid in the initial state. It just covers the data points.
In the right upper panel, we draw a function representing how the radius of neighborhood change with the iteration times. The right lowwer panel shows a learning rate function. This is only one of setting of SOM. Note that, both of D and R are decressing with t.

In each iteration, We use 100 random samples from data set. And then put them into the **NN**, they will stimulate the nearest Node and Nodes within the radius D. And influence those Nodes by a proportion of R. **Formula**
By the way, we set the iteration number as 35, the rightmost end of the coordinate axis.

We divide the whole process into three parts. Let's goon.

## Stage 1

You can follow the green points to get the current parameters.
In the initial stage, D and R are relateivly large.
Which means Nodes are widely influenced by the inputs.

You can see, position of Nodes have changed a lot.
Because each input will effect almost all Nodes with the bigest learning rate.

## Stage 2

Radius of Neighborhood decreases with time t, when t is about 10.
The situation changed.
Each input will influence the Node around them about 2 distance. Those winner nodes take all. WTA: each node competes with each other only winners could update themselves. 

## Stage 3

Pass

## SOM Result

Now, lets look at the result of SOM.
It gives 7 clusters more than the real number.
In left picture, Different clusters are ploted with different colors. And points show their position.
The right panel contains 9 sub-pictures, they correspond to the 3*3 grid.
the picture shows the parameters/weights of each node. 
the middle node is very interesting. it has only 2 observations, and it is very familiar with the left one.
After we campare their cendiods, we can merge those two clusters.

what's more, there two Node with no observation. It's really interesting.

## Other experiment

In UCI ML repository, there are many data sets for clustering or classification.
We choose 5 data sets, they are listed in the first column. Maybe you are familiar with some of they.
They have different numbers observations from 150 to more than 4000, num. of Attributes, area and data categorical.

They are so famous, Thanks to their reputation, we found a paper experiment with those data set. and compare with K-means.
The numbers in the table are calculated by a special formula, we only need know that the smaller the better. It called DB index.

In terms of this index, SOM is better than K-means.

## Summary

Now let's summar up the method.
First, we compare with K-means.

1. K-Means suffers more from initialization. Let's recall the K-mean. **K**
some nodes in SOM is allowed to have empty set.

2. K-Means suffers more from noise.

3. SOM gives a elegant topological diagram, did well in visualization than K-Means.

## Advantages

SOM is useful for visualizing low-dimensional views of high-dimensional data

## Disadvantages

It may not converage, it depends on the parameters and functions choosen. 
Its algorithem is complex than K-means, need more computation time.

It perform poor on unbalanced data high-dim data.

Althought, it doesnot require input the number of clusters. It require other things. 

A cluster of som doesnot match a nature cluster

## Thank you
