K-means Clustering
==============================

General instructions
--------------------

-   In this task you will implement the **K-means** algorithm and its
    applications. We provide the bootstrap code and you are expected to
    complete the functions.
-   Do not import libraries other than those already imported in the
    original code.
-   Please follow the type annotations. You have to make the function's
    return values match the required type.
-   Only modifications in files {`kmeans.py`} in the \"work\" directory
    will be accepted and graded. All other modifications will be
    ignored. You can work directly on Vocareum, or download all files
    from \"work\", code in your own workspace, and then upload the
    changes (recommended).
-   Click the Submit button when you are ready to submit your code for
    auto-grading. Your final grade is determined by your **last**
    submission.

High Level Description
----------------------

In this assignment you are asked to implement:

-   the K-means++ initialization,
-   the standard K-means algorithm given the initialization,
-   the nearest neighbor algorithm using the K-means centroids,
-   image compression via the K-means algorithm.

NOTE: Depending on your environment you may need to install the python
library named \"pillow\", which is used by matplotlib to process some of
the images needed for this assignment. You can install it by running
\"pip3 install pillow\" in your command line.

Q1. K-means++ initialization
----------------------------

K-means++ generally performs much better than the vanilla K-means
algorithm. The only difference is in the initialization of the
centroids. According to the discussions in the lecture, implement this
initialization in function `get_k_means_plus_plus_center_indices`. (Note
that we also provide the vanilla initialization method in
`get_lloyd_k_means`.)

Q2. K-means algorithm
---------------------

Recall that for a dataset
[[[$x_{1},...,x_{N} \in R^{D}$]{.katex-mathml}[[[]{.strut
style="height: 0.73354em; vertical-align: -0.19444em;"}[[x]{.mord
.mathdefault}[[[[[[]{.pstrut style="height: 2.7em;"}[[1]{.mord
.mtight}]{.sizing .reset-size6 .size3
.mtight}]{style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"}]{.vlist
style="height: 0.301108em;"}[​]{.vlist-s}]{.vlist-r}[[]{.vlist
style="height: 0.15em;"}]{.vlist-r}]{.vlist-t
.vlist-t2}]{.msupsub}]{.mord}[,]{.mpunct}[]{.mspace
style="margin-right: 0.166667em;"}[.]{.mord}[.]{.mord}[.]{.mord}[,]{.mpunct}[]{.mspace
style="margin-right: 0.166667em;"}[[x]{.mord
.mathdefault}[[[[[[]{.pstrut style="height: 2.7em;"}[[N]{.mord
.mathdefault .mtight style="margin-right: 0.10903em;"}]{.sizing
.reset-size6 .size3
.mtight}]{style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"}]{.vlist
style="height: 0.328331em;"}[​]{.vlist-s}]{.vlist-r}[[]{.vlist
style="height: 0.15em;"}]{.vlist-r}]{.vlist-t
.vlist-t2}]{.msupsub}]{.mord}[]{.mspace
style="margin-right: 0.277778em;"}[∈]{.mrel}[]{.mspace
style="margin-right: 0.277778em;"}]{.base}[[]{.strut
style="height: 0.841331em; vertical-align: 0em;"}[[R]{.mord .mathdefault
style="margin-right: 0.00773em;"}[[[[[[]{.pstrut
style="height: 2.7em;"}[[D]{.mord .mathdefault .mtight
style="margin-right: 0.02778em;"}]{.sizing .reset-size6 .size3
.mtight}]{style="top: -3.063em; margin-right: 0.05em;"}]{.vlist
style="height: 0.841331em;"}]{.vlist-r}]{.vlist-t}]{.msupsub}]{.mord}]{.base}]{.katex-html
aria-hidden="true"}]{.katex}]{.katex--inline}, the K-means distortion
objective is:\
[[[[$F(\{\mu_{k}\},\{ r_{nk}\}) = \sum\limits_{i = 1}^{N}\sum\limits_{k = 1}^{K}r_{nk}\parallel\mu_{k} - x_{n}\parallel_{2}^{2}\qquad(1)$]{.katex-mathml}[[[]{.strut
style="height: 1em; vertical-align: -0.25em;"}[J]{.mord .mathdefault
style="margin-right: 0.13889em;"}[(]{.mopen}[{]{.mopen}[[μ]{.mord
.mathdefault}[[[[[[]{.pstrut style="height: 2.7em;"}[[k]{.mord
.mathdefault .mtight style="margin-right: 0.03148em;"}]{.sizing
.reset-size6 .size3
.mtight}]{style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"}]{.vlist
style="height: 0.336108em;"}[​]{.vlist-s}]{.vlist-r}[[]{.vlist
style="height: 0.15em;"}]{.vlist-r}]{.vlist-t
.vlist-t2}]{.msupsub}]{.mord}[}]{.mclose}[,]{.mpunct}[]{.mspace
style="margin-right: 0.166667em;"}[{]{.mopen}[[r]{.mord .mathdefault
style="margin-right: 0.02778em;"}[[[[[[]{.pstrut
style="height: 2.7em;"}[[[n]{.mord .mathdefault .mtight}[k]{.mord
.mathdefault .mtight style="margin-right: 0.03148em;"}]{.mord
.mtight}]{.sizing .reset-size6 .size3
.mtight}]{style="top: -2.55em; margin-left: -0.02778em; margin-right: 0.05em;"}]{.vlist
style="height: 0.336108em;"}[​]{.vlist-s}]{.vlist-r}[[]{.vlist
style="height: 0.15em;"}]{.vlist-r}]{.vlist-t
.vlist-t2}]{.msupsub}]{.mord}[}]{.mclose}[)]{.mclose}[]{.mspace
style="margin-right: 0.277778em;"}[=]{.mrel}[]{.mspace
style="margin-right: 0.277778em;"}]{.base}[[]{.strut
style="height: 3.13045em; vertical-align: -1.30211em;"}[[[[[[]{.pstrut
style="height: 3.05em;"}[[[n]{.mord .mathdefault .mtight}[=]{.mrel
.mtight}[1]{.mord .mtight}]{.mord .mtight}]{.sizing .reset-size6 .size3
.mtight}]{style="top: -1.87233em; margin-left: 0em;"}[[]{.pstrut
style="height: 3.05em;"}[∑]{.mop .op-symbol
.large-op}]{style="top: -3.05001em;"}[[]{.pstrut
style="height: 3.05em;"}[[N]{.mord .mathdefault .mtight
style="margin-right: 0.10903em;"}]{.sizing .reset-size6 .size3
.mtight}]{style="top: -4.30001em; margin-left: 0em;"}]{.vlist
style="height: 1.82834em;"}[​]{.vlist-s}]{.vlist-r}[[]{.vlist
style="height: 1.27767em;"}]{.vlist-r}]{.vlist-t .vlist-t2}]{.mop
.op-limits}[]{.mspace style="margin-right: 0.166667em;"}[[[[[[]{.pstrut
style="height: 3.05em;"}[[[k]{.mord .mathdefault .mtight
style="margin-right: 0.03148em;"}[=]{.mrel .mtight}[1]{.mord
.mtight}]{.mord .mtight}]{.sizing .reset-size6 .size3
.mtight}]{style="top: -1.84789em; margin-left: 0em;"}[[]{.pstrut
style="height: 3.05em;"}[∑]{.mop .op-symbol
.large-op}]{style="top: -3.05001em;"}[[]{.pstrut
style="height: 3.05em;"}[[K]{.mord .mathdefault .mtight
style="margin-right: 0.07153em;"}]{.sizing .reset-size6 .size3
.mtight}]{style="top: -4.30001em; margin-left: 0em;"}]{.vlist
style="height: 1.82834em;"}[​]{.vlist-s}]{.vlist-r}[[]{.vlist
style="height: 1.30211em;"}]{.vlist-r}]{.vlist-t .vlist-t2}]{.mop
.op-limits}[]{.mspace style="margin-right: 0.166667em;"}[[r]{.mord
.mathdefault style="margin-right: 0.02778em;"}[[[[[[]{.pstrut
style="height: 2.7em;"}[[[n]{.mord .mathdefault .mtight}[k]{.mord
.mathdefault .mtight style="margin-right: 0.03148em;"}]{.mord
.mtight}]{.sizing .reset-size6 .size3
.mtight}]{style="top: -2.55em; margin-left: -0.02778em; margin-right: 0.05em;"}]{.vlist
style="height: 0.336108em;"}[​]{.vlist-s}]{.vlist-r}[[]{.vlist
style="height: 0.15em;"}]{.vlist-r}]{.vlist-t
.vlist-t2}]{.msupsub}]{.mord}[∥]{.mord}[[μ]{.mord
.mathdefault}[[[[[[]{.pstrut style="height: 2.7em;"}[[k]{.mord
.mathdefault .mtight style="margin-right: 0.03148em;"}]{.sizing
.reset-size6 .size3
.mtight}]{style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"}]{.vlist
style="height: 0.336108em;"}[​]{.vlist-s}]{.vlist-r}[[]{.vlist
style="height: 0.15em;"}]{.vlist-r}]{.vlist-t
.vlist-t2}]{.msupsub}]{.mord}[]{.mspace
style="margin-right: 0.222222em;"}[−]{.mbin}[]{.mspace
style="margin-right: 0.222222em;"}]{.base}[[]{.strut
style="height: 1.11411em; vertical-align: -0.25em;"}[[x]{.mord
.mathdefault}[[[[[[]{.pstrut style="height: 2.7em;"}[[n]{.mord
.mathdefault .mtight}]{.sizing .reset-size6 .size3
.mtight}]{style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"}]{.vlist
style="height: 0.151392em;"}[​]{.vlist-s}]{.vlist-r}[[]{.vlist
style="height: 0.15em;"}]{.vlist-r}]{.vlist-t
.vlist-t2}]{.msupsub}]{.mord}[[∥]{.mord}[[[[[[]{.pstrut
style="height: 2.7em;"}[[2]{.mord .mtight}]{.sizing .reset-size6 .size3
.mtight}]{style="top: -2.453em; margin-left: 0em; margin-right: 0.05em;"}[[]{.pstrut
style="height: 2.7em;"}[[2]{.mord .mtight}]{.sizing .reset-size6 .size3
.mtight}]{style="top: -3.113em; margin-right: 0.05em;"}]{.vlist
style="height: 0.864108em;"}[​]{.vlist-s}]{.vlist-r}[[]{.vlist
style="height: 0.247em;"}]{.vlist-r}]{.vlist-t
.vlist-t2}]{.msupsub}]{.mord}[]{.mspace
style="margin-right: 2em;"}[(]{.mopen}[1]{.mord}[)]{.mclose}]{.base}]{.katex-html
aria-hidden="true"}]{.katex}]{.katex-display}]{.katex--display}

where [[[$µ_{1},...,µ_{K}$]{.katex-mathml}[[[]{.strut
style="height: 0.522771em; vertical-align: -0.19444em;"}[[µ]{.mord}[[[[[[]{.pstrut
style="height: 2.7em;"}[[1]{.mord .mtight}]{.sizing .reset-size6 .size3
.mtight}]{style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"}]{.vlist
style="height: 0.301108em;"}[​]{.vlist-s}]{.vlist-r}[[]{.vlist
style="height: 0.15em;"}]{.vlist-r}]{.vlist-t
.vlist-t2}]{.msupsub}]{.mord}[,]{.mpunct}[]{.mspace
style="margin-right: 0.166667em;"}[.]{.mord}[.]{.mord}[.]{.mord}[,]{.mpunct}[]{.mspace
style="margin-right: 0.166667em;"}[[µ]{.mord}[[[[[[]{.pstrut
style="height: 2.7em;"}[[K]{.mord .mathdefault .mtight
style="margin-right: 0.07153em;"}]{.sizing .reset-size6 .size3
.mtight}]{style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"}]{.vlist
style="height: 0.328331em;"}[​]{.vlist-s}]{.vlist-r}[[]{.vlist
style="height: 0.15em;"}]{.vlist-r}]{.vlist-t
.vlist-t2}]{.msupsub}]{.mord}]{.base}]{.katex-html
aria-hidden="true"}]{.katex}]{.katex--inline} are centroids of the K
clusters and [[[$r_{ik} \in {0,1}$]{.katex-mathml}[[[]{.strut
style="height: 0.6891em; vertical-align: -0.15em;"}[[r]{.mord
.mathdefault style="margin-right: 0.02778em;"}[[[[[[]{.pstrut
style="height: 2.7em;"}[[[n]{.mord .mathdefault .mtight}[k]{.mord
.mathdefault .mtight style="margin-right: 0.03148em;"}]{.mord
.mtight}]{.sizing .reset-size6 .size3
.mtight}]{style="top: -2.55em; margin-left: -0.02778em; margin-right: 0.05em;"}]{.vlist
style="height: 0.336108em;"}[​]{.vlist-s}]{.vlist-r}[[]{.vlist
style="height: 0.15em;"}]{.vlist-r}]{.vlist-t
.vlist-t2}]{.msupsub}]{.mord}[]{.mspace
style="margin-right: 0.277778em;"}[∈]{.mrel}[]{.mspace
style="margin-right: 0.277778em;"}]{.base}[[]{.strut
style="height: 0.83888em; vertical-align: -0.19444em;"}[[{0]{.mord}[,]{.mpunct}[]{.mspace
style="margin-right: 0.166667em;"}[1}]{.mord}]{.mord}]{.base}]{.katex-html
aria-hidden="true"}]{.katex}]{.katex--inline} represents whether example
n belongs to cluster k.\
\
In this part, you need to implement the K-means procedure that
iteratively computes the new cluster centroids and assigns data points
to the new clusters. The procedure stops whenever 1) the number of
updates has reached the given maximum number, or 2) when the \*average\*
K-means distortion objective J changes less than a given threshold
between two iterations.\
\
Implement this part in the `fit`function of the class `KMeans`. While
assigning a sample to a cluster, if there is a tie (i.e. the sample is
equidistant from two or more centroids), you should choose the one with
the smaller index (which is what numpy.argmin does already).

After you complete the implementation, run `KmeansTest.py` to see the
results of this on a toy dataset. You should be able to see three images
generated in a folder called plots. In particular, you can see
`toy_dataset_predicted_labels.png` and `toy_dataset_real_labels.png`,
and compare the clusters identified by the algorithm against the real
clusters. Your implementation should be able to recover the correct
clusters sufficiently well. Representative images are shown below. Red
dots are cluster centroids. Note that color coding of recovered clusters
may not match that of correct clusters. This is due to mis-match in
ordering the retrieved clusters and the correct clusters (which is
fine).

![](clusters.png)

Q3 Classification with K-means
------------------------------

Another application of clustering is to obtain a faster version of the
nearest neighbor algorithm. Recall that nearest neighbor evaluates the
distance of a test sample from every training point to predict its
label, which can be very slow. Instead, we can compress the entire
training dataset to just K centroids, where each centroid is now labeled
as the majority class of the corresponding cluster. After this
compression the prediction time of nearest neighbor is reduced from O(N)
to just O(K) (see below for the pseudocode).\
\
![](Algo.png)\
\
You need to complete the `fit` and `predict` function in
`KMeansClassifier` following the comments in the code. Again, whenever
you need to break a tie, pick the one with the smallest index.\
\
Once completed, run `KmeansTest.py` again to evaluate the classifier on
a test set (digits). For comparison, the script will also print accuracy
of a logistic classifier and a vanilla nearest neighbor classifier. An
example is shown below. ![](classification_output.png)

Q4 Image compression with K-means
---------------------------------

In this part, we will take lossy image compression as another
application of clustering. The idea is simply to treat each pixel of an
image as a point, then perform K-means algorithm to cluster these
points, and finally replace each pixel with its closest centroid.\

What you need to implement is to compress an image with K centroids
given (called code\_vectors). Specifically, complete the function
`transform_image` following the comments in the code.\
\
After your implementation, run `KmeansTest.py` again. You should be able
to see an image `compressed_baboon.png` in the plots folder. You can see
that this image is slightly distorted as compared to the original
baboon.tiff. The ideal result should take about 35-40 iterations and the
Mean Square Error (between the two images) should be less than 0.0098.
It takes about 1-2 minutes to complete normally.
