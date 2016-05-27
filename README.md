# cnncancer
lung cancer pathological image analyzer based on convolutional autoencoder

extern_params.py		:define parameters and global variables
tensorflow_ae_base.py		:basic functions for CNN
tensorflow_ae_stage1.py		:define network of the first stage
tensorflow_train_stage1.py	:train network of the first stage
myutil.py			:utility functions

DATA PREPARATION

First, you need to convert input jpeg data to npy files.
Since the original jpeg is 512x512, and hundreds pictures for each sample,
we randomly pickup and divide it smaller slices such as 32x32, 64x64 etc.
This process is described in 'make_tcge_sample.py'.

BEFORE exec the script 'make_tcge_sample.py' you need to set parameter variables
for example,
nx = ny = 32   # slice size 
sx = sy = 32   # stride size
na = 1         # number of sample sets
nn  = 6400     # number of train slice for each set
'nn' is a number of slices in a single file.

When you increase slice size (nx,ny), you'd better to reduce 'nn'
to avoid creating too large input data file.
However, it decreases the number of trainable samples.
So we you can save the data multiple 'na' files separately.
For example, to use double sized training set, you can set
# nx=ny=sx=sy=64
# na,nn=4,1600
4 times larger slice size, so 4 fold smaller slice number but
for each data file, so create 4 different data files.


The original data will be found under
"/project/hikaku_db/data/tissue_images"
or you may copy them to your local, for example,
"/home/UNAME/Documents/data/tissue_images"   ( in linux, or )
"/Users/UNAME/Documents/data/tissue_images"  ( in Mac OS X )
where UNAME is your username.

We have a bunch of samples downloaded from TCGA,
At present, we use some of them for testing.
In 'make_tcge_sample.py', we use a list below,
("TCGA-05-4384-01A-01-BS1_files/15/",
"TCGA-38-4631-11A-01-BS1_files/15/",
"TCGA-05-4425-01A-01-BS1_files/15/")

The processed npy data will be put in the directory
"/home/UNAME/Documents/data/tissue_images/input_w**"
"/Users/UNAME/Documents/data/tissue_images/input_w**"
where ** corresponds the size of the slices (i.e. nx).


LEARNING

Once you have prepared sample data, you can exec network trainig.
It will be easyer to train the network layer by layer.
So the first optimization will be
'tensorflow_train_stage1.py'
it will read
qqq_trn_w32_1.npy

BEFORE you run the optimization, you can set the max iteration steps,
tmax
and, the step to show intermidiate socre
tprint

the default values are
tmax, tprint = 3,1

They will be defined in the script 'extern_params.py' as global variables
ONLY IF THEY ARE NOT DEFINED.
So if you use interactive python shell, you may set these variables
interactively, otherwise, you may set them anywhere in your script.

Now you can run the script 'tensorflow_train_stage1.py'.
The optimized network parameters will be saved as
'out1/weight1.DDD.pkl' and 'out1/bias1.DDD.pkl'
where DDD shows the time stamp (year-month-day-hour-min) of the executed process.
# to avoid overwriting trained data, it would be better such time stamp
# suffix should be appended, automatically.

You may continue the optimization from the saved parameters by setting

stamp1 = 'DDD'

then exec 'tensorflow_train_stage1.py' again.
It will load 'out1/weight1.DDD.pkl' and 'out1/bias1.DDD.pkl' automatically,
and continue 'tmax' steps to optimize.


LEARNING SECOND LAYER

We are implementing stacked autoencoder.
In means the encoding layer of the first stage will be input of the
convolution layer of the next stage.

Once you optimized the network of stage1,
use 'make_tcge_encoded1.py' to save the output tof the encoding layer.

Set 'stamp1' to choose the parameter values before, and 
it will create files named
'qqq_encode1_tcga_w32_1.DDD.npy'
where DDD is the value of 'stamp1' corresponds to the saved parameters
'out1/weight1.DDD.pkl' and 'out1/bias1.DDD.pkl'.

Then exec 'tensorflow_train_stage2.enc.py' to optimize the networks of
the second stage.


VERIFY

*up to coming*