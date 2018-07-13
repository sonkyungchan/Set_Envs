# TensorFlow

Installation and setting TensorFlow

Link: https://www.tensorflow.org

## Requirements to run TensorFlow with GPU support (*NVIDIA GTX 1080 Ti*)

CUDA Toolkit 9.0: [http://developer.nvidia.com/cuda-downloads](http://developer.nvidia.com/cuda-downloads)  
CUDNN v7.0.5 (CUDA v9.0): [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)

## Anaconda Cloud

Site: https://anaconda.org  
[Anaconda installer archive](https://repo.continuum.io/archive/)


## Installation on Windows

1. Install Anaconda

	Download and Install [*Anaconda3-4.2.0-Windows-x84_64.exe*](https://repo.continuum.io/archive/Anaconda3-4.2.0-Windows-x86_64.exe) (Python 3.5.2)

2. Install TensorFlow GPU

	```
	C:\> pip3 install --upgrade tensorflow-gpu
	```

## Installation on Ubuntu 16.04

### 1. Install Nvidia Graphic Driver & CUDA for TensorFlow GPU

1. Nvidia Graphic Driver
	
	Add PPA and install
	
	```
	$ sudo add-apt-repository ppa:graphics-drivers/ppa
	$ sudo apt-get update
	$ sudo apt-get install nvidia-384
	```
	Reboot after installation

	```
	$ sudo reboot
	```
	
	Check installation

	```
	$ nvidia-smi
	```
	
	
2. CUDA Toolkit v9.0

	Download runfile of Ubuntu 16.04 on download page and run
	[Link](https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run)
	
	```
	$ sudo sh cuda_9.0.176_384.81_linux.run
	```
	
	Answer the questions for installation
	
	```	
	Do you accept the previously read EULA?
	accept/decline/quit: accept
	
	Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 384.81?
	(y)es/(n)o/(q)uit: n
	
	Install the CUDA 9.0 Toolkit?  
	(y)es/(n)o/(q)uit: y
	
	Enter Toolkit Location  
	 [ default is /usr/local/cuda-9.0 ]: 
	
	Do you want to install a symbolic link at /usr/local/cuda?  
	(y)es/(n)o/(q)uit: y
	
	Install the CUDA 9.0 Samples?  
	(y)es/(n)o/(q)uit: n
	```
	
	Set Environment variables
	
	1.  Use Terminal

		```
		$ echo -e "\n## CUDA and cuDNN paths"  >> ~/.bashrc
		$ echo 'export PATH=/usr/local/cuda-9.0/bin:${PATH}' >> ~/.bashrc
		$ echo 'export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:${LD_LIBRARY_PATH}' >> ~/.bashrc
		```
	2. Use gedit

		```
		$ sudo gedit ~/.bashrc
		```

		Add texts below
		
		```
		## CUDA and cuDNN paths 
		export PATH = /usr/local/cuda-9.0/bin : $ { PATH } 
		export LD_LIBRARY_PATH = /usr/local/cuda-9.0/lib64 : $ { LD_LIBRARY_PATH }
		```
		
		```
		$ source ~/.bashrc
		```

3. CuDNN
	
	Download CuDNN v7.0.5 for CUDA 9.0 (Login required) [Link](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.0.5/prod/9.1_20171129/cudnn-9.1-linux-x64-v7)
	
	```
	$ tar xzvf cudnn-9.0-linux-x64-v7.tgz
	$ sudo cp cuda/lib64/* /usr/local/cuda-9.0/lib64/
	$ sudo cp cuda/include/* /usr/local/cuda-9.0/include/
	$ sudo chmod a+r /usr/local/cuda-9.0/lib64/libcudnn*
	$ sudo chmod a+r /usr/local/cuda-9.0/include/cudnn.h
	```
	Success if you can see the texts below
	
	```
	$ cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
	#define CUDNN_MAJOR 7
	#define CUDNN_MINOR 0
	#define CUDNN_PATCHLEVEL 5
	--
	#define CUDNN_VERSION    (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)

	#include "driver_types.h"
	```
	
4. Nvidia CUDA Profiler Tools Interface

	```
	$ sudo apt-get install libcupti-dev
	```	
	
### 2. Installat TensorFlow GPU

#### Installation on Anaconda (Not conda environment)


1. Install Anaconda3 4.2.0 (Python 3.5.2)
	1. Download and install [Anaconda3-4.3.0-Linux-x86_64.sh](https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh) on Terminal

	```
	$ sudo sh Anaconda3-4.2.0-Linux-x86_64.sh
	```
	
2. Install TensorFlow GPU

	```
	$ pip install tensorflow-gpu
	```

#### Installation on Native Pip

##### Prerequisite

Python and Pip

```
sudo apt-get install python-pip3 python3-dev # for Python 3.n
```

#### Install TensorFlow

1. Install TensorFlow GPU

	```
	$ pip3 install tensorflow-gpu
	```
	
2. If Step 1 failed, install the latest version of TensorFlow

	```
	$ sudo pup3 install --upgrade
	```


## Validate installation

```{.python}
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```
	