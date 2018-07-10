#TensorFlow

Installation and setting TensorFlow

Link: https://www.tensorflow.org

## Installation on Windows

###Requirements to run TensorFlow with GPU support (*NVIDIA GTX 1080 Ti*)

CUDA Toolkit 9.0: [http://developer.nvidia.com/cuda-downloads](http://developer.nvidia.com/cuda-downloads)  
CUDNN v7.0: [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)

### Anaconda Cloud

Site: https://anaconda.org  
[Anaconda installer archive](https://repo.continuum.io/archive/)

Install *Anaconda3-4.2.0-Windows-x84_64.exe* (Python 3.5.2)

```
C:\> pip3 install --upgrade tensorflow-gpu

```

 
#### Validate installation

```{.python}
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
```

## Installation on Ubuntu 16.04

### Prerequisite
Python and Pip

```
sudo apt-get install python-pip3 python3-dev # for Python 3.n
```

### Install TensorFlow

1. Install TensorFlow

	```
	$ pip3 install tensorflow-gpu
	```
	
2. If Step 1 failed, install the latest version of TensorFlow

	```
	$ sudo pup3 install --upgrade
	```
	
#### Validate installation

```{.python}
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
```
	