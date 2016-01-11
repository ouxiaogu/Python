

## [how-to-execute-a-file-within-the-python-interpreter](http://stackoverflow.com/questions/1027714/how-to-execute-a-file-within-the-python-interpreter)

Reminder: if you want to run it in SublimeText, Environment Variable of python must be added

1. `python -i program.py`
2.  execfile( "someFile.py")
  
  ```python
  variables= {}
  execfile( "someFile.py", variables )
  print variables # globals from the someFile module
  ```


  set HTTP_PROXY=proxy1-as.asml.com\peyang:peyang@myproxy:8080

## [Install downloaded module in Pyhon](http://stackoverflow.com/questions/7322334/how-to-use-python-pip-install-software-to-pull-packages-from-github)

for windows users:

1) I first download and unpack the file.

2) Then in the python directory going to \Scripts

3) Starting here the command prompt

4) `pip install C:\Theano-master` # theano-master is example library

Another method

`C:\Users\peyang\canopy\User\Scripts\python.exe setup.py build --compiler=mingw64 install --user` term of options can't be exchanged.


### Anaconda Usage

1. usage install OpenCV as an example

```
anaconda search -t conda opencv
conda install 
```


2.
