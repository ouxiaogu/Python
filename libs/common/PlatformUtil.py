"""
-*- coding: utf-8 -*-
Created: peyang, 2018-01-25 11:33:29

Last Modified by: ouxiaogu

PlatformUtil: identify current platform,
to ensure some cross platform compatibility
"""

from sys import platform
import sys
__all__ = ['inLinux', 'inWindows']

'''
if platform == "linux" or platform == "linux2":
elif platform == "darwin":
elif platform == "win32":
'''

def inLinux():
    return (platform == "linux" or platform == "linux2")

def inWindows():
    '''https://stackoverflow.com/a/2145582/1819824'''
    return (platform == "win32")

def home():
    if (sys.version_info > (3, 5)):
        from pathlib import Path
        home = str(Path.home())
    else:
        from os.path import expanduser
        home = expanduser("~")
    return home

if __name__ == '__main__':
    print(inWindows() )
    print(home() )
