'''
-*- coding: utf-8 -*-
Created: peyang, 2018-01-12 16:25:58

Last Modified by:  ouxiaogu

FileUtil: File/Folder handling utility module
'''

import os.path
import string
import re
from PlatformUtil import inLinux, inWindows
import sys

__all__ = [ 'gpfs2WinPath', 'consolidatePathSep',
            'splitFileName', 'getFileLabels', 'genOutFilePath'
            'FileScanner']

def gpfs2WinPath(src):
    dst = os.path.normpath(src)
    if inLinux():
        return dst
    prefix = r"\\devshare-brion.asml.com\cnfs-"
    if inWindows():
        if 'gpfs' not in src:
            # print("Warning, can't find 'gpfs' in input file: {}".format(dst))
            return dst
            # raise IOError(e)
        try:
            dst = string.replace(dst, r'/', r"\\")
            dst = string.replace(dst, '\gpfs\\', prefix, maxreplace=1)
        except AttributeError:
            dst = dst.replace(r'/', r"\\")
            dst = dst.replace('\gpfs\\', prefix, 1)
    return dst

def splitFileName(src):
    dirname, basename = os.path.split(src)
    try:
        if (sys.version_info > (3, )):
            filelabel, extension = basename.rsplit(sep='.', maxsplit=1)
        else:
            filelabel, extension = string.rsplit(basename, sep='.', maxsplit=1)
    except ValueError:
        filelabel = basename
        extension = ''
    return dirname, filelabel, extension

def getFileLabels(files):
    return [list(splitFileName(aa))[1] for aa in files]

def genOutFilePath(src, postfix="_new", extn=None):
    dirname, filelabel, extension = splitFileName(src)
    extension = extension if extn is None else extn
    dst = os.path.join(dirname, filelabel+postfix+'.'+extension)
    return dst

def consolidatePathSep(src):
    dst = src
    if inWindows():
        dst = os.path.join(*src.split('/'))
    else:
        dst = os.path.join(*src.split('\\'))
    return dst

class FileScanner(object): # can be replaced by glob
    '''scan the file and derive all the subfiles and directions'''
    def __init__(self, directory):
        if not os.path.exists(directory):
            e = "Can't find path: {}".format(directory)
            raise IOError(e)
        self.directory=directory

    def is_valid_file(self, dirname, basename, prefix=None, postfix=None, regex_pattern=None):
        '''check whether current file name meet the file name filter'''
        if not os.path.isfile(os.path.join(dirname, basename)):
            return False
        if prefix and  (not basename.startswith(prefix)):
            return False
        if postfix and (not basename.endswith(postfix)):
            return False
        if regex_pattern and (re.search(regex_pattern, basename) is None):
            return False
        return True

    def scan_file_recursive(self, prefix=None, postfix=None, regex_pattern=None):
        '''os.walk to recursively scan file in directory and its subdirs'''
        files_list=[]
        for dirpath,dirnames,filenames in os.walk(self.directory):
            for filename in filenames:
                if not self.is_valid_file(dirpath, filename, prefix=prefix, postfix=postfix, regex_pattern=regex_pattern):
                    continue
                files_list.append(os.path.join(dirpath, filename))
        return files_list

    def scan_file_non_recursive(self, prefix=None, postfix=None, regex_pattern=None):
        '''os.listdir to only scan file with current directory, depth=1'''
        files_list = []
        for filename in os.listdir(self.directory):
            if not self.is_valid_file(self.directory, filename, prefix=prefix, postfix=postfix, regex_pattern=regex_pattern):
                continue
            files_list.append(os.path.join(self.directory, filename))
        return files_list

    def scan_files(self, prefix=None, postfix=None, regex_pattern=None, recursive=False):
        files_list = []
        if recursive == True:
            files_list = self.scan_file_recursive(prefix=prefix, postfix=postfix, regex_pattern=regex_pattern)
        else:
            files_list = self.scan_file_non_recursive(prefix=prefix, postfix=postfix, regex_pattern=regex_pattern)
        return files_list

    def scan_subdir(self):
        subdir_list=[]
        for dirpath,dirnames,files in os.walk(self.directory):
            subdir_list.append(dirpath)
        return subdir_list

if __name__ == '__main__':
    # '''test 1'''
    INDIR = '/gpfs/WW/BD/MXP/SHARED/SEM_IMAGE/IMEC/Case02_calaveras_v3/3Tmp/ContourSelection/007_ContourSel_Tree_debug'
    INDIR = gpfs2WinPath(INDIR)
    print(INDIR)

    # '''test 2'''
    # fsn = FileScanner(INDIR)
    # files = fsn.scan_files(postfix=r'.pgm') ## , recursive=True)
    # labels = getFileLabels(files)
    # print(files)
    # print(labels)

    '''test 3'''
    # path = r"C:\Localdata\D\4Development\imageSynthesisTool\data\image"
    # fsn = FileScanner(path)
    # files = fsn.scan_files(regex_pattern=r'.tif') ## not use * in regex header
    # labels = getFileLabels(files)
    # print(files)
    # print(labels)

    # print(re.match('.tif', 'Tech_Demo_rectangular_BWALL_set_8img_08302015_merged.csv'))
    # print(re.search('.tif', r'a.tif'))
    
    path = 'test'
    print(os.path.dirname(path)=='')
    print(os.path.basename(path))
    print(splitFileName(path))