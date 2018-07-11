import os, os.path
import re
import sys


PhotoPath = r"D:\DCIM\100CANON"

# 1. scan the JPG file, and keep the

class ScanFile(object):   #scan the file and derive all the subfiles and directions
    def __init__(self,directory):
        self.directory=directory
    def scan_files(self, prefix=None, postfix=None):
        files_list=[]
        for dirpath,dirnames,filenames in os.walk(self.directory):
            for special_file in filenames:
                if prefix and postfix:
                    if special_file.endswith(postfix) and special_file.startswith(prefix):
                        files_list.append(os.path.join(dirpath,special_file))
                elif postfix:
                    if special_file.endswith(postfix):
                        files_list.append(os.path.join(dirpath,special_file))
                elif prefix:
                    if special_file.startswith(prefix):
                        files_list.append(os.path.join(dirpath,special_file))
                else:
                    files_list.append(os.path.join(dirpath,special_file))
        return files_list
    def strip_basename(self, file_list, postfix=None):
        return [ aa.strip(postfix) for aa in file_list]

    def scan_subdir(self):
        subdir_list=[]
        for dirpath,dirnames,files in os.walk(self.directory):
            subdir_list.append(dirpath)
        return subdir_list

scan = ScanFile(PhotoPath)
jpgfiles = scan.scan_files(postfix=".JPG")
basenames = scan.strip_basename(jpgfiles, postfix=".JPG")
CR2files = scan.scan_files(postfix=".CR2")
for cur_raw_file in CR2files:
    if not any(aa in cur_raw_file for aa in basenames):
        os.rename( cur_raw_file, os.path.join(PhotoPath, os.path.basename(cur_raw_file).strip('.CR2')+'.CR2.rm') )
RMfiles = scan.scan_files(postfix=".rm")

print jpgfiles
print CR2files
print RMfiles
