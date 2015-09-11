# -*- coding: utf-8 -*-
"""
Created on Mon Jun 01 10:09:44 2015

@author: peyang
"""

import pip
installed_packages = pip.get_installed_distributions()
installed_packages_list = sorted(["%s==%s" % (i.key, i.version)
     for i in installed_packages])
print(installed_packages_list)
