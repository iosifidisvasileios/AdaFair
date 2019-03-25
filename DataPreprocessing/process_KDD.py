from __future__ import division
import urllib2
import os,sys
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn import feature_extraction
from sklearn import preprocessing
from random import randint

output = open("sampledKDD.csv", "w")
myFile = open("kdd.csv", "r")

output.write(myFile.readline())
cnt = 0
for line in myFile:
	if "0" in line.split(",")[-1]:
		if randint(0, 13) < 2 :

			cnt +=1
			output.write(line)
	else:
		output.write(line)

print cnt
output.close()
myFile.close()