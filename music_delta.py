# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 09:23:33 2019

@author: ecupl
"""

from pydub import AudioSegment
import os

os.chdir(r"D:")

sound = "yinghuochong.mp3"
start = "0:00"
stop = "1:43"
s = AudioSegment.from_mp3(sound)

start = 0
stop = 1000*(103)

s[start:stop]
