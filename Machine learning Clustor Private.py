# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 19:55:46 2018

@author: ecupl
"""
import pandas as pd
import numpy as np
import os
os.chdir("D:\\mywork\\test\\ML_CCB")
pd.set_option('max_columns',200)
pd.set_option('max_rows',200)
data_ori = pd.read_excel("CLUSTER_PRIVATE.xlsx")
'''挑出特征变量'''
data_feature=pd.DataFrame()
data_feature=data_ori[['id', 'bodong1', 'bodong2', 'bodong3', 'bodong4', 'bodong5', 'bodong6',
       'gender', 'constellation','address']]
'''挑出数值变量'''
data_number=data_ori.drop(['id', 'bodong1', 'bodong2', 'bodong3', 'bodong4', 'bodong5', 'bodong6',
       'gender', 'constellation','address'],axis=1)
data_number.describe().T
data_number.age.value_counts()      #年龄众数56
data_number.age=data_number.age.fillna(56)      #年龄用众数填补缺失值
data_number=data_number.fillna(0)       #剩下的缺失就当作0
'''相关系数矩阵'''
corrmatrix=data_number.corr(method='pearson')
corrmatrix1=corrmatrix[np.abs(corrmatrix)>0.5]
'''消费数据'''
data_number_consume=data_number.loc[:,'crcrd_totmoney':'other_cnum']
'''消费数据矩阵'''
corrmatrixc=data_number_consume.corr(method='pearson')

#%%
#################星座和年龄################
data_constellation = data_feature
data_constellation[["age","avgaum","avgcash"]]=data_number[["age","avg_monthaum","avg_cashaum"]]
'''去除空值'''
data_constellation.drop(data_constellation[data_constellation.constellation.isnull()].index,inplace=True)

'''玫瑰图'''
from pyecharts import Pie
attr = list(data_constellation.constellation.value_counts().index)
v1 = list(data_constellation.constellation.value_counts())
pie = Pie("星座数量-玫瑰图", title_pos='center', width=900)
pie.add(
    "私行客户数",
    attr,
    v1,
    center=[50, 50],
    is_random=True,
    radius=[20, 75],
    rosetype="area",
    is_legend_show=False,
    is_label_show=True,
)
pie.render()

'''热力图'''
bins = [0,20,30,40,50,60,70,80,100]
data_constellation['age_bins'] = pd.cut(data_constellation['age'],bins,labels=False)
from pyecharts import HeatMap, Bar, Grid
'''星座-年龄数量分布'''
crosstab_constellation = pd.crosstab(data_constellation['age_bins'],data_constellation.constellation)
x_axis = list(crosstab_constellation.columns)
y_axis = [
    "20以下",
    "20~30",
    "30~40",
    "40~50",
    "50~60",
    "60~70",
    "70~80",
    "80以上"
]
data = [[i, j, crosstab_constellation.iloc[j,i]] for i in range(12) for j in range(8)]
heatmap = HeatMap("热力图")
heatmap.add(
    "星座-年龄私行客户数量分布",
    x_axis,
    y_axis,
    data,
    is_visualmap=True,
    visual_top="70%",
    visual_text_color="#000",
    visual_orient="horizontal",
    visual_range=[crosstab_constellation.values.min(),crosstab_constellation.values.max()]
)
grid = Grid(height=600)
grid.add(heatmap,grid_bottom="40%")
grid.render('星座-年龄数量热力图.html')
'''星座-年龄人均AUM分布'''
data_constellation["avgaum"][data_constellation["avgaum"]>vmax]=vmax
crosstab_constellation_avgaum = pd.crosstab(data_constellation['age_bins'],data_constellation.constellation,
                                            values=data_constellation.avgaum,aggfunc=np.mean)
crosstab_constellation_avgaum=crosstab_constellation_avgaum.fillna(0)
x_axis2 = list(crosstab_constellation_avgaum.columns)
y_axis2 = [
    "20以下",
    "20~30",
    "30~40",
    "40~50",
    "50~60",
    "60~70",
    "70~80",
    "80以上"
]
data2 = [[i, j, crosstab_constellation_avgaum.iloc[j,i]] for i in range(12) for j in range(8)]
heatmap = HeatMap("热力图")
heatmap.add(
    "星座-年龄私行人均AUM分布",
    x_axis2,
    y_axis2,
    data2,
    is_visualmap=True,
    visual_top="70%",
    visual_text_color="#000",
    visual_orient="horizontal",
    visual_range=[crosstab_constellation_avgaum.values.min(),crosstab_constellation_avgaum.values.max()]
)
grid = Grid(height=600)
grid.add(heatmap,grid_bottom="40%")
grid.render('星座-年龄人均AUM热力图.html')
'''射手座80岁以上太高，故进行相应调整，原来是5.75645e+07'''
'''星座-年龄存款分布'''
data_constellation["avgcash"][data_constellation["avgcash"]>vmax]=vmax
crosstab_constellation_avgcash = pd.crosstab(data_constellation['age_bins'],data_constellation.constellation,
                                            values=data_constellation.avgcash,aggfunc=np.mean)
crosstab_constellation_avgcash=crosstab_constellation_avgcash.fillna(0)
x_axis3 = list(crosstab_constellation_avgcash.columns)
y_axis3 = [
    "20以下",
    "20~30",
    "30~40",
    "40~50",
    "50~60",
    "60~70",
    "70~80",
    "80以上"
]
data3 = [[i, j, crosstab_constellation_avgcash.iloc[j,i]] for i in range(12) for j in range(8)]
heatmap = HeatMap("热力图")
heatmap.add(
    "星座-年龄私行人均存款分布",
    x_axis3,
    y_axis3,
    data3,
    is_visualmap=True,
    visual_top="70%",
    visual_text_color="#000",
    visual_orient="horizontal",
    visual_range=[crosstab_constellation_avgcash.values.min(),crosstab_constellation_avgcash.values.max()]
)
grid = Grid(height=600)
grid.add(heatmap,grid_bottom="40%")
grid.render('星座-年龄人均存款热力图.html')
'''不超过三倍标准差'''
'''散点图'''
import matplotlib.pyplot as plt
'''年龄-人均AUM'''
data_constellation["avgaum"][data_constellation["avgaum"]>100000000]=100000000
fig1 = plt.figure(figsize=(12,8),dpi=80)
plt.scatter(data_constellation.age,data_constellation.avgaum)
plt.show
'''年龄-人均存款'''
data_constellation["avgcash"][data_constellation["avgcash"]>100000000]=100000000
fig2 = plt.figure(figsize=(12,8),dpi=80)
plt.scatter(data_constellation.age,data_constellation.avgcash)
plt.show
'''年龄-人均AUM深浅图'''
import seaborn as sns
sns.jointplot(x='avgaum', y='age', data=data_constellation,kind='kde')

#%%
##############波动情况##############
data_bodong=data_ori.loc[:,'id':'address']
data_shangxia=pd.DataFrame(columns=['increase','decrease','ping'])
for i in range(8192):
    increase = 0
    decrease = 0
    ping = 0
    cst=data_bodong.iloc[i]
    for j in range(1,7):
        if cst[j]==1:
            increase+=1
        elif cst[j]==-1:
            decrease+=1
        else:
            ping+=1
    data_shangxia=data_shangxia.append({'increase':increase,'decrease':decrease,'ping':ping},ignore_index=True)
data_bodong[['increase','decrease','ping']]=data_shangxia[['increase','decrease','ping']]
'''上升玫瑰饼图'''
shang = data_bodong.increase.value_counts().sort_index()
xia = data_bodong.decrease.value_counts().sort_index()
zhong = data_bodong.ping.value_counts().sort_index()
from pyecharts import Pie
attr = ['0次', '1次', '2次', '3次', '4次', '5次', '6次']
v1 = list(shang)
pie = Pie("AUM上升次数统计图", title_pos='left', width=900)
pie.add(
    "上升次数对应客户数",
    attr,
    v1,
    center=[50, 50],
    is_random=True,
    radius=[20, 75],
    rosetype="area",
    is_legend_show=True,
    is_label_show=True,
)
pie.render("shang.html")
'''下降玫瑰饼图'''
attr = ['0次', '1次', '2次', '3次', '4次', '5次', '6次']
v2 = list(xia)
pie = Pie("AUM下降次数统计图", title_pos='left', width=900)
pie.add(
    "下降次数对应客户数",
    attr,
    v2,
    center=[50, 50],
    is_random=True,
    radius=[20, 75],
    rosetype="area",
    is_legend_show=True,
    is_label_show=True,
)
pie.render("xia.html")
'''持平玫瑰饼图'''
attr = ['0次', '1次', '2次', '3次', '4次', '5次', '6次']
v2 = list(xia)
pie = Pie("AUM持平次数统计图", title_pos='left', width=900)
pie.add(
    "持平次数对应客户数",
    attr,
    v2,
    center=[50, 50],
    is_random=True,
    radius=[20, 75],
    rosetype="area",
    is_legend_show=True,
    is_label_show=True,
)
pie.render("zhong.html")
'''极坐标图，有点不好看，先放弃了'''
from pyecharts import Polar
angle = ['0次', '1次', '2次', '3次', '4次', '5次', '6次']
polar = Polar("波动次数极坐标图", width=1500, height=800)
shang = data_bodong.increase.value_counts().sort_index()
xia = data_bodong.decrease.value_counts().sort_index()
zhong = data_bodong.ping.value_counts().sort_index()
polar.add(
    "上升",
    list(shang),
    angle_data=angle,
    type="barAngle",
    is_stack=True,
)
polar.add(
    "下降",
    list(xia),
    angle_data=angle,
    type="barAngle",
    is_stack=True,
)
polar.add(
    "持平",
    list(zhong),
    angle_data=angle,
    type="barAngle",
    is_stack=True,
)
polar.render("波动总体情况.html")
'''星座-波动数量热力图'''

#%%
############转移#############
from pyecharts import GeoLines, Style
style = Style(
    title_top="#fff",
    title_pos = "center",
    title_color = "#FFFFFF",
    width=1600,
    height=1000,
    background_color="#646464"
)
style_geo = style.add(
    is_label_show=True,
    line_curve=0.2,
    line_opacity=1,
    legend_text_color="#eee",
    legend_pos="right",
    geo_effect_symbol="plane",
    geo_effect_symbolsize=15,
    label_color=['#1E90FF','#00FF00', '#ffa022'],
    label_pos="right",
    label_formatter="{b}",
    label_text_color="#FFFFFF",
    label_text_size=24,
    geo_normal_color='#F4A460',
    geo_emphasis_color='#FFD700'
)

style_geo1 = style.add(
    is_label_show=True,
    is_geo_effect_show =True,
    line_curve=0.2,
    line_opacity=1,
    legend_text_color="#15e8ff",
    legend_pos="right",
    geo_effect_symbol="arrow",
    geo_effect_symbolsize=15,
    label_color=[ '#1E90FF','#00FF00', '#ffa022'],
    label_pos="right",
    label_formatter="{b}",
    label_text_color="#FFFFFF",
    label_text_size=24,
    geo_normal_color='#F4A460',
    geo_emphasis_color='#FFD700'

)
    
data_zhuanru = [["温州", "上海"],
                ["嘉善", "上海"],
                ["沈阳", "上海"],
                ["蚌埠", "上海"],
                ["长春", "上海"],
                ["南通", "上海"],
                ["北京", "上海"],
                ["成都", "上海"],
                ["大同", "上海"],
                ["济宁", "上海"],
                ["中山", "上海"]                
]
data_zhuanchu = [
    ["上海","酒泉"],
    ["上海","无锡"],
    ["上海","衡阳"]
]

geolines = GeoLines("客户关系转移一览", **style.init_style)
geolines.add("申请转入", data_zhuanru, **style_geo)
geolines.add("被申请转出", data_zhuanchu, **style_geo1)
geolines.render("客户关系转移导航图.html")




