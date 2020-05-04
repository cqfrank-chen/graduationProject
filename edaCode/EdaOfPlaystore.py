import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
import scipy.stats as stats
import plotly.express as px

df = pd.read_csv('../googlePlayStore/googleplaystore.csv')


df.drop_duplicates('App', 'first', inplace=True)
df = df[df['Installs'] != 'Free']
df = df[df['Installs'] != 'Paid']
df = df[df['Android Ver'] != np.nan]
df = df[df['Android Ver'] != 'NaN']
# df.to_csv('../googlePlayStore/googleplaystoreFilted.csv')
# 
# print('the number of apps in dataset:', len(df))
# print(df.sample(7))

# 规范Installs的数据，去掉 '+' 和 ','
df['Installs'] = df['Installs'].apply(lambda x: str(x).replace('+', '') if '+' in str(x) else x)
df['Installs'] = df['Installs'].apply(lambda x: str(x).replace(',', '') if ',' in str(x) else x)
df['Installs'] = df['Installs'].apply(lambda x: float(x))
# print(df['Installs'].values)

# 统一size的单位，去除varies with device等噪音
df['Size'] = df['Size'].apply(lambda x: str(x).replace('Varies with device', 'NaN') if 'Varies with device' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: str(x).replace(',', '') if ',' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: float(str(x).replace('k', '')) / 1000 if 'k' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: float(x))

# 去掉price的单位
df['Price'] = df['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else x)
df['Price'] = df['Price'].apply(lambda x: float(x))

df['Reviews'] = df['Reviews'].apply(lambda x: float(x))
# df.to_csv('../googlePlayStore/googleplaystoreFilted.csv')


# # basic EDA
# x = df['Rating'].dropna()
# y = df['Size'].dropna()
# z = df['Installs'][df.Installs != 0].dropna()
# p = df['Reviews'][df.Reviews != 0].dropna()
# t = df['Type'].dropna()
# price = df['Price']
# p = sns.pairplot(pd.DataFrame(list(zip(x, y, np.log(z), np.log10(p), t, price)), columns=['Rating', 'Size', 'Installs', 'Reviews', 'Type', 'Price']), hue='Type', palette='Set2', diag_kind="hist")
# plt.savefig('EDA.png')

# # 探究对最受欢迎的应用类型
# numberOfCategory = df['Category'].value_counts().sort_values(ascending=False)
# # print(numberOfCategory)
# data = [go.Pie(
#     labels = numberOfCategory.index,
#     values = numberOfCategory.values,
#     hoverinfo = 'label+value')]
# plotly.offline.plot(data, filename='populationOfCategory.html')

# # 评分分布情况和平均值
# data = [go.Histogram(
#     x = df.Rating,
#     xbins = {'start': 1, 'size': 0.1, 'end': 5}
# )]
# print('Average app rating = ', np.mean(df['Rating']))
# plotly.offline.plot(data, filename='ratingDistribution.html')

# # 单因素方差分析，验证各类型应用的评分分布是不同的
# # 数据不符合正态分布，所以不能使用单因素方差分析
# f = stats.f_oneway(df.loc[df.Category == 'FAMILY']['Rating'].dropna(),
#                 df.loc[df.Category == 'GAME']['Rating'].dropna(),
#                 df.loc[df.Category == 'TOOLS']['Rating'].dropna(),
#                 df.loc[df.Category == 'BUSINESS']['Rating'].dropna(),
#                 df.loc[df.Category == 'MEDICAL']['Rating'].dropna(),
#                 df.loc[df.Category == 'PERSONALIZATION']['Rating'].dropna(),
#                 df.loc[df.Category == 'PRODUCTIVITY']['Rating'].dropna(),
#                 df.loc[df.Category == 'LIFESTYLE']['Rating'].dropna(),
#                 df.loc[df.Category == 'FINANCE']['Rating'].dropna()
#                 )
# # print(f)
# groups = df.groupby('Category').filter(lambda x: len(x) > 286).reset_index()
# array = groups['Rating'].hist(by = groups['Category'], sharex=True, figsize=(20,20))
# plt.savefig('OneWayAnova.png')

# # 评分较好的应用类型
# groups = df.groupby('Category').filter(lambda x: len(x) > 170).reset_index()
# print('Average Rating = ', np.nanmean(list(groups['Rating'])))
# layout = {
#     'title': 'App ratings across major categories',
#     'xaxis': {'tickangle': -40},
#     'yaxis': {'title': 'Rating'},
#     'plot_bgcolor': 'rgb(250,250,250)',
#     'shapes': [{
#         'type': 'line',
#         'x0': -.5,
#         'y0': np.nanmean(list(groups.Rating)),
#         'x1': 19,
#         'y1': np.nanmean(list(groups.Rating)),
#         'line': {'dash': 'dash'}
#     }]
# }
# data = [{
#     'y': df.loc[df.Category == category]['Rating'],
#     'type': 'violin',
#     'name': category,
#     'showlegend': False
# } for i, category in enumerate(list(set(groups.Category)))]
# plotly.offline.plot({'data': data, 'layout': layout}, filename='BestPerformingCategory.html')


# # 软件大小的影响
# groups = df.groupby('Category').filter(lambda x: len(x) > 50).reset_index()
# sns.set_style('darkgrid')
# ax = sns.jointplot(df.Size, df.Rating, kind='hex')
# plt.savefig('sizeVSrating.png')

# subdf = df[df.Size > 40]
# tmpGroups = subdf.groupby('Category').filter(lambda x: len(x) > 20)
# layout = {
#     'title': 'Rating vs Size',
#     'xaxis': {'title': 'Rating'},
#     'yaxis': {'title': 'Size(in MB)'},
#     'plot_bgcolor': 'black'
# }
# data = [{
#     'x': tmpGroups.loc[subdf.Category == category]['Rating'],
#     'type': 'scatter',
#     'y': subdf['Size'],
#     'name': category,
#     'mode': 'markers',
#     'showlegend': True
# } for i, category in enumerate(['GAME', 'FAMILY'])]
# plotly.offline.plot({'data': data, 'layout': layout}, filename='ratingVSsize.html')


# # 软件价格的影响
# paidApps = df[df.Price > 0]
# p = sns.jointplot('Price', 'Rating', paidApps)
# plt.savefig('priceVSrating.png')

# subdf = df[df.Category.isin(['GAME', 'FAMILY', 'PHOTOGRAPHY', 'MEDICAL', 'TOOLS', 'FINANCE', 'LIFESTYLE', 'BUSINESS'])]
# sns.set_style('darkgrid')
# fig, ax = plt.subplots()
# fig.set_size_inches(15, 8)
# p = sns.stripplot(x = 'Price', y = 'Category', data = subdf, jitter=True, linewidth=1)
# title = ax.set_title('App price trend across categories')
# plt.savefig('priceVScategory.png')

# filtedSubdf = subdf[subdf.Price < 100]
# fig, ax = plt.subplots()
# fig.set_size_inches(15, 8)
# p = sns.stripplot(x = 'Price', y = 'Category', data = filtedSubdf, jitter=True, linewidth=1)
# title = ax.set_title('App price trend across categories - after filtering for junk apps')
# plt.savefig('priceVScategory-filted.png')

# newDF = df.groupby(['Category', 'Type']).agg({'App': 'count'}).reset_index()
# # print(newDF)
# outerGroupNames = ['GAME', 'FAMILY', 'MEDICAL', 'TOOLS']
# outerGroupValues = [len(df.App[df.Category == category]) for category in outerGroupNames]
# # print(majorCateNames, majorCateValues)
# a, b, c, d = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Purples]
# innerGroupNames = ['Paid', 'Free'] * 4
# innerGroupValues = []
# for category in outerGroupNames:
#     for t in ['Paid', 'Free']:
#         x = newDF[newDF.Category == category]
#         try:
#             innerGroupValues.append(int(x.App[x.Type == t].values[0]))
#         except:
#             innerGroupValues.append(0)
# explode = (0.025, 0.025, 0.025, 0.025)
# fig, ax = plt.subplots(figsize = (10, 10))
# ax.axis(option='equal')
# mypie, texts, _ = ax.pie(outerGroupValues, radius=1.2, labels=outerGroupNames, autopct='%1.1f%%',
#                          pctdistance=1.1, labeldistance=0.75, explode=explode, colors=[a(0.6), b(0.6), c(0.6), d(0.6)],
#                          textprops={'fontsize': 16})
# plt.setp(mypie, width=0.5, edgecolor='black')
# mypie2, _ = ax.pie(innerGroupValues, radius=1.2-0.5, labels=innerGroupNames, labeldistance=0.7,
#                     textprops={'fontsize': 12}, colors=[a(0.4), a(0.2), b(0.4), b(0.2), c(0.4), c(0.2), d(0.4), d(0.2)])
# plt.setp(mypie2, width=0.5, edgecolor='black')
# plt.margins(0,0)
# plt.tight_layout()
# plt.savefig('freePaidPie.png')

# newdf = df.copy()
# newdf['PriceBand'] = None
# newdf.loc[df.Price == 0, 'PriceBand'] = '0 free'
# newdf.loc[(df.Price > 0) & (df.Price <= 0.99), 'PriceBand'] = '1 cheap'
# newdf.loc[(df.Price > 0.99) & (df.Price <= 2.99), 'PriceBand'] = '2 not cheap'
# newdf.loc[(df.Price > 2.99) & (df.Price <= 4.99), 'PriceBand'] = '3 normal'
# newdf.loc[(df.Price > 4.99) & (df.Price <= 14.99), 'PriceBand'] = '4 expensive'
# newdf.loc[(df.Price > 14.99) & (df.Price <= 29.99), 'PriceBand'] = '5 too expensive'
# newdf.loc[df.Price > 29.99, 'PriceBand'] = '6 astronomical figures'

# newdf[['PriceBand', 'Rating']].groupby('PriceBand', as_index=False).mean()
# p = sns.catplot(x='PriceBand', y='Rating', data=newdf, kind='boxen', height=10, palette='Pastel1')
# p.set_xticklabels(rotation = 90)
# p.set_ylabels('Rating')
# ax = plt.gca()
# fig = plt.gcf()
# ax.set_title('Rating VS PriceBand')
# fig.set_size_inches(8, 15)
# fig.subplots_adjust(top=0.95, bottom=0.2)
# plt.savefig('pricebandVSrating.png')



# # 价格对于下载量得影响
# trace0 = go.Box(
#     y=np.log10(df['Installs'][df.Type == 'Paid']),
#     name='Paid',
#     marker=dict(color='rgb(214,12,140)')
# )
# trace1 = go.Box(
#     y=np.log10(df['Installs'][df.Type == 'Free']),
#     name='Free',
#     marker=dict(color='rgb(0, 128, 128)')
# )
# layout = go.Layout(
#     title='number of downloads of paid apps Vs free apps',
#     yaxis={'title': 'numbe of downloads (log-scaled)'}
# )
# data = [trace0, trace1]
# plotly.offline.plot({'data': data, 'layout': layout}, filename='paidVSfree.html')

# # 各属性相关性分析
# corrMat = df.corr()
# p = sns.heatmap(corrMat, annot=True, cma  p=sns.diverging_palette(220, 20, as_cmap=True))
# plt.savefig('correlations.png')
# newdf = df.copy()
# newdf = newdf[newdf.Reviews > 10]
# newdf = newdf[newdf.Installs > 0]
# newdf['Installs'] = np.log10(df['Installs'])
# newdf['Reviews'] = np.log10(df['Reviews'])
# p = sns.lmplot('Reviews', 'Installs', data=newdf)
# fig = plt.gcf()
# fig.set_size_inches(8, 8)
# ax = plt.gca()
# ax.set_title('Number of Reviews Vs Number of Downloads (Log Scaled)')
# plt.savefig('reviewsVSdownloads.png')

# KMeans-cluster

def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

def randCent(dataMat, k):
    # print('ddddd=', dataMat)
    n = np.shape(dataMat)[1]
    centroIds = np.mat(np.zeros((k, n)))
    for j in range(n):
        minJ = min(dataMat[:, j])
        rangeJ = float(max(dataMat[:, j]) - minJ)
        centroIds[:, j] = np.mat(minJ + rangeJ * np.random.rand(k, 1))
    return centroIds

def getDistance(dataMat, centList):
    m,n = np.shape(centList)
    distance = np.zeros((m, np.shape(dataMat)[0]))
    for i in range(m):
        for j in range(np.shape(dataMat)[0]):
            distance[i, j] = distEclud(centList[i], dataMat[j])
    distance = np.min(distance, axis=0)
    return distance

def rollSelect(distance):
    n = len(distance)
    cnt = np.zeros(n)
    p = distance / np.sum(distance)
    cumP = np.cumsum(p)
    # print(cumP)
    for i in range(int(0.15 * n)):
        randNum = np.random.random()
        for j in range(n):
            if cumP[j] >= randNum:
                cnt[j] += 1
                break
    selectIndex = sorted(cnt)[-1]
    return selectIndex

def plusCent(dataMat, k):
    centList = dataMat[np.random.randint(np.shape(dataMat)[0])]
    # print(type(centList))
    for i in range(k-1):
        distance = getDistance(dataMat, centList)
        selectIndex = rollSelect(distance)
        # print('select:', selectIndex)
        centList = np.row_stack((centList, dataMat[int(selectIndex)]))
    return centList

def kMeans(dataMat, k, distMeans=distEclud, createCent=plusCent):
    m = np.shape(dataMat)[0]
    clusterAssment = np.mat(np.zeros((m,2)))
    centroIds = createCent(dataMat, k)
    # print('123124124=',centroIds)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = distMeans(centroIds[j, :], dataMat[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex: # 0号族是没有记录距离的
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2
        # print(centroIds)
        for cent in range(k):
            ptsInClust = dataMat[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroIds[cent, :] = np.mean(ptsInClust, axis=0)
    return centroIds, clusterAssment

def biKmeans(dataMat, k ,distMeans=distEclud):
    # print(type(dataMat))
    m = np.shape(dataMat)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroId0 = np.mean(dataMat, axis=0).tolist()[0]
    # print(centroId0)
    centList = [centroId0]
    # print(centList)
    for j in range(m):
        # print(np.mat(centroId0))
        # print(dataMat.iloc[j, :])
        clusterAssment[j, 1] = distMeans(np.mat(centroId0), dataMat[j, :]) ** 2

    while len(centList) < k:
        lowestSSE = np.inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataMat[np.nonzero(clusterAssment[:, 0].A == i)[0], :]
            # print('pppppp=',ptsInCurrCluster)
            centroIdMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeans)
            sseSplit = np.sum(splitClustAss[:, 1])
            sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print('sseSplit, sseNotSplit', sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroIdMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
    
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit is :', bestCentToSplit)
        print('the len of bestClustAss is :', len(bestClustAss))

        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] =  bestClustAss
    return np.mat(centList), clusterAssment

def nomalization(x):
    Range = max(x) - min(x)
    m = min(x)
    return [(float(i) - m) / Range for i in x]

newdf = df[['Rating', 'Size', 'Price']]
newdf = newdf.dropna()
newdf = newdf.loc[(newdf.Price > 0) & (newdf.Price < 5)]
newdf_copy = newdf.copy()
# print(newdf)
# print(newdf.reset_index())


# print(newdf.describe())
# newdf.to_csv('../googlePlayStore/tmpdf.csv')

newdf['Rating'] = nomalization(newdf['Rating']) 
# newdf['Reviews'] = nomalization(newdf['Reviews']) 
newdf['Size'] = nomalization(newdf['Size'])
newdf['Price'] = nomalization(newdf['Price'])
# newdf['Installs'] = nomalization(newdf['Installs'])

# centList, clusterAssment = kMeans(np.mat(newdf), 3)
# # print(clusterAssment)
# clusterAssment = pd.DataFrame(clusterAssment)
# clusterAssment.to_csv('../googlePlayStore/tmp.csv')
# # print('centList=', centList)
# # print('clusterAssment=', clusterAssment)

# newdf_copy = newdf_copy.reset_index()
# newdf_copy['Cluster'] = clusterAssment[0]
# # print(newdf)
# f = px.scatter_3d(newdf_copy, x='Rating', y='Size', z='Price', color='Cluster')
# plotly.offline.plot(f, filename='clusterKmeans++.html')

## k值得选择
# sse = [0]
# dfMat = np.mat(newdf)
# for k in range(1,6):
#     centList, clusterAssment = biKmeans(dfMat, k)
#     sum = 0
#     for i in range(np.shape(dfMat)[0]):
#         sum += distEclud(centList[int(clusterAssment[i, 0])], dfMat[i])
#     sse.append(sum)
# print(sse)
# plt.plot(sse)
# # plt.xticks(range(1,6))
# plt.xlim(1, 5)
# plt.ylim(80, 180)
# plt.xlabel('K')
# plt.ylabel('SSE')
# plt.savefig('selectK.png')
# plt.show()