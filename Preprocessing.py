import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_item=pd.read_csv(r'C:\tianchi_fresh_comp_train_item.csv')
data_user=pd.read_csv(r'C:\tianchi_fresh_comp_train_user.csv')

#观察数据，geohash缺失超过50%，且维度过高，暂时忽略该特征
data_item.info()
data_item.item_geohash.value_counts().count()
data_user.info()
data_user.user_geohash.value_counts().count()

#观察用户购买总数分布，未购买的有2346人，这部分人无法观察购买倾向，无法预测之后是否会购买 
data_count=data_user[data_user['behavior_type']==4].user_id.value_counts().to_frame().reset_index()
data_count=data_count.user_id.value_counts()
data_count=data_count.sort_index().to_frame().reset_index()
plt.figure(figsize=(6.0,4.0))
plt.bar(data_count.index,data_count.user_id)
plt.xlabel('times_buy')
plt.ylabel('count_user')
plt.title('user_buy')
plt.show()

#观察商品被购买总数分布，被购买的商品绝大部分只被买了1、2次，交互用户数低，协同过滤算法不适用该数据集
#结合两张图：user_buy的分布比item_buy的分布均衡，原因是很多用户重复购买同种商品→用户对商品的回购率可能是一个有效特征
data_count_1=data_user[data_user['behavior_type']==4].item_id.value_counts().to_frame().reset_index()
data_count_1=data_count_1.item_id.value_counts()
data_count[0]=4588033
data_count_1=data_count_1.sort_index().to_frame().reset_index()*100
data_count_1=data_count_1.drop(index=0)
buy_rate=data_count_1.item_id.sum()/data_user.item_id.nunique()
print('%.2f'%buy_rate,'%')
plt.figure(figsize=(6.0,4.0))
plt.bar(data_count_1.index,data_count_1.item_id)
plt.xlabel('times_buy')
plt.ylabel('count_item')
plt.title('item_buy')
plt.show()

#观察转化率，加购和收藏明显高于浏览，不同行为的次数是影响是否购买的因素之一
data_behavior=data_user.behavior_type.value_counts()
browse_buy=data_behavior[4]/data_behavior[1]*100
collect_buy=data_behavior[4]/data_behavior[3]*100
cart_buy=data_behavior[4]/data_behavior[2]*100
print('浏览转化率：','%.2f'%browse_buy,'%',
    '收藏转化率：','%.2f'%collect_buy,'%',
      '加购转化率：','%.2f'%cart_buy,'%')

#整理数据，提取在商品子集上的用户行为，去除没有购买的user记录
date=data_user.time.str.split(' ',expand=True)
date.columns=['day','time']
data_user=pd.concat([data_user,date],axis=1)
data_user=data_user.drop(columns='time')
data_user=data_user.drop(columns='user_geohash')
data=data_user[data_user['item_id'].isin (data_item['item_id'])]
new_user=data[data['behavior_type']==4].user_id.unique()
non_user=pd.DataFrame(list(set(data.user_id.unique()).difference(set(new_user))))
data=data[~data['user_id'].isin(non_user)]

#观察行为总数在时序上的分布，12.12频数明显高于其他时间
sample=data
sample=sample.groupby(by=['day','behavior_type']).count().reset_index()
sample=sample[['behavior_type','day','user_id']]
y=sample.user_id.values
x=sample.day.values
plt.bar(x,y,fc='grey')

#观察购买总数时序分布，12.12前后不受12.12影响，符合整体的分布，只需除去12.12这天的数据
sample=data
sample=sample.groupby(by=['day','behavior_type']).count().reset_index()
sample=sample[['behavior_type','day','user_id']]
y1=sample[sample['behavior_type']==1].reset_index()
y_1=y1.user_id.values
x_1=y1.day.values
y2=sample[sample['behavior_type']==2].reset_index()
y_2=y2.user_id.values
y3=sample[sample['behavior_type']==3].reset_index()
y_3=y3.user_id.values
y4=sample[sample['behavior_type']==4].reset_index()
y_4=y4.user_id.values
plt.figure(figsize=(18.0,10.0))
plt.subplot(221)
plt.bar(x_1,y_1,fc='b')
plt.subplot(222)
b=plt.bar(x_1,y_2,fc='r')
plt.subplot(223)
c=plt.bar(x_1,y_3,fc='y')
plt.subplot(224)
d=plt.bar(x_1,y_4,fc='g')
plt.show()
plt.bar(x,y,fc='grey')
     
      
