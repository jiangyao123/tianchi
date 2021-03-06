#构建特征user-item行为次数
def user_item_feature(data,end_date):
    daybefore_1=data[data['day']==end_date- datetime.timedelta(days=1)]
    user_item_1=pd.crosstab([daybefore_1['user_id'],daybefore_1['item_id']],daybefore_1.behavior_type)
    daybefore_2=data[(data['day']<=end_date- datetime.timedelta(days=1))&(data['day']>=end_date- datetime.timedelta(days=2))]
    user_item_2=pd.crosstab([daybefore_2['user_id'],daybefore_2['item_id']],daybefore_2.behavior_type)
    daybefore_3=data[(data['day']<=end_date- datetime.timedelta(days=1))&(data['day']>=end_date- datetime.timedelta(days=3))]
    user_item_3=pd.crosstab([daybefore_3['user_id'],daybefore_3['item_id']],daybefore_3.behavior_type)
    daybefore_4=data[(data['day']<=end_date- datetime.timedelta(days=1))&(data['day']>=end_date- datetime.timedelta(days=4))]
    user_item_4=pd.crosstab([daybefore_4['user_id'],daybefore_4['item_id']],daybefore_4.behavior_type)
    daybefore_5=data[(data['day']<=end_date- datetime.timedelta(days=1))&(data['day']>=end_date- datetime.timedelta(days=5))]
    user_item_5=pd.crosstab([daybefore_5['user_id'],daybefore_5['item_id']],daybefore_5.behavior_type)
    user_item_repeatperiod=data[(data['day']<=end_date- datetime.timedelta(days=1))&(data['behavior_type']==4)]
    user_item_repeattime=pd.crosstab([ user_item_repeatperiod['user_id'],user_item_repeatperiod['item_id']],data.behavior_type)
    user_item_feature=pd.merge(user_item_5, user_item_4,how='left',right_index=True,left_index=True) 
    user_item_feature=pd.merge(user_item_feature, user_item_3,how='left',right_index=True,left_index=True) 
    user_item_feature=pd.merge(user_item_feature, user_item_2,how='left',right_index=True,left_index=True)
    user_item_feature=pd.merge(user_item_feature, user_item_1,how='left',right_index=True,left_index=True)
    user_item_feature=pd.merge(user_item_feature, user_item_repeattime,how='left',right_index=True,left_index=True)
    pd.set_option('mode.use_inf_as_na', True)
    user_item_feature=user_item_feature.fillna(0)
    return user_item_feature
 
#构建特征user行为次数、转化率
def user_feature(data,end_date):
    daybefore_1=data[data['day']==end_date- datetime.timedelta(days=1)]
    user_1=pd.crosstab([daybefore_1.user_id],daybefore_1.behavior_type)
    user_browse_buy_1=user_1[4]/user_1[1]
    user_collect_buy_1=user_1[4]/user_1[2]
    user_cart_buy_1=user_1[4]/user_1[3]
    user_1['user_browse_buy_1']=user_browse_buy_1
    user_1['user_collect_buy_1']=user_collect_buy_1
    user_1['user_cart_buy_1']=user_cart_buy_1
    daybefore_2=data[(data['day']<=end_date- datetime.timedelta(days=1))&(data['day']>=end_date- datetime.timedelta(days=2))]
    user_2=pd.crosstab([daybefore_2.user_id],daybefore_2.behavior_type)
    user_browse_buy_2=user_2[4]/user_2[1]
    user_collect_buy_2=user_2[4]/user_2[2]
    user_cart_buy_2=user_2[4]/user_2[3] 
    user_2['user_browse_buy_2']=user_browse_buy_2
    user_2['user_collect_buy_2']=user_collect_buy_2
    user_2['user_cart_buy_2']=user_cart_buy_2
    daybefore_3=data[(data['day']<=end_date- datetime.timedelta(days=1))&(data['day']>=end_date- datetime.timedelta(days=3))]
    user_3=pd.crosstab([daybefore_3.user_id],daybefore_3.behavior_type)
    user_browse_buy_3=user_3[4]/user_3[1]
    user_collect_buy_3=user_3[4]/user_3[2]
    user_cart_buy_3=user_3[4]/user_3[3]
    user_3['user_browse_buy_3']=user_browse_buy_3
    user_3['user_collect_buy_3']=user_collect_buy_3
    user_3['user_cart_buy_3']=user_cart_buy_3
    daybefore_4=data[(data['day']<=end_date- datetime.timedelta(days=1))&(data['day']>=end_date- datetime.timedelta(days=4))]
    user_4=pd.crosstab([daybefore_4.user_id],daybefore_4.behavior_type)
    user_browse_buy_4=user_4[4]/user_4[1]
    user_collect_buy_4=user_4[4]/user_4[2]
    user_cart_buy_4=user_4[4]/user_4[3]
    user_4['user_browse_buy_4']=user_browse_buy_4
    user_4['user_collect_buy_4']=user_collect_buy_4
    user_4['user_cart_buy_4']=user_cart_buy_4
    daybefore_5=data[(data['day']<=end_date- datetime.timedelta(days=1))&(data['day']>=end_date- datetime.timedelta(days=5))]
    user_5=pd.crosstab([daybefore_5.user_id],daybefore_5.behavior_type)
    user_browse_buy_5=user_5[4]/user_5[1]
    user_collect_buy_5=user_5[4]/user_5[2]
    user_cart_buy_5=user_5[4]/user_5[3]
    user_5['user_browse_buy_5']=user_browse_buy_5
    user_5['user_collect_buy_5']=user_collect_buy_5
    user_5['user_cart_buy_5']=user_cart_buy_5
    user_feature=pd.merge(user_5, user_4,how='left',right_index=True,left_index=True)
    user_feature=pd.merge(user_feature, user_3,how='left',right_index=True,left_index=True)
    user_feature=pd.merge(user_feature, user_2,how='left',right_index=True,left_index=True)
    user_feature=pd.merge(user_feature, user_1,how='left',right_index=True,left_index=True)
    pd.set_option('mode.use_inf_as_na', True)
    user_feature=user_feature.fillna(0)
    return user_feature
    
#构建特征item行为次数、转化率    
def item_feature(data,end_date):
    daybefore_1=data[data['day']==end_date- datetime.timedelta(days=1)]
    item_1=pd.crosstab([daybefore_1.item_id],daybefore_1.behavior_type)
    item_browse_buy_1=item_1[4]/item_1[1]
    item_collect_buy_1=item_1[4]/item_1[2]
    item_cart_buy_1=item_1[4]/item_1[3]
    item_1['item_browse_buy_1']=item_browse_buy_1
    item_1['item_collect_buy_1']=item_collect_buy_1
    item_1['item_cart_buy_1']=item_cart_buy_1
    daybefore_2=data[(data['day']<=end_date- datetime.timedelta(days=1))&(data['day']>=end_date- datetime.timedelta(days=2))]
    item_2=pd.crosstab([daybefore_2.item_id],daybefore_2.behavior_type)
    item_browse_buy_2=item_2[4]/item_2[1]
    item_collect_buy_2=item_2[4]/item_2[2]
    item_cart_buy_2=item_2[4]/item_2[3]
    item_2['item_browse_buy_2']=item_browse_buy_2
    item_2['item_collect_buy_2']=item_collect_buy_2
    item_2['item_cart_buy_2']=item_cart_buy_2
    daybefore_3=data[(data['day']<=end_date- datetime.timedelta(days=1))&(data['day']>=end_date- datetime.timedelta(days=3))]
    item_3=pd.crosstab([daybefore_3.item_id],daybefore_3.behavior_type)
    item_browse_buy_3=item_3[4]/item_3[1]
    item_collect_buy_3=item_3[4]/item_3[2]
    item_cart_buy_3=item_3[4]/item_3[3]
    item_3['item_browse_buy_3']=item_browse_buy_3
    item_3['item_collect_buy_3']=item_collect_buy_3
    item_3['item_cart_buy_3']=item_cart_buy_3
    daybefore_4=data[(data['day']<=end_date- datetime.timedelta(days=1))&(data['day']>=end_date- datetime.timedelta(days=4))]
    item_4=pd.crosstab([daybefore_4.item_id],daybefore_4.behavior_type)
    item_browse_buy_4=item_4[4]/item_4[1]
    item_collect_buy_4=item_4[4]/item_4[2]
    item_cart_buy_4=item_4[4]/item_4[3]
    item_4['item_browse_buy_4']=item_browse_buy_4
    item_4['item_collect_buy_4']=item_collect_buy_4
    item_4['item_cart_buy_4']=item_cart_buy_4
    daybefore_5=data[(data['day']<=end_date- datetime.timedelta(days=1))&(data['day']>=end_date- datetime.timedelta(days=5))]
    item_5=pd.crosstab([daybefore_5.item_id],daybefore_5.behavior_type)
    item_browse_buy_5=item_5[4]/item_5[1]
    item_collect_buy_5=item_5[4]/item_5[2]
    item_cart_buy_5=item_5[4]/item_5[3]
    item_5['item_browse_buy_5']=item_browse_buy_5
    item_5['item_collect_buy_5']=item_collect_buy_5
    item_5['item_cart_buy_5']=item_cart_buy_5
    item_feature=pd.merge(item_5, item_4,how='left',right_index=True,left_index=True)
    item_feature=pd.merge(item_feature, item_3,how='left',right_index=True,left_index=True)
    item_feature=pd.merge(item_feature, item_2,how='left',right_index=True,left_index=True)
    item_feature=pd.merge(item_feature, item_1,how='left',right_index=True,left_index=True)
    pd.set_option('mode.use_inf_as_na', True)
    item_feature=item_feature.fillna(0)
    return item_feature  
    
 #构建特征user-category行为次数 
def user_category_feature(data,end_date):
    daybefore_1=data[data['day']==end_date- datetime.timedelta(days=1)]
    user_category_1=pd.crosstab([daybefore_1['user_id'],daybefore_1['item_category']],daybefore_1.behavior_type)
    daybefore_2=data[(data['day']<=end_date- datetime.timedelta(days=1))&(data['day']>=end_date- datetime.timedelta(days=2))]
    user_category_2=pd.crosstab([daybefore_2['user_id'],daybefore_2['item_category']],daybefore_2.behavior_type)
    daybefore_3=data[(data['day']<=end_date- datetime.timedelta(days=1))&(data['day']>=end_date- datetime.timedelta(days=3))]
    user_category_3=pd.crosstab([daybefore_3['user_id'],daybefore_3['item_category']],daybefore_3.behavior_type)
    daybefore_4=data[(data['day']<=end_date- datetime.timedelta(days=1))&(data['day']>=end_date- datetime.timedelta(days=4))]
    user_category_4=pd.crosstab([daybefore_4['user_id'],daybefore_4['item_category']],daybefore_4.behavior_type)
    daybefore_5=data[(data['day']<=end_date- datetime.timedelta(days=1))&(data['day']>=end_date- datetime.timedelta(days=5))]
    user_category_5=pd.crosstab([daybefore_5['user_id'],daybefore_5['item_category']],daybefore_5.behavior_type)
    user_category_feature=pd.merge(user_category_5, user_category_4,how='left',right_index=True,left_index=True) 
    user_category_feature=pd.merge(user_category_feature, user_category_3,how='left',right_index=True,left_index=True) 
    user_category_feature=pd.merge(user_category_feature, user_category_2,how='left',right_index=True,left_index=True)
    user_category_feature=pd.merge(user_category_feature, user_category_1,how='left',right_index=True,left_index=True)
    pd.set_option('mode.use_inf_as_na', True)
    user_category_feature=user_category_feature.fillna(0)
    return user_category_feature
    
if __name__ == '__main__':
import datetime
data['buy_or_not']=data['behavior_type']//4 
#构建训练集
datelist=list(set(pd.date_range('2014-11-22', freq = 'D', periods =20))|set(pd.date_range('2014-12-13', freq = 'D', periods =5)))
train_data=pd.DataFrame()
for i in range(25):
    end_date=datelist[i]
    a=data[['user_id','item_id','item_category','buy_or_not']][data['day']==end_date].drop_duplicates()
    user_item_feature_result=user_item_feature(data,end_date)
    user_feature_result=user_feature(data,end_date)
    item_feature_result=item_feature(data,end_date)
    user_category_feature_result=user_category_feature(data,end_date)
    a=pd.merge(a,user_item_feature_result,how='left',on=['user_id','item_id'],sort=False)
    a=pd.merge(a,user_feature_result,how='left',on='user_id',sort=False)
    a=pd.merge(a,item_feature_result,how='left',on='item_id',sort=False)
    a=pd.merge(a,user_category_feature_result,how='left',on=['user_id','item_category'],sort=False)
    a=a.fillna(0)
    train_data=train_data.append(a)
train_data.to_csv('train.csv',index=False)
#构建测试集
test_data=data[['user_id','item_id','item_category','buy_or_not']][data['day']=='2014-12-18'].drop_duplicates()
test_user_item_feature_result=user_item_feature(data,end_date=datetime.datetime.strptime('2014-12-18','%Y-%m-%d'))
test_user_feature_result=user_item_feature(data,end_date=datetime.datetime.strptime('2014-12-18','%Y-%m-%d'))
test_item_feature_result=item_feature(data,end_date=datetime.datetime.strptime('2014-12-18','%Y-%m-%d'))
test_user_category_feature_result=user_category_feature(data,end_date=datetime.datetime.strptime('2014-12-18','%Y-%m-%d'))    
test_data=pd.merge(test_data,test_user_item_feature_result,how='left',on=['user_id','item_id'],sort=False)
test_data=pd.merge(test_data,test_user_feature_result,how='left',on='user_id',sort=False)
test_data=pd.merge(test_data,test_item_feature_result,how='left',on='item_id',sort=False)
test_data=pd.merge(test_data,test_user_category_feature_result,how='left',on=['user_id','item_category'],sort=False)
test_data=test_data.fillna(0)
test_data.to_csv('test.csv',index=False)
#构建需要预测12-19的特征集
pred_X=data[['user_id','item_id','item_category']][data['behavior_type']==4].drop_duplicates()
pred_user_item_feature_result=user_item_feature(data,end_date=datetime.datetime.strptime('2014-12-19','%Y-%m-%d'))
pred_item_feature_result=item_feature(data,end_date=datetime.datetime.strptime('2014-12-19','%Y-%m-%d'))
pred_user_category_feature_result=user_category_feature(data,end_date=datetime.datetime.strptime('2014-12-19','%Y-%m-%d'))
pred_user_feature_result=user_item_feature(data,end_date=datetime.datetime.strptime('2014-12-19','%Y-%m-%d'))
pred_X=pd.merge(pred_X,pred_user_item_feature_result,how='left',on=['user_id','item_id'],sort=False)
pred_X=pd.merge(pred_X,pred_user_feature_result,how='left',on='user_id',sort=False)
pred_X=pd.merge(pred_X,pred_item_feature_result,how='left',on='item_id',sort=False)
pred_X=pd.merge(pred_X,pred_user_category_feature_result,how='left',on=['user_id','item_category'],sort=False)
pred_X=pred_X.fillna(0)
pred_X.to_csv('pred.csv',index=False)

from sklearn.metrics import f1_score
from xgboost.sklearn import XGBClassifier
train_data=pd.read_csv(r'C:\train.csv')
test_data=pd.read_csv(r'C:\test.csv')
train_x=train_data.drop(columns=['buy_or_not','item_id','user_id','item_category'])
train_y=train_data['buy_or_not']
test_x=test_data.drop(columns=['buy_or_not','item_id','user_id','item_category'])
y_true=test_data['buy_or_not']
model=XGBClassifier(scale_pos_weight=10)
model.fit(train_x.values,train_y.values.ravel())
y_pred = model.predict(test_x.values)
f1_score(y_true, y_pred, average=None)
pred_x=pd.read_csv(r'C:\pred.csv')
pred_X=pred_x.drop(columns=['item_id','user_id','item_category'])
pred_X=pred_X.fillna(0)
pred_Y = model.predict(pred_X.values)
pred_Y=pd.DataFrame(list(pred_Y))
pred=pred_x[['user_id','item_id']].astype(str)
pred=pd.concat([pred,pred_Y],axis=1)
pred.columns=['user_id','item_id','buy_or_not']
pred=pred[pred['buy_or_not']==1]
pred.to_csv('tianchi_mobile_recommendation_predict.csv',index=False)
