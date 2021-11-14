import config as F
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
#:import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from hmmlearn.hmm import GaussianHMM
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import  AdaBoostClassifier
#import ground_truth as G
import pickle
import glob
data_train=F.data_dir_train
data_test=F.data_dir_test
groud_truth=F.data_ground_truth
model_list=F.model_list
strim=F.strim
import xlwt
import time

title_test=["模型名称","测试集总数","测试集中上涨的个数","测试集中上涨比例",\
"测试集中预测的上涨个数","真实上涨个数","测试集中预测上涨比例",\
"测试集中预测下跌的个数","真实下跌个数","测试集中预测下跌比例",\
"测试集预测正确数","测试集预测正确率","训练时间"]

title_train=["模型名称","训练集总数","训练集中上涨的个数","训练集中上涨比例",\
"训练集中预测的上涨个数","真实上涨个数","训练集中预测上涨比例",\
"训练集中预测下跌的个数","真实下跌个数","训练集中预测下跌比例",\
"训练集预测正确数","训练集预测正确率","训练时间"]
def write(data,name):
    f = xlwt.Workbook()
    #创建sheet1
    sheet_test = f.add_sheet('test',cell_overwrite_ok=True)
    for ll,title in enumerate(title_test):
        sheet_test.write(0,ll,title)
    sheet_train = f.add_sheet('train',cell_overwrite_ok=True)
    for ll,title in enumerate(title_train):
        sheet_train.write(0,ll,title)
    line_number=1
    for model,content in data.items():
        sheet_test.write(line_number,0,model)
        sheet_train.write(line_number,0,model)
        total,up_total,pre_up,pre_up_right,pre_down,pre_down_right,t1=content["train"]
        to_wirte=[total,up_total,up_total/total,pre_up,pre_up_right,pre_up_right/pre_up,pre_down,pre_down_right,\
        pre_down_right/pre_down,pre_up_right+pre_down_right,(pre_up_right+pre_down_right)/total,t1]
        for ii,co in enumerate(to_wirte):
            sheet_train.write(line_number,ii+1,co)
        total,up_total,pre_up,pre_up_right,pre_down,pre_down_right,t1=content["test"]
        to_wirte=[total,up_total,up_total/total,pre_up,pre_up_right,pre_up_right/pre_up,pre_down,pre_down_right,\
        pre_down_right/pre_down,pre_up_right+pre_down_right,(pre_up_right+pre_down_right)/total,t1]
        for ii,co in enumerate(to_wirte):
            sheet_test.write(line_number,ii+1,co)
        line_number +=1
    f.save(name)
def load_data_train(train_path,strim=-1):
    x=[]
    y=[]
    if strim ==-1:
        for line in open(train_path,"r",encoding="utf-8"):
            x_list,label=line.strip().split("\t")
            x_list=[float(xx) for xx in x_list.split("<=>")]
            x.append(x_list)
            y.append(label)
        return x,y
    else:
        for line in open(train_path,"r",encoding="utf-8"):
            x_list,label=line.strip().split("\t")
            x_list=[float(xx) for xx in x_list.split("<=>")]
            x_list=x_list[:strim] if len(x_list) >strim else x_list + [-1]*(strim-len(x_list))
            x.append(x_list)
            y.append(label)
        return x,y

def load_data_test(test_path,strim=-1):
    x_test=[]
    y_test=[]
    if strim ==-1:
        for line in open(test_path,"r",encoding="utf-8"):
            x_test_list,label=line.strip().split("\t")
            x_test_list=[float(xx_test) for xx_test in x_test_list.split("<=>")]
           # x_list=x_list[:strim] if len(x_list) >strim else x_list + [-1]*(strim-len(x_list))
            x_test.append(x_test_list)
            y_test.append(label)
        return x_test,y_test
    else:
        for line in open(test_path,"r",encoding="utf-8"):
            x_test_list,label=line.strip().split("\t")
            x_test_list=[float(xx_test) for xx_test in x_test_list.split("<=>")]
            x_test_list=x_test_list[:strim] if len(x_test_list) >strim else x_test_list + [-1]*(strim-len(x_test_list))
            x_test.append(x_test_list)
            y_test.append(label)
        return x_test,y_test

def RandomForest(x,y,x_test,y_test):
    st=time.time()
    Model = RandomForestClassifier(max_depth=10, random_state=0)
    Model.fit(x,y)
    print("trained for %s seconds"%(time.time()-st))
    t1 = time.time()-st
    y_predict=Model.predict(x)
    y_test_predict=Model.predict(x_test)
    return y_predict,y_test_predict
    
    
def SupportVectorMachine(x,y,x_test,y_test):
    st=time.time()
    Model = SVC(verbose=True)
    Model.fit(x,y)
    print("trained for %s seconds"%(time.time()-st))
    t1 = time.time()-st
    y_predict=Model.predict(x)
    y_test_predict=Model.predict(x_test)
    return y_predict,y_test_predict

def BayesClassifier(x,y,x_test,y_test):
    st=time.time()
    Model = GaussianNB()#alpha拉普拉斯平滑系数，防止概率为0出现
    Model.fit(x,y)
    print("trained for %s seconds"%(time.time()-st))
    t1 = time.time()-st
    y_predict = Model.predict(x)
    y_test_predict=Model.predict(x_test)
    return y_predict,y_test_predict
    

def DecisionTree(x,y,x_test,y_test):
    st=time.time()
    Model = tree.DecisionTreeClassifier()  #实例化，全部选择了默认的参数
    Model.fit(x,y)               #拟合
    print("trained for %s seconds"%(time.time()-st))
    t1 = time.time()-st
    y_predict=Model.predict(x)  
    y_test_predict=Model.predict(x_test)
    return y_predict,y_test_predict

def KNN(x,y,x_test,y_test):
    st=time.time()
    Model = KNeighborsClassifier()
    Model.fit(x,y)              #拟合
    print("trained for %s seconds"%(time.time()-st))
    t1 = time.time()-st
    y_predict=Model.predict(x)  
    y_test_predict=Model.predict(x_test)
    return y_predict,y_test_predict
    
def Multi_LayerPerceptron(x,y,x_test,y_test):
    st=time.time()
    Model = MLPClassifier(hidden_layer_sizes=(30, 30, 30), max_iter=500)
    Model.fit(x,y)
    print("trained for %s seconds"%(time.time()-st))
    t1 = time.time()-st
    y_predict=Model.predict(x)  
    y_test_predict=Model.predict(x_test)
    return y_predict,y_test_predict
    
def Logistic_Regression(x,y,x_test,y_test):
    st=time.time()
    Model = LogisticRegression(penalty = 'l2')
    Model.fit(x,y)
    print("trained for %s seconds"%(time.time()-st))
    t1 = time.time()-st
    y_predict=Model.predict(x)  
    y_test_predict=Model.predict(x_test)
    return y_predict,y_test_predict
    
def LinearDiscriminant(x,y,x_test,y_test):
    st=time.time()
    Model = LinearDiscriminantAnalysis()
    Model.fit(x,y)
    print("trained for %s seconds"%(time.time()-st))
    t1 = time.time()-st
    y_predict=Model.predict(x)  
    y_test_predict=Model.predict(x_test)
    return y_predict,y_test_predict
    
def GBDT(x,y,x_test,y_test):
    st=time.time()
    Model = GradientBoostingClassifier(n_estimators=200)
    Model.fit(x,y)
    print("trained for %s seconds"%(time.time()-st))
    t1 = time.time()-st
    y_predict=Model.predict(x)  
    y_test_predict=Model.predict(x_test)
    return y_predict,y_test_predict

def QDA(x,y,x_test,y_test):
    st=time.time()
    Model = QuadraticDiscriminantAnalysis()
    Model.fit(x,y)
    print("trained for %s seconds"%(time.time()-st))
    t1 = time.time()-st
    y_predict=Model.predict(x)  
    y_test_predict=Model.predict(x_test)
    return y_predict,y_test_predict
    
def Ada(x,y,x_test,y_test):
    st=time.time()
    Model = AdaBoostClassifier()
    Model.fit(x, y)
    print("trained for %s seconds"%(time.time()-st))
    t1 = time.time()-st
    y_predict=Model.predict(x)  
    y_test_predict=Model.predict(x_test)
    return y_predict,y_test_predict
    
def test(y_predict,y,f):
    right_up=0
    right_down=0
    up_total=1
    down_total=0
    total=1
    right=0
    right_total_give=1
    pre_down_total=1
    pre_up_total=1
    for pre,rea in zip(y_predict,y):
            #print(pre,rea,rea==pre)
        if pre=="down":
            pre_down_total +=1
            if pre==rea:
                right_down +=1

        if rea=="down":
            down_total +=1

        if pre=="up":
            pre_up_total +=1
            if pre==rea:
                right_up +=1
            
            
        if rea =="up":
            up_total +=1
            
        if pre !="no":
            right_total_give +=1
            if pre==rea:
                right +=1
        total +=1
    
    
    f.write("训练数据中有%s个数，涨的有%s，比例是%s\n"%(total,up_total,up_total/total))
    f.write("训练数据中预测涨的有%s个数，真实涨的有%s，比例是%s\n"%(pre_up_total,right_up,right_up/pre_up_total))
    f.write("--------------------------------------\n")
    f.write("训练数据中有%s个数，跌的有%s，比例是%s\n"%(total,down_total,down_total/total))
    f.write("训练数据中预测跌的有%s个数，真实跌的有%s，比例是%s\n"%(pre_down_total,right_down,right_down/pre_down_total))
    f.write("--------------------------------------\n")
    f.write("预测正确的数量是%s，给出预测有%s，准确率是%s\n"%(right,right_total_give,right/right_total_give))
    #plt.plot(y_predict)
    #plt.show()
    return total,up_total,pre_up_total,right_up,pre_down_total,right_down
def test2(y_test_predict,y_test,f):
    right_up1=0
    right_down1=0
    up_total1=0
    down_total1=0
    total1=1
    right1=0
    right_total_give1=1
    pre_down_total1=1
    pre_up_total1=1
    for pre1,rea1 in zip(y_test_predict,y_test):
            #print(pre,rea,rea==pre)
        if pre1=="down":
            pre_down_total1 +=1
            if pre1==rea1:
                right_down1 +=1

        if rea1=="down":
            down_total1 +=1

        if pre1=="up":
            pre_up_total1 +=1
            if pre1==rea1:
                right_up1 +=1
            
            
        if rea1 =="up":
            up_total1 +=1
            
        if pre1 !="no":
            right_total_give1 +=1
            if pre1==rea1:
                right1 +=1
        total1 +=1
    
   
    f.write("测试数据中有%s个数，涨的有%s，比例是%s\n"%(total1,up_total1,up_total1/total1))
    f.write("测试数据中预测涨的有%s个数，真实涨的有%s，比例是%s\n"%(pre_up_total1,right_up1,right_up1/pre_up_total1))
    f.write("--------------------------------------\n")
    f.write("测试数据中有%s个数，跌的有%s，比例是%s\n"%(total1,down_total1,down_total1/total1))
    f.write("测试数据中预测跌的有%s个数，真实跌的有%s，比例是%s\n"%(pre_down_total1,right_down1,right_down1/pre_down_total1))
    f.write("--------------------------------------\n")
    f.write("预测正确的数量是%s，给出预测有%s，准确率是%s\n"%(right1,right_total_give1,right1/right_total_give1))
   # plt.plot(y_test_predict)
    #plt.show()
    return total1,up_total1,pre_up_total1,right_up1,pre_down_total1,right_down1
 


def get_num(string):
    return int(string.strip().split(" ")[-1])

def one(train_path,test_path):
    x, y = load_data_train(train_path)
    x_test,y_test = load_data_test(test_path)
    size=len(x)
    st=time.time()  
    data={}
    num=size#train_path.split("_")[-1]
    with open("result/%s_TRAIN_result"%num,"w",encoding="utf-8")as f,open("result/%s_TEST_result"%num,"w",encoding="utf-8")as f_test:
        for model in model_list:
            print(model)
            if model=="RandomForest":
                f.write("result of train model %s\n"%model)
                y_predict,y_test_predict=RandomForest(x,y,x_test,y_test)
                t1=0
                total,up_total,pre_up,pre_up_right,pre_down,pre_down_right=test(y_predict,y,f) 
                data["RandomForest"]={"train":[total,up_total,pre_up,pre_up_right,pre_down,pre_down_right,t1]}
                total,up_total,pre_up,pre_up_right,pre_down,pre_down_right=test2(y_test_predict,y_test,f_test)
                data["RandomForest"]["test"]=[total,up_total,pre_up,pre_up_right,pre_down,pre_down_right,t1]
            elif model =="SupportVectorMachine":
                f.write("result of train model %s\n"%model)
                y_predict,y_test_predict=SupportVectorMachine(x,y,x_test,y_test)
                t1 = time.time()-st
                total,up_total,pre_up,pre_up_right,pre_down,pre_down_right=test(y_predict,y,f)
                data["SupportVectorMachine"]={"train":[total,up_total,pre_up,pre_up_right,pre_down,pre_down_right,t1]}
                total,up_total,pre_up,pre_up_right,pre_down,pre_down_right=test2(y_test_predict,y_test,f_test)
                data["SupportVectorMachine"]["test"]=[total,up_total,pre_up,pre_up_right,pre_down,pre_down_right,t1]
            elif model=="BayesClassifier":
                f.write("result of train model %s\n"%model)
                y_predict,y_test_predict=BayesClassifier(x,y,x_test,y_test)
                t1 = time.time()-st
                total,up_total,pre_up,pre_up_right,pre_down,pre_down_right=test(y_predict,y,f)
                data["BayesClassifier"]={"train":[total,up_total,pre_up,pre_up_right,pre_down,pre_down_right,t1]}
                total,up_total,pre_up,pre_up_right,pre_down,pre_down_right=test2(y_test_predict,y_test,f_test)
                data["BayesClassifier"]["test"]=[total,up_total,pre_up,pre_up_right,pre_down,pre_down_right,t1]
            elif model=="DecisionTree":
                f.write("result of train model %s\n"%model)
                y_predict,y_test_predict=DecisionTree(x,y,x_test,y_test)
                t1 = time.time()-st
                total,up_total,pre_up,pre_up_right,pre_down,pre_down_right=test(y_predict,y,f)
                data["DecisionTree"]={"train":[total,up_total,pre_up,pre_up_right,pre_down,pre_down_right,t1]}
                total,up_total,pre_up,pre_up_right,pre_down,pre_down_right=test2(y_test_predict,y_test,f_test)
                data["DecisionTree"]["test"]=[total,up_total,pre_up,pre_up_right,pre_down,pre_down_right,t1]
            elif model=="KNN":
                f.write("result of train model %s\n"%model)
                y_predict,y_test_predict=KNN(x,y,x_test,y_test)
                t1 = time.time()-st
                total,up_total,pre_up,pre_up_right,pre_down,pre_down_right=test(y_predict,y,f)
                data["KNN"]={"train":[total,up_total,pre_up,pre_up_right,pre_down,pre_down_right,t1]}
                total,up_total,pre_up,pre_up_right,pre_down,pre_down_right=test2(y_test_predict,y_test,f_test)
                data["KNN"]["test"]=[total,up_total,pre_up,pre_up_right,pre_down,pre_down_right,t1]
            elif model=="Multi_LayerPerceptron":
                f.write("result of train model %s\n"%model)
                y_predict,y_test_predict=Multi_LayerPerceptron(x,y,x_test,y_test)
                t1 = time.time()-st
                total,up_total,pre_up,pre_up_right,pre_down,pre_down_right=test(y_predict,y,f)
                data["Multi_LayerPerceptron"]={"train":[total,up_total,pre_up,pre_up_right,pre_down,pre_down_right,t1]}
                total,up_total,pre_up,pre_up_right,pre_down,pre_down_right=test2(y_test_predict,y_test,f_test)
                data["Multi_LayerPerceptron"]["test"]=[total,up_total,pre_up,pre_up_right,pre_down,pre_down_right,t1]
            elif model=="Logistic_Regression":
                f.write("result of train model %s\n"%model)
                y_predict,y_test_predict=Logistic_Regression(x,y,x_test,y_test)
                t1 = time.time()-st
                total,up_total,pre_up,pre_up_right,pre_down,pre_down_right=test(y_predict,y,f)
                data["Logistic_Regression"]={"train":[total,up_total,pre_up,pre_up_right,pre_down,pre_down_right,t1]}
                total,up_total,pre_up,pre_up_right,pre_down,pre_down_right=test2(y_test_predict,y_test,f_test)
                data["Logistic_Regression"]["test"]=[total,up_total,pre_up,pre_up_right,pre_down,pre_down_right,t1]
            elif model=="LinearDiscriminant":
                f.write("result of train model %s\n"%model)
                y_predict,y_test_predict=LinearDiscriminant(x,y,x_test,y_test)
                t1 = time.time()-st
                total,up_total,pre_up,pre_up_right,pre_down,pre_down_right=test(y_predict,y,f)
                data["LinearDiscriminant"]={"train":[total,up_total,pre_up,pre_up_right,pre_down,pre_down_right,t1]}
                total,up_total,pre_up,pre_up_right,pre_down,pre_down_right=test2(y_test_predict,y_test,f_test)
                data["LinearDiscriminant"]["test"]=[total,up_total,pre_up,pre_up_right,pre_down,pre_down_right,t1]
            elif model=="GBDT":
                f.write("result of train model %s\n"%model)
                y_predict,y_test_predict=GBDT(x,y,x_test,y_test)
                t1 = time.time()-st
                total,up_total,pre_up,pre_up_right,pre_down,pre_down_right=test(y_predict,y,f)
                data["GBDT"]={"train":[total,up_total,pre_up,pre_up_right,pre_down,pre_down_right,t1]}
                total,up_total,pre_up,pre_up_right,pre_down,pre_down_right=test2(y_test_predict,y_test,f_test)
                data["GBDT"]["test"]=[total,up_total,pre_up,pre_up_right,pre_down,pre_down_right,t1]
            elif model=="QDA":
                f.write("result of train model %s\n"%model)
                y_predict,y_test_predict=QDA(x,y,x_test,y_test)
                t1 = time.time()-st
                total,up_total,pre_up,pre_up_right,pre_down,pre_down_right=test(y_predict,y,f)
                data["QDA"]={"train":[total,up_total,pre_up,pre_up_right,pre_down,pre_down_right,t1]}
                total,up_total,pre_up,pre_up_right,pre_down,pre_down_right=test2(y_test_predict,y_test,f_test)
                data["QDA"]["test"]=[total,up_total,pre_up,pre_up_right,pre_down,pre_down_right,t1]
            elif model=="Ada":
                f.write("result of train model %s\n"%model)
                y_predict,y_test_predict=Ada(x,y,x_test,y_test)
                t1 = time.time()-st
                total,up_total,pre_up,pre_up_right,pre_down,pre_down_right=test(y_predict,y,f)
                data["Ada"]={"train":[total,up_total,pre_up,pre_up_right,pre_down,pre_down_right,t1]}
                total,up_total,pre_up,pre_up_right,pre_down,pre_down_right=test2(y_test_predict,y_test,f_test)
                data["Ada"]["test"]=[total,up_total,pre_up,pre_up_right,pre_down,pre_down_right,t1]
            elif model=="g_t":
                f.write("result of train model %s\n"%model)
                t1 = time.time()-st
                total,up_total,pre_up,pre_up_right,pre_down,pre_down_right=test(y_predict,y,f)
                for line in open(groud_truth,"r",encoding="utf-8"):
                    if line.find("总标签数量")!=-1:
                        total=get_num(line)#(line.strip().split(" ")[-1])
                        continue
                    if line.find("原始标签中 0的绝对数量")!=-1:
                        up_total=get_num(line)#int((line.strip().split(" ")[-1])
                        continue
                    if line.find("预测标签中0的绝对数量")!=-1:
                        pre_up=get_num(line)
                        continue
                    if line.find("预测上涨的正确数量")!=-1:
                        pre_up_right=get_num(line)
                        continue
                    if line.find("预测标签中1的绝对数量")!=-1:
                        pre_down=get_num(line)#(line.strip().split(" ")[-1])
                        continue
                    if line.find("预测下跌的正确数量")!=-1:
                        pre_down_right=get_num(line)#int((line.strip().split(" ")[-1])
                        continue
                        
                
                data["g_t"]={"train":[total,up_total,pre_up,pre_up_right,pre_down,pre_down_right,t1]}
                #total,up_total,pre_up,pre_up_right,pre_down,pre_down_right=test2(y_test_predict,y_test,f_test)
                data["g_t"]["test"]=[total,up_total,pre_up,pre_up_right,pre_down,pre_down_right,t1]
    with open("result/data_result_%s"%size,"wb")as f:
        pickle.dump(data,f)
    write(data,"result/final_result_%s222.xls"%size)
if __name__=="__main__":
    train_path_list=glob.glob(data_train+"/*")
    test_path=glob.glob(data_test+"/*")[0]
    for train in train_path_list:
        print(train)
        one(train,test_path)
