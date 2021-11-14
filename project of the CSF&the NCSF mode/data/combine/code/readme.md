1)设置config.py中的 windows长度（一般不变）
2) 运行 generate_weigh_config 为不同的组合特征随机设置权重，生成weight_config
3) 设置config.py中的score_thred，运行 0fake_data.py  生成combine和random
   生成的combine数据要真实，如果不真实，就调整一下score_thred
4) 运行1get_train_test.py 得到xy_train_combine,xy_train_random
   其中，每个数据中的涨跌比例要相当，如果不相当就要重新返回3）


#先调整参数，生成比较真实的具有组合特征的数据