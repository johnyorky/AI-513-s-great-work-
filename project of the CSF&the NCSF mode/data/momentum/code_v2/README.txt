生成NCSF mode series和random series：
步骤：
1.config.py设置生成数据参数:设置config.py中的 windows长度（一般不变）
2.运行0fake_random_momentum.py生成原始NCSF mode series和random series：momentum和random文件
3.运行1get_train_test.py将生成的momentum和random文件处理成可输入模型的格式，以便进行训练和测试
4.运行2fake_test.py，用ground_truth模型测试数据的结果是否符合要求，如果不符合，则修改参数，再次从1开始运行。




