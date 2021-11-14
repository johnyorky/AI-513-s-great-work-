###用于探究股票历史量价数据中的未来收益信息是否包含在形态组合特征中，以及包含了多少：

一，构造模拟金融数据：
‘data文件夹’	用于生成数据：combine===>生成CSF mode series;
                                                      momentum===>生成NCSF mode series;
                                                      random===>生成random series；
                                                      real===>真实金融量价数据。


‘models文件夹‘	用于测试数据的模型：统计模型（SM-CSF）；
				机器学习模型（SVM，RF,DTR,MLP,BAYES CLASSIFIER)；
				深度学习模型(CNN,BiLSTM,Embedded_CNN,Embedded_BiLSTM)。


#######
It is used to explore whether the future return information in the historical stock price data is included in the curve-shape-features and how much:

1. Construct simulated financial data:

"data"folders Used to generate data: combine===> generate CSF mode series;

Momentum===> generate NCSF mode series;

Random===> generate random series;

Real===> real financial volume price data.

"Models"folders Models for testing data: statistical models (SM-CSF);

Machine learning model (SVM, RF, DTR, MLP, BAYES CLASSIFIER);

Deep learning model (CNN, BiLSTM, Embedded CNN, Embedded BiLSTM).
