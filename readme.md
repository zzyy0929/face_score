一、准备工作：

1、下载dlib库，下载特征提取模型。

该模型的作用是通过卷积神经网络产生128维的特征向量，用以代表这张脸。网络输入参数为人脸landmark的68个特征点shape和整幅图像。可猜想网络特征与人脸的68特征点坐标有关，在网络中进行归一化并进一步处理，使得提出的特征具有独立、唯一性。

考虑到人脸的颜值与五官位置，拍照时的表情有关，故本网络可作为一种方案进行尝试。

Dlib下载：

http://dlib.net/

本模型原用于人脸识别，原型为CNN_ResNet。残差网络是为了减弱在训练过程中随着网络层数增加而带来的梯度弥散/爆炸的问题。该方法在LFW上进行人脸识别达到99.38%的准确率。

模型名称：dlib_face_recognition_resnet_model_v1，迭代次数为10000，训练时用了约300万的图片。输入层的图片尺寸是150。

下载地址：

提取特征的网络模型地址：

http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2

landmark 68特征点位置提取模型：

http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2

2、数据准备：准备不同类型的脸部图像，注意选用颜值不同的照片，该部分具有一定的主观性，也是对最后评分影响最重要的一个环节，所以数据量应尽可能大，选用的图像尽可能典型。

我们设置6个分数，分别为：95,90,85,80,70,65

95分人数仅2人，其余分数在15人左右。85人最多，约20人。数据符合正态分布。
