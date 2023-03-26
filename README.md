# Deep-Convolutional-Neural-Networks-with-Wide-First-layer-Kernels
这是一个首层卷积为宽卷积的深度神经网络Deep Convolutional Neural Networks with Wide First-layer Kernels (WDCNN)的实现，该模型具有优越的抗噪能力，可用于轴承的智能故障诊断。

通过借鉴该模型，发表科研论文两篇
> [Intelligent Motor Bearing Fault Diagnosis Using Channel Attention-Based CNN](https://www.mdpi.com/2032-6653/13/11/208)

> [Fault diagnosis method for imbalanced bearing data based on W-DCGAN](https://ieeexplore.ieee.org/abstract/document/9862722)
# 模型结构
![Architecture of the proposed WDCNN model](./gitIMGs/WDCNN.png)
# 不同数据量对模型性能的影响
![Diagnosis results using different numbers of training samples](./gitIMGs/samples.png)
# t-SNE可视化
![Feature visualization via t-SNE](./gitIMGs/tsne.png)

![Feature visualization via t-SNE](./gitIMGs/tsne2.png)
# 抗噪性分析
![Results of the proposed WDCNN and WDCNN-AdaBN of six domain shifts on the Datasets A, B and C, compared with FFT-SVM, FFT-MLP and FFT-DNN](./gitIMGs/noise.png)

# 卷积可视化
![Visualization of all convolutional neuron activations in WDCNN](./gitIMGs/vis_feature_map.png)


