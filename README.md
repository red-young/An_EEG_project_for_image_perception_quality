# An_EEG_project_for_image_perception_quality
A project that uses EEG signals to judge the quality of vision

2023/5/11

初次跑通，准确率0.5625

主要原因：

1、使用网络数据集，对质量无法保证

2、该数据集为双变量，分别对应两个条件，但仅跑了30轮，迭代60次，算力严重不足

3、学习率为常数0.01，收敛复杂度较高
