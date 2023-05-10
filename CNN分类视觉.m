
%这里有两个数据，left_hand_data, feet_data，是两种不同的脑电信号
%下面是一个浅层卷积神经网络分类的代码
%浅层卷积神经网络
    % 数据预处理
    numElectrodes = 126;
    timePoints = 2000;
    trials = 40;
    % 将两类数据合并
    allData = cat(3, left_hand_data, feet_data);
    % 创建标签
    labels = [ones(trials, 1); 2 * ones(trials, 1)];
    % 划分数据集为训练集和测试集
    trainRatio = 0.8;
    numTrain = floor(trials * trainRatio);%40*0.8=32
    numTest = trials - numTrain;%=8

    trainIdx = [1:numTrain, trials + 1:trials + numTrain];%选了64个数据
    testIdx = [numTrain + 1:trials, 2 * trials - numTest + 1:2 * trials];%剩下的16个数据

    trainData = allData(:, :, trainIdx);%训练数据
    trainLabels = labels(trainIdx, :);%训练label
    testData = allData(:, :, testIdx);%测试数据
    testLabels = labels(testIdx, :);%测试label

    % 将数据转换为图像格式
    trainData = reshape(trainData, [numElectrodes, timePoints, 1, 2 * numTrain]);
    testData = reshape(testData, [numElectrodes, timePoints, 1, 2 * numTest]);

    % 创建浅层卷积神经网络
    layers = [
              imageInputLayer([numElectrodes timePoints 1])
              convolution2dLayer([numElectrodes 25], 8, 'Padding', 'same')
              batchNormalizationLayer
              reluLayer
              maxPooling2dLayer([1 4], 'Stride', [1 4])
              fullyConnectedLayer(2)
              softmaxLayer
              classificationLayer
              ];

    % 设置训练选项
    options = trainingOptions('sgdm', ...%优化器
        'InitialLearnRate', 0.01, ... %初始的学习率
        'MaxEpochs', 30, ...%迭代次数
        'Shuffle', 'every-epoch', ...%在每个 epoch 开始时对数据进行随机打乱
        'MiniBatchSize', 32, ... % 可以减小 MiniBatchSize减轻计算负担
        'ValidationData', {testData, categorical(testLabels)}, ...%指定用于网络验证的数据集
        'ValidationFrequency', 20, ...% 30 个迭代验证一次模型
        'Verbose', false, ...%是否在命令行窗口中显示训练进度和其他信息
        'ExecutionEnvironment', 'cpu', ...%使用CPU训练
        'Plots', 'training-progress');%显示训练进度

    % 训练网络
    net = trainNetwork(trainData, categorical(trainLabels), layers, options);

    % 评估网络性能
    predictedLabels = classify(net, testData, 'ExecutionEnvironment', 'cpu');
    accuracy = sum(predictedLabels == categorical(testLabels)) / (2*numTest);
    disp(['准确率为：', num2str(accuracy)]);

    %如果是普通的CNN，用以下的代码
    layers = [imageInputLayer([numElectrodes timePoints 1]) % 输入层，接受大小为 [numElectrodes timePoints 1] 的图像作为输入
        convolution2dLayer([numElectrodes 25], 16, 'Padding', 'same') % 卷积层1，16个大小为 [numElectrodes 25] 的卷积核
         batchNormalizationLayer % 批标准化层，对每一批输入数据进行标准化，加速网络的收敛
         reluLayer % ReLU激活函数层，对输入数据进行非线性变换
         maxPooling2dLayer([1 4], 'Stride', [1 4]) % 最大池化层，每个电极上的时间点划分为大小为 [1 4] 的池化区域

        convolution2dLayer(3, 32, 'Padding', 'same') % 卷积层2，32个大小为 3 的卷积核
        batchNormalizationLayer % 批标准化层
        reluLayer % ReLU激活函数层
        maxPooling2dLayer(2, 'Stride', 2) % 最大池化层，大小为 2 的池化区域

        convolution2dLayer(3, 64, 'Padding', 'same') % 卷积层3，64个大小为 3 的卷积核
        batchNormalizationLayer % 批标准化层
        reluLayer % ReLU激活函数层
        maxPooling2dLayer(2, 'Stride', 2) % 最大池化层，大小为 2 的池化区域

        fullyConnectedLayer(128) % 全连接层，有 128 个输出
        reluLayer % ReLU激活函数层
        dropoutLayer(0.5) % dropout层，以 0.5 的概率随机失活输入神经元，防止过拟合

        fullyConnectedLayer(2) % 全连接层，有 2 个输出
        softmaxLayer % softmax层，将全连接层的输出转换为概率值
        classificationLayer % 分类层，将 softmax 层的输出与标签进行比较，计算模型的分类误差
        ];


