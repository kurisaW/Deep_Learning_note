
## **神经网络与深度学习期末作业 ------ 基于Tensorflow的手势识别**


**1.采集数据集**

---

- 运行: get_gesture_images.py文件
  可用不用运行，因为样本集已有

- 样本集存放的目录：train_gesture_data

---

**2.训练模型**

---

- 1. 新建目录：
   gesture_recognition_model/gestureModel 和gesture_recognition_model/gestureModel_one
  2. 运行: gesture_recongnition.py文件

  

**3.测试样本的预测**

---

 - 运行 pred_gesture.py文件

**4.conda环境**

在当前目录下名为 `environment.yml` 的文件，包含我项目运行环境的所有包和依赖项信息。可按照以下步骤将项目环境进行迁移复刻：

要迁移 Conda 虚拟环境，您需要执行以下步骤：

1. 在源环境中导出环境配置：在命令行中运行以下命令，将当前环境的配置导出到一个 YAML 文件中：

```
conda env export > environment.yml
```

这将创建一个名为 `environment.yml` 的文件，其中包含当前环境的所有包和依赖项信息。

2. 将导出的环境配置文件传输到目标系统：将生成的 `environment.yml` 文件从源系统传输到目标系统。您可以使用文件传输工具（如 SCP、FTP 或云存储服务）将文件从一台计算机复制到另一台计算机。

3. 在目标系统中创建新的 Conda 虚拟环境：在目标系统中打开命令行，并使用以下命令创建一个新的 Conda 虚拟环境：

```
conda env create -f environment.yml
```

这将使用 `environment.yml` 文件中的配置信息创建一个与源环境相同的新环境。

4. 激活新环境：在目标系统上激活新创建的环境，以便开始使用它。运行以下命令：

```
conda activate <environment_name>
```

将 `<environment_name>` 替换为您为新环境选择的名称。

现在，您已成功迁移 Conda 虚拟环境到目标系统。您可以在新环境中使用相同的包和依赖项，以及在源环境中相同的方式运行代码和应用程序。

---

**注：本项目参考自[https://github.com/qiulc/gesture_recongnition](https://github.com/qiulc/gesture_recongnition)，仅供学习参考**

