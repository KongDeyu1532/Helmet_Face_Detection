# Helmet_Face_Detection
## 《基于 Yolo V3 和 OpenCV 的安全帽检测和人脸识别系统》
### 项目描述: 本项目基于 OpenCV 使用 Haar 级联与 dlib 库进行人脸检测，同时使用 Yolo V3 网络进行安全帽检测，应用 LBPH 算法开发了一个功能相对完善的人脸检测系统。系统采用 sqlite3 数据库进行序列化数据存储， 能够对陌生人脸以及未佩戴安全帽的的人员进行报警，并拥有基于 PyQt5 设计的 GUI 界面的实现。
#### 负责模块:安全帽图像的训练、人脸检测的实现、多线程问题
##### 技术要点:
- 1.标注:
> 本人使用LabelImg对安全帽图片进行标注，以便在后续神经网络进行学习的时候网络能够知道标注
的矩形区域内为安全帽。标注完成后，软件会为每个图片生成一个 txt 文件，文件中包含矩形区域 4 个点的坐标。
- 2.训练网络:
> 通过代码解析上述生成的txt文件，从大图中根据标签数据把小图抠出来。之后调用shtrain.sh
命令开始训练网络。并且基于 Yolo V3 网络，提出了一种改进方法:改进的非极大值抑制算法。
- 3.人脸识别:
> 利用opencv开源的LBPH算法实现人脸识别功能
- 4.数据库:
> (1)当有新员工时，录入员工的信息，并保存在数据库。(2)在工地出入口匝道处，对员工进行
考勤处理，把相应的信息保存到数据库中。
- 5.多进程同步:
> 在工地出入口匝道处，对员工进行安全帽检测以及人脸识别，需要同时进行。所以本人提出
先进行人脸识别，将识别的结果封装到设计的类中。然后安全帽检测进程从这个类中取出该员工信息，这
样可以保证数据库一条记录中存放的是同一个人的检测结果。
