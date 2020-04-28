#encoding: utf-8
#author: KongDeyu

#!/usr/bin/python
# -*- coding: UTF-8 -*-

import threading
import time,os
import multiprocessing
import random

'''
    多进程及进程间通信的问题：
    主进程启动后，创建二个子进程，然后：
    1)主进程从视频流中取得一帧图像，分别提交给2个子进程来处理；
    2)主程序等待子进程的处理结果：每个子进程处理完成后，向主进程报告处理结果；
    3)主进程收到二个子进程的处理结果后向用户界面或后台记录处理结果；
    4）重复上述步骤1-3。
'''
# 人脸检测。检测给定参数image中是否有人脸。
def isFace(image):
    hasFace = False;
    xTop, yLeft = (0, 0)

    # 根据image，检测图片中人脸是否存在，如果存在，hasFace＝True,(xTop, yLeft)给出人脸位置的左上角坐标。
    # 以下为模拟
    yes = (random.random() * 10)  % 2  # random.random()随机生成0-1之间的小数

    if yes:
        hasFace = True;
        xTop, yLeft = (random.random() * 300, random.random() * 300)

    # 返回处理结果
    return yes, xTop, yLeft

'''
 人脸识别的进程，进程通信使用了Evnet
    event = threading.Event()  # 设置一个事件实例
    event.set()  # 设置标志位,evnet标志位为True时，该Event有效
    event.clear()  # 清空标志位,evnet标志位为False时，该Event无效
    event.wait()  # 等待设置标志位，标志位False时，阻塞调用些操作的进程，等待该事件的标志为True。
'''
class FaceRecognitionThread(threading.Thread):
    def __init__(self,q , startEvent, endEvnet):
        threading.Thread.__init__(self)
        self.q = q
        self.startEvent = startEvent
        self.endEvent = endEvnet
        self.isRunning = True

    def setIsRunning(self, isRunning):
        self.isRunning = isRunning

    def run(self):
        while (self.isRunning):
            isSuccess = False # 识别成功，置为True
            objectName = ''   # 识别成功，保存目标对象的名字或者编码。
            oName = ['张华', '李华', '王华', '孙华', '沈华']

            # 1、识别image对象（图片）中的人脸，返回对象名称
            # 2、修改isSuccess和objectName属性
            # 模拟：
            self.startEvent.wait()               # 等待主进程准备好图片后，置位Event对象
            time.sleep(random.random() * 5)

            index = int(((random.random() * 100) % 6))

            if(index >=0 and index < 5):
                isSuccess = True
                objectName = oName[index]

                self.q.put(FaceRecognitionResult(isSuccess, objectName))
                print("\t检测到的信息：", '识别成功：',  objectName)
            else:
                self.q.put(FaceRecognitionResult(False, ''))

            self.startEvent.clear()      # 清除标志，由主进程重置，以便开始下一帧图像的处理
            self.endEvent.set()          # 重置事件，向主进程报告本次处理结束

# 头盔检测,image 检测的图像，q--保存检测结果
class HelmetDetectionThread(threading.Thread):
    def __init__(self, q, startEvent, endEvnet):
        threading.Thread.__init__(self)
        self.q = q
        self.startEvent = startEvent
        self.endEvent = endEvnet
        self.isRunning = True

    def setIsRunning(self, isRunning):
        self.isRunning = isRunning

    def run(self):
        while(self.isRunning):
            hasHelmet = False  # 识别完成，根据识别结果设置为True或False
            x = -1             # 识别完成，根据识别结果设置头盔的左上角坐标(x,y)
            y = -1

            # while(True):
            # 1、识别image对象（图片）中的有无头盔
            # 2、修改hasHelmet属性
            # 模拟：
            self.startEvent.wait()   # 等待主进程准备好图片后，置位Event对象
            time.sleep(random.random() * 5)

            # 检测结果模拟
            result = (int(random.random() * 100)) % 2  # random.random()随机生成0-1之间的小数

            if result == 0:
                hasHelmet = True;
                x, y = (random.random() * 300, random.random() * 300)

                self.q.put(HelmetDetectionResult(hasHelmet, x, y))
            else:
                self.q.put(HelmetDetectionResult(False, -1, -1))

            self.startEvent.clear() # 清除标志，由主进程重置，以便开始下一帧图像的处理
            self.endEvent.set()      # 重置事件，向主进程报告本次处理结束

# 人脸识别中的数据通信的对象，传递识别结果
class FaceRecognitionResult:
    def __init__(self, isSuccess, objectName):
        self.isSuccess = isSuccess  # 识别成功，置为True
        self.objectName = objectName    # 识别成功，保存目标对象的名字或者编码。

    def setResult(self, isSuccess, oName):
        self.isSuccess = isSuccess
        self.objectName = oName

    def getResult(self):
        return  self.isSuccess, self.objectName

# 头盔检测中的数据通信的对象，传递识别结果
class HelmetDetectionResult():
    def __init__(self, hasHelmet, x, y):
        self.hasHelmet = hasHelmet  # 识别完成，根据识别结果设置为True或False
        self.x = x             # 识别完成，根据识别结果设置头盔的左上角坐标(x,y)
        self.y = y

    def setResult(self, hasHelmet, x, y):
        self.hasHelmet = hasHelmet
        self.x = x
        self.y = y

    def getResult(self):
        return self.hasHelmet, self.x, self.y


# import winsound



def CoreGUI():

    # 0 初始化：创建进程：人脸识别进程、头盔检测进程
    # 先创建Event, 以实现进程的同步。
    # 开始事件
    faceStartEvent = multiprocessing.Event()
    helmetStartEvent = multiprocessing.Event()

    # 结束事件
    faceEndEvent = multiprocessing.Event()
    helmetEndEvent = multiprocessing.Event()


    # 通信的数据队列。此queue是multiprocessing重新封装的
    faceQueue = multiprocessing.Queue()
    helmetQueue = multiprocessing.Queue()

    # 1 设置当前程序的功能属性配置：
    # 需要头盔检测功能，调用头盔检测
    isHelmet = True

    # 需要考勤记录功能，调用人脸识别
    isAttendanceRecord = True
    # 是签到(True,否则签退False)还是签退功能
    isSignin = True

    # 创建子进程，并传递进程同步与通信的对象：事件对象、队列
    faceRecognition = FaceRecognitionThread(faceQueue, faceStartEvent,faceEndEvent)
    helmetDetection = HelmetDetectionThread(helmetQueue, helmetStartEvent, helmetEndEvent)

        # 2 主进程中启动子进程，并根据要求操作处理。
    # 实际应该使用：while (True):
    faceRecognition.start()
    helmetDetection.start()

    count = 1
    # 改为True
    while(count <= 10 ):

        # 2.1 从视频流中取一帧图像
        # 2.2 图像中检测是否有人脸存在
        # 2.3 如果图征中有人脸存在，则根据功能配置，进行：
        #         启动人脸识别进程
        #         启动头盔检测进程
        #   否则，休眠一段时间，准备下一次处理；
        # 2.4 2.3中启动的进程完成：
        #     根据功能配置，1)处理考勤功能；2)处理头盔检测警报功能

        ##########################模拟##########################
        # 2.1 从视频流中取一帧图像
        print("已经采集一帧图像...")
        image = ""

        # 2.2 图像中检测是否有人脸存在
        print("检测图像中有没有人脸...", end='')

        start = time.clock()
        yes, xTop, yLeft = isFace(image)

        # 2.3 如果图像中有人脸存在，则根据功能配置，进行：
        #         启动人脸识别进程
        #         启动头盔检测进程
        #   否则，休眠一段时间，准备下一次处理；

        if(yes):

            # 2.4.1
            # faceRecognition = multiprocessing.Process(target=FaceRecognitionProcess, args=(None , faceQueue))
            # helmetDetection = multiprocessing.Process(target=HelmetDetectionProcess, args=(None , helmetQueue))
            # 置开始事件，进程进入数据处理阶段

            print("[图像中的人脸存在]")

            if(isAttendanceRecord):     # 需要考勤记录功能
                print("开始识别图像中的人脸...")
                faceStartEvent.set()

            if(isHelmet):       # 需要头盔检测功能
                print("开始检测图像中的人是否戴上头盔...")
                helmetStartEvent.set()

            # 等待完成人脸识别与头盔检测(结束事件发生)
            if (isAttendanceRecord):   # 需要考勤记录功能
                faceEndEvent.wait()

            if (isHelmet):      # 需要头盔检测功能
                helmetEndEvent.wait()

            # 继续下一步处理
            print("图像的识别与检测完成.")

            # 2.4.2 启动的进程完成后：
            #     根据功能配置，1)处理考勤功能；2)处理头盔检测警报功能
            print("处理检测结果...")
            if (isAttendanceRecord):
                fResult = faceQueue.get()
                if (fResult.isSuccess):
                    print("\t%s: 进入工地." % (fResult.objectName))
                else:
                    print("\t此人进入工地没有识别出来，请采取措施." )

                # 其他需要的处理，如调用考勤处理功能，数据记录功能

                    faceEndEvent.clear()    # 清除结束事件标志，等待再次置位，必须有这一行。

            if (isHelmet):
                hResult = helmetQueue.get()
                # 输出检测结果
                # print(hResult.hasHelmet , ", ", hResult.x, ", ", hResult.y)
                if (hResult.hasHelmet):
                    if (isAttendanceRecord and fResult.isSuccess):
                        print('\t%s ，戴头盔. 图像位置( %d , %d)' % (fResult.objectName, hResult.x, hResult.y))
                    else:
                        print('\t来人没有识别，戴头盔. 图像位置( %d , %d)' % ( hResult.x, hResult.y))
                else:
                    if (isAttendanceRecord and fResult.isSuccess):
                        print('\t%s ，没有戴头盔.' % (fResult.objectName))
                    else:
                        print('\t来人没有识别，也没有戴头盔.' )

                    # 播放警报声音
                    # try:
                        # winsound.PlaySound('*', winsound.SND_ALIAS)
                    # except  e:
                    #     print('Sound system has problems', e)


                # 其他需要的处理
                helmetEndEvent.clear()      # 清除结束事件标志，等待再次置位，必须有这一行。

        else:   # 未检测到图像
            print("[没有检测到图像中的人脸].")
            time.sleep(random.random() * 10)

        # 计算处理时间
        elapsed = (time.clock() - start)
        print("耗时:", elapsed)

        # 重复
        print("继续获取下一帧图像...\n")
        count += 1

if (__name__ == '__main__'):
    myprocess = multiprocessing.Process(target=CoreGUI,args=())
    # 置进程是否为守护进程，如果是守护进程，则主语种结束时，子进程立即结束
    # myprocess.daemon = True
    myprocess.start()
    print("============== 主线程结束(End). ==============")
