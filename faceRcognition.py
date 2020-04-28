#encoding: utf-8
#author: KongDeyu

import cv2
import dlib
from PyQt5.QtCore import QThread
from core2 import CoreUI
import core2



class FaceProcessingThread(QThread):
    def __init__(self,faceQueue,faceStartEvent,faceEndEvent):
        super(FaceProcessingThread, self).__init__()
        self.faceQueue = faceQueue
        self.startevent = faceStartEvent
        self.endevent = faceEndEvent
        self.isRunning = True

    def setIsRunning(self, isRunning):
        self.isRunning = isRunning

    def run(self):
        while self.isRunning:
            isSuccess = True
            self.startevent.wait()
            prm = self.faceQueue.get()
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            face_id, confidence = recognizer.predict(prm.gray)
            self.facequeue.put(core2.FaceResult(isSuccess,face_id,confidence))
            self.startevent.clear()# 清除标志，由主进程重置，以便开始下一帧图像的处理
            self.endevent.set()# 重置事件，向主进程报告本次处理结束
