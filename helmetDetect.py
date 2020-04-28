#encoding: utf-8
#author: KongDeyu
import argparse
import mxnet as mx
from PyQt5.QtCore import QThread
from gluoncv.model_zoo import model_zoo
from gluoncv import model_zoo, utils, data
from face_recognition_py.core2 import CoreUI
import core2


class HelmetProcessingThread(QThread):
    def __init__(self, helmetQueue,helmetStartEvent,helmetEndEvent):
        super(HelmetProcessingThread, self).__init__()
        self.helmetQueue = helmetQueue
        self.startevent = helmetStartEvent
        self.endevent = helmetEndEvent
        self.isRunning = True

    def setIsRunning(self,isRunning):
        self.isRunning = isRunning


    def run(self):
        def parse_args():
            parser = argparse.ArgumentParser(description='Train YOLO networks with random input shape.')
            parser.add_argument('--network', type=str, default='yolo3_darknet53_voc',
                                # use yolo3_darknet53_voc, yolo3_mobilenet1.0_voc, yolo3_mobilenet0.25_voc
                                help="Base network name which serves as feature extraction base.")
            parser.add_argument('--short', type=int, default=416,
                                help='Input data shape for evaluation, use 320, 416, 512, 608, '
                                     'larger size for dense object and big size input')
            parser.add_argument('--threshold', type=float, default=0.4,
                                help='confidence threshold for object detection')

            parser.add_argument('--gpu', action='store_false',
                                help='use gpu or cpu.')

            args = parser.parse_args()
            return args

        while self.isRunning:
                prm = self.helmetQueue.get()
                args = parse_args(self)

                ctx = mx.cpu()

                net = model_zoo.get_model(args.network, pretrained=False)

                classes = ['hat', 'person']
                for param in net.collect_params().values():
                    if param._data is not None:
                        continue
                    param.initialize()
                net.reset_class(classes)
                net.collect_params().reset_ctx(ctx)

                if args.network == 'yolo3_darknet53_voc':
                    net.load_parameters('darknet.params', ctx=ctx)
                    print('use darknet to extract feature')
                elif args.network == 'yolo3_mobilenet1.0_voc':
                    net.load_parameters('mobilenet1.0.params', ctx=ctx)
                    print('use mobile1.0 to extract feature')
                else:
                    net.load_parameters('mobilenet0.25.params', ctx=ctx)
                    print('use mobile0.25 to extract feature')


                x = mx.nd.array(prm.frame)
                x, orig_img = data.transforms.presets.yolo.transform_test(x)
                x = x.as_in_context(ctx)
                box_ids, scores, bboxes = net(x)
                status = utils.viz.cv_plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], class_names=net.classes,
                                            thresh=args.threshold)
                self.hasHelmet = True
                self.helmetQueue.put(core2.HelmetResult(self.hasHelmet,status))
                self.startevent.clear()  # 清除标志，由主进程重置，以便开始下一帧图像的处理
                self.endevent.set()  # 重置事件，向主进程报告本次处理结束
