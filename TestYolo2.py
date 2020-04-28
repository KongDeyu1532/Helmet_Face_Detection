import argparse
import os

import cv2
# from matplotlib import pyplot as plt
import mxnet as mx
from gluoncv import model_zoo, utils, data


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



if __name__ == '__main__':
    args = parse_args()

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

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    while True:
        _, frame = cap.read()
        orig_img = frame
        x = mx.nd.array(frame)
        x, orig_img = data.transforms.presets.yolo.transform_test(x)
        x = x.as_in_context(ctx)
        box_ids, scores, bboxes = net(x)
        ax = utils.viz.cv_plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], class_names=net.classes,
                                    thresh=args.threshold)
        # cv2.imshow("s",ax)
        # print(ax)

        cv2.imshow('image', orig_img)
        print(ax)
        key = cv2.waitKey(10)

        if key & 0xFF == 27:
            break
    #
    cv2.destroyAllWindows()
