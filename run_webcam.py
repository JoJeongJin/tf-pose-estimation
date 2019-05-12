import argparse
import logging
import time

import cv2
import numpy as np
#import Opengl_test

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def drawCube(vertices):
    glBegin(GL_LINES)
    for vertex in vertices:
            glVertex3fv(vertex)
    glEnd()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    # Opengl_test.myOpenGL()

    #이 부분이 렌더링 시작을 위한 초기화 부분
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -9)

    while True:

        ##opengl
        # Opengl_test.myOpenGL()
        ##opengl_end

        ret_val, image = cam.read()

        logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        print(image.shape)

        alist = []
        # for i in range (0, 480):
        #     for j in range (0, 640):
        #         if (image[i, j, 0] == 255) and (image[i, j, 1] == 255) and (image[i, j, 2] == 255):
        #             alist.append([i, j])
        # arr = np.array(alist)

        # cv2.imshow('tf-pose-estimation result', image)
        vertices = []
        only_pose_image = image
        for i in range (0, 480):
            for j in range (0, 640):
                if (image[i, j, 0] == 255) and (image[i, j, 1] == 255) and (image[i, j, 2] == 255):

                    x_input = float(i) / 195
                    y_input = float(j) / 162
                    vertices.append([x_input,y_input,0])
                    print([x_input,y_input,0])
                    continue
                else:
                    only_pose_image[i,j, 0] = 0
                    only_pose_image[i,j, 1] = 0
                    only_pose_image[i,j, 2] = 0

        #이미지를 이미지 프로세싱을 통해서 흰색인 것만 남기고 나머지는 까맣게 만든다.
        #행렬에 그 좌표로 넣는다.

        #alist.append([i,j,k])
        #arr = np.array(alist)

        #그리고 그 행렬을 저장하고
        #OpenGL에서 띄워본다.

        #선 연결된 부분은 연결된 것으로 처리하는게 좋다.

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        drawCube(vertices)
        pygame.display.flip()

        logger.debug('show+')
        # cv2.putText(image,
        #             "FPS: %f" % (1.0 / (time.time() - fps_time)),
        #             (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             (0, 255, 0), 2)
        # cv2.imshow('only_pose_image', only_pose_image)
        import matplotlib.pyplot as plt

        plt.plot(vertices[0],vertices[1])
        print("plot 출력")
        plt.savefig('demo.png')

        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')

    cv2.destroyAllWindows()
