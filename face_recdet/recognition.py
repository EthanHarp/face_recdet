#server
#receives response, sends request

from custom_interfaces.srv import ReadyForRec                                                          # CHANGE

import rclpy
from rclpy.node import Node

from PIL import Image
import torch
import numpy as np
import os
from facenet_pytorch import fixed_image_standardization
from torchvision import transforms


class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(ReadyForRec, 'ready_for_rec', self.ready_for_rec_callback)       # CHANGE

    def ready_for_rec_callback(self, request, response):
        index = recognize_face()
        response.res = index                                            # CHANGE
        self.get_logger().info('Incoming request\nrequest: %d' % (request.req))  # CHANGE

        return response

#create new function for recognition, returns index 0 for ethan 1 for rando

def recognize_face():
    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization,
    ])


    #https://www.youtube.com/watch?v=Upw4RaERZic
    img = Image.open("/YOUR_DIRECTORY/face_recdet/data/subject.jpg")
    model = torch.load('/YOUR_DIRECTORY/face_recdet/data/CustomModel.pyh')
    img = trans(img)
    img = torch.unsqueeze(img, 0)
    print(img.shape)
    model = model.eval()
    probs = model(img)
    value,index = torch.max(probs,1)
    int(index.numpy())
    #labels[str(int(index.numpy()))]
    print("made it here")
    print(index)
    return index.item()


def main(args=None):
    rclpy.init(args=args)

    minimal_service = MinimalService()

    rclpy.spin(minimal_service)

    rclpy.shutdown()

if __name__ == '__main__':
    main()