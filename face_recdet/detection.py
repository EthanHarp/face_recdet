#client
#sends request, receives response

from custom_interfaces.srv import ReadyForRec                          # CHANGE
import sys
import rclpy
from rclpy.node import Node

from mtcnn import MTCNN 
import cv2 


class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(ReadyForRec, 'ready_for_rec')       # CHANGE
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = ReadyForRec.Request()                                   # CHANGE

    def send_request(self, state):
        detect_face()
        self.req.req = state                                  # CHANGE
        self.future = self.cli.call_async(self.req)


#create new function for the actual detection that says subject.jpg if it finds something and returns 1 for new and 0 for nothing found
def detect_face():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    img = frame
    img = cv2.imread("/YOUR_DIRECTORY/face_recdet/data/subject.jpg")

    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    results = detector.detect_faces(img2)
    #add handling for no face
    print(results)
    #no need to actually display anything or draw box
    cv2.imshow('img2', img2)
    cv2.waitKey(0)
    bounding_box = results[0]['box']
    cv2.rectangle(img, (bounding_box[0], bounding_box[1]), (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]), (0,155,255), 2)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 900, 900)
    

    cropped = img[(bounding_box[1]):(bounding_box[1] + bounding_box[3]), (bounding_box[0]):(bounding_box[0]+bounding_box[2])]
   
    resized = cv2.resize(cropped, (160, 160), interpolation= cv2.INTER_LINEAR)
    cv2.imwrite("/YOUR_DIRECTORY/face_recdet/data/subject.jpg", resized)
    cv2.imshow('cropped', resized)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)

    minimal_client = MinimalClientAsync()
    #HERE is where mtcnn says when it is ready for recognition. 1 for there is a new subject, 0 for no
    minimal_client.send_request(1)

    while rclpy.ok():
        rclpy.spin_once(minimal_client)
        if minimal_client.future.done():
            try:
                #translate the response index into tag of Ethan or Pedestrian
                response = minimal_client.future.result()
            except Exception as e:
                minimal_client.get_logger().info(
                    'Service call failed %r' % (e,))
            else:
                minimal_client.get_logger().info(
                    'Result of ready_for_rec: request = %d, response = %d' %                                # CHANGE
                    (minimal_client.req.req, response.res))  # CHANGE
            break

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()