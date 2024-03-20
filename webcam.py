from cv2 import CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_DSHOW
from cv2 import VideoCapture, waitKey

from structs import Frame

class Webcam:
    def __init__ (self, width : int, height : int):
        # init video capture
        # self.video_capture = VideoCapture('device-0_video.avi')
        self.video_capture = VideoCapture(0, CAP_DSHOW)
        self.video_capture.set(CAP_PROP_FRAME_WIDTH, width)
        self.video_capture.set(CAP_PROP_FRAME_HEIGHT, height)

    @property
    def width (self) -> float:
        return self.video_capture.get(CAP_PROP_FRAME_WIDTH)
    
    @property
    def height (self) -> float:
        return self.video_capture.get(CAP_PROP_FRAME_HEIGHT)
    
    def capture (self) -> tuple[bool, Frame]:
        '''
            return: [do frame captured, frame]
        '''
        ret, frame = self.video_capture.read()
        if not ret:
            print('[webcam] not frame captured')
            return
        return ret, frame
    
    def wait (self, delay : int = 1):
        waitKey(delay)
