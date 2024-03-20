from webcam import Webcam
from meter import Meter
from display import Display
from datetime import datetime
from cv2 import Mat

HAARCASCADE_URI = 'haarcascade_frontalface_alt.xml'
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30
WINDOW_SIZE = 1

webcam = Webcam(
    width=FRAME_WIDTH,
    height=FRAME_HEIGHT
)
meter = Meter(
    classifierPath=HAARCASCADE_URI,
    targetFps=TARGET_FPS,
    windowSize=WINDOW_SIZE
)
display = Display(
    name='webcam',
    width=FRAME_WIDTH,
    height=FRAME_HEIGHT
)

last_rppg_called_time = datetime.now()
overlay_mask : Mat = None

display.start()
cnt = 0
while True:
    # capture frame from webcam
    ret, frame = webcam.capture()
    if not ret:
        webcam.wait()
        continue

    # process frame by the meter
    _, do_rppg_updated = meter.process_frame(frame)

    if meter.fps is not None and cnt % 10 == 0:
        print(f'FPS: {meter.fps}')

    if meter.face_pos is not None:
        display.mack_face(frame, meter.face_pos)
    
    # if do_rppg_updated:
    #     overlay_mask = display.make_overlay_mask(
    #         frame=frame, 
    #         rppg=meter.rppg, rppg_fft=meter.rppg_fft,
    #         face_pos=meter.face_pos, fps=meter.fps, bpm=meter.bpm
    #     )
    # if overlay_mask is not None:
    #     frame = display.mack_overlay(frame, overlay_mask)
    
    display.show_frame(frame)
    
    webcam.wait()
    cnt += 1
