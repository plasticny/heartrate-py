import numpy as np
from cv2 import Mat
from cv2 import rectangle, line, imshow, putText
from cv2 import destroyWindow, namedWindow, resizeWindow
from cv2 import minMaxLoc
from cv2 import WINDOW_NORMAL, LINE_4, FONT_HERSHEY_PLAIN
from cv2.typing import MatLike
from structs import FacePosition, Frame

LOW_BPM = 42
HIGH_BPM = 240
SEC_PER_MIN = 60

class Display:
    def __init__ (self, name : str, width : int, height : int):
        self.name = name
        self.width = width
        self.height = height
        
        try:
            destroyWindow(self.name)
        except:
            pass
    
    def start (self) -> None:
        namedWindow(self.name, WINDOW_NORMAL)
        resizeWindow(self.name, self.width, self.height)

    def stop (self) -> None:
        destroyWindow(self.name)
        
    def show_frame (self, frame : Frame):
        imshow(self.name, frame)
        
    def mack_face (self, frame : Frame, face_pos : FacePosition):
        face_x, face_y, face_width, face_height = face_pos
        
        rectangle(
            frame,
            (face_x, face_y),
            (face_x + face_width, face_y + face_height),
            (0, 255, 0)
        )
        
    def mack_overlay (self, frame : Frame, overlay_mask : MatLike) -> Frame:
        frame[np.all(overlay_mask != 0, axis=2)] = [255, 0, 0]
        return frame

    def make_overlay_mask (
            self, frame : Frame,
            rppg : Mat, rppg_fft : Mat,
            face_pos : FacePosition,
            fps : int, bpm : float
        ) -> Mat:
        """
            mack the mask that include the rppg, frequency and bpm onto the frame
        """
        # draw overlay mask
        overlay_mask = Mat(np.zeros((frame.shape[0], frame.shape[1], 1), np.uint8))
        self.drawTime(rppg, face_pos, overlay_mask)
        self.drawFrequency(rppg_fft, fps, overlay_mask)
        self.drawBPM(bpm, face_pos, overlay_mask)

        return overlay_mask
        
    # Draw time domain signal to overlayMask
    def drawTime(self, rppg : Mat, face_pos : FacePosition, overlay_mask : Mat) -> MatLike:
        face_x, face_y, face_width, face_height = face_pos
        
        # Display size
        displayHeight = face_height / 2.0
        displayWidth = face_width * 0.8
        # Signal
        min_val, max_val, _, _ = minMaxLoc(rppg)
        heightMult = displayHeight / (max_val - min_val)
        widthMult = displayWidth / (rppg.shape[0] - 1)
        drawAreaTlX = face_x + face_width + 10
        drawAreaTlY = face_y

        start = (drawAreaTlX, drawAreaTlY + (max_val - rppg[0]) * heightMult)
        for i in range(1, rppg.shape[0]):
            end = (drawAreaTlX + i * widthMult, drawAreaTlY + (max_val - rppg[i]) * heightMult)
            line(overlay_mask, start, end, 1, 2, LINE_4, 0)
            start = end
            
        return overlay_mask

    # Draw frequency domain signal to overlayMask
    def drawFrequency(self, rppg_fft : Mat, fps : int, overlay_mask) -> MatLike:        
        low = int(rppg_fft.shape[0] * LOW_BPM / SEC_PER_MIN / fps)
        high = int(rppg_fft.shape[0] * HIGH_BPM / SEC_PER_MIN / fps)

        bandMask = Mat(np.zeros((rppg_fft.shape[0], 1), np.uint8))
        bandMask[low:high+1, :] = 1

        # Display size
        displayHeight = self.face.height / 2.0
        displayWidth = self.face.width * 0.8
        
        # Signal
        min_val, max_val, _, _ = minMaxLoc(rppg_fft, mask=bandMask)
        heightMult = displayHeight / (max_val - min_val)
        widthMult = displayWidth / (high - low)
        drawAreaTlX = self.face.x + self.face.width + 10
        drawAreaTlY = self.face.y + self.face.height / 2.0

        start = (drawAreaTlX, drawAreaTlY + (max_val - rppg_fft[low]) * heightMult)
        for i in range(low + 1, high + 1):
            end = (drawAreaTlX + (i - low) * widthMult, drawAreaTlY + (max_val - rppg_fft[i]) * heightMult)
            line(overlay_mask, start, end, 1, 2, LINE_4, 0)
            start = end

        return overlay_mask

    # Draw bpm string to overlayMask
    def drawBPM(self, bpm : float, face_pos : FacePosition, overlay_mask) -> MatLike:
        return putText(
            overlay_mask,
            str(bpm),
            (face_pos.x, face_pos.y - 10),
            FONT_HERSHEY_PLAIN,
            1.5, 1, 2
        )
