import cv2
import numpy as np
from cv2.typing import MatLike
from typing import Union
from structs import FacePosition, Frame

RESCAN_INTERVAL = 1 # in second
RPPG_INTERVAL = 0.25 # in second
DEFAULT_FPS = 30
LOW_BPM = 42
HIGH_BPM = 240
REL_MIN_FACE_SIZE = 0.4
SEC_PER_MIN = 60
MSEC_PER_SEC = 1000
MAX_CORNERS = 10
MIN_CORNERS = 5
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 10

class Meter:
    def __init__(self, classifierPath : str, targetFps : int, windowSize : int):
        self.TARGET_FPS = targetFps
        self.WINDOW_SIZE = windowSize
        self.MAX_SIGNAL_SIZE = self.TARGET_FPS * self.WINDOW_SIZE
        
        self.RESCAN_INTERVAL_TICKS = RESCAN_INTERVAL * cv2.getTickFrequency()
        self.RPPG_INTERVAL_TICKS = RPPG_INTERVAL * cv2.getTickFrequency()

        # Load face detector
        self.classifier = cv2.CascadeClassifier()
        classifier_loaded = self.classifier.load(classifierPath)
        if not classifier_loaded:
            raise '[heartbeat] cannot load the face detector'

        # Set variables
        self.lastScanTime : int = 0
        self.last_compute_rppg_time : int = 0
        
        self.fps : float = DEFAULT_FPS
        self.bpm : float = None
        
        self.face_pos : Union[FacePosition, None] = None # Position of the face
        
        self.signal = [] # 120 x 3 raw rgb values
        self.rppg : cv2.Mat
        self.rppg_fft : cv2.Mat
        
        self.timestamps : list[int] = [] # 120 x 1 timestamps
        self.rescan : list[bool] = [] # 120 x 1 rescan bool        

    def process_frame(self, frame : MatLike) -> tuple[bool, bool]:
        """
            Add one frame to raw signal\n
            Return:
                - face updated
                - rppg updated
        """
        do_face_updated = False
        do_rppg_updated = False
        
        time = cv2.getTickCount()
        frameGray : Frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)

        # scan face
        do_rescan = time - self.lastScanTime >= self.RESCAN_INTERVAL_TICKS
        if self.face_pos is None or do_rescan:
            # scan face if no valid face or need to rescan
            self.lastScanTime = time
            self.face_pos = self.detectFace(frameGray)
            do_face_updated = True
        
        # if no face detected
        if self.face_pos is None:
            self.signal = []
            self.timestamps = []
            self.rescan = []
            return do_face_updated, do_rppg_updated
        
        # Update the signal
        shift_idx = len(self.signal) - self.MAX_SIGNAL_SIZE
        if shift_idx > 0:
            self.signal = self.signal[shift_idx:]
            self.timestamps = self.timestamps[shift_idx:]
            self.rescan = self.rescan[shift_idx:]

        # get rbg mean
        means = cv2.mean(frame, self.makeMask(frameGray, self.face_pos))        
        # Add new values to raw signal buffer
        self.signal.append(means[:3])
        self.timestamps.append(time)
        self.rescan.append(do_rescan)
        
        # update fps
        self.fps = self.compute_fps(self.timestamps)
        
        # compute rppg
        do_enough_sample = len(self.signal) >= self.MAX_SIGNAL_SIZE
        if not do_enough_sample:
            print(f'Not enough sample: {len(self.signal)} / {self.MAX_SIGNAL_SIZE}')
        elif time - self.last_compute_rppg_time >= self.RPPG_INTERVAL_TICKS:
            self.rppg = self.compute_rppg(self.signal)
            self.rppg_fft = self.fft_rppg(self.rppg, True)
            self.bpm = self.compute_bpm(self.rppg_fft, self.fps)
            
            self.last_compute_rppg_time = time
            do_rppg_updated = True
        
        return do_face_updated, do_rppg_updated

    # Run face classifier
    def detectFace(self, gray : MatLike) -> Union[FacePosition, None]:
        faces = cv2.CascadeClassifier.detectMultiScale(self.classifier, gray, 1.1, 3, 0)
        return faces[0] if len(faces) > 0 else None

    # Make ROI mask from face
    def makeMask(self, frameGray : Frame, face : FacePosition) -> MatLike:
        pt1 = (int(face[0] + 0.3 * face[2]), int(face[1] + 0.1 * face[3]))
        pt2 = (int(face[0] + 0.7 * face[2]), int(face[1] + 0.25 * face[3]))
        return cv2.rectangle(
            np.uint8(np.zeros(frameGray.shape)),
            pt1, pt2,
            (255, 255, 255, 255), -1
        )

    # Compute rppg signal and estimate HR
    def compute_rppg(self, signal : list):
        assert len(signal) >= self.MAX_SIGNAL_SIZE, "enough signal for estimate"

        rppg = cv2.Mat(np.array(signal).reshape(-1, 1, 3).astype(np.float32))
        
        # Filtering
        rppg = self.denoise(rppg, self.rescan)
        rppg = self.standardize(rppg)
        rppg = self.detrend(rppg, self.fps)
        rppg = self.movingAverage(rppg, 3, max(int(self.fps/6), 2))
        
        # HR estimation
        rppg = self.selectGreen(rppg)
        
        return rppg
    
    def compute_fps (self, ts : list[int]) -> int:
        if len(ts) <= 1:
            return DEFAULT_FPS
        
        diff = ts[-1] - ts[0]
        return len(ts) / diff * cv2.getTickFrequency()
    
    # Remove noise from face rescanning
    def denoise(self, signal : cv2.Mat, rescan : list[bool]) -> cv2.Mat:
        row_cnt = signal.shape[0]

        diff = cv2.subtract(signal[1:], signal[:-1])
        for i in range(1, row_cnt):
            if not rescan[i]:
                continue
            
            diff_i = diff[i-1][0]
            
            adjR = np.zeros(row_cnt, np.float32)
            adjG = np.zeros(row_cnt, np.float32)
            adjB = np.zeros(row_cnt, np.float32)
            adjR[i:] = diff_i[0]
            adjG[i:] = diff_i[1]
            adjB[i:] = diff_i[2]
            
            adjV = cv2.merge([adjR, adjG, adjB])
                        
            signal = signal - adjV
            # signal = cv2.subtract(signal, adjV)
            
        return signal

    # Standardize signal
    def standardize(self, signal : cv2.Mat):
        r_ls, g_ls, b_ls = [], [], []
        for frame in signal:
            frame = frame[0]
            r_ls.append(frame[0])
            g_ls.append(frame[1])
            b_ls.append(frame[2])
        r_ls = np.array(r_ls, np.float32)
        g_ls = np.array(g_ls, np.float32)
        b_ls = np.array(b_ls, np.float32)
        
        mean_r, std_dev_r = cv2.meanStdDev(r_ls)
        mean_g, std_dev_g = cv2.meanStdDev(g_ls)
        mean_b, std_dev_b = cv2.meanStdDev(b_ls)
        
        means_c3 = cv2.Mat(np.array([[mean_r[0][0], mean_g[0][0], mean_b[0][0]]]))
        stdDev_c3 = cv2.Mat(np.array([[std_dev_r[0][0], std_dev_g[0][0], std_dev_b[0][0]]]))
        
        means = cv2.repeat(means_c3, signal.shape[0], 1).reshape(-1, 1, 3)
        stdDevs = cv2.repeat(stdDev_c3, signal.shape[0], 1).reshape(-1, 1, 3)

        signal = signal - means
        signal = signal / stdDevs
        # signal = cv2.subtract(signal, means)
        # signal = cv2.divide(signal, stdDevs)
        
        return signal

    # Remove trend in signal
    def detrend(self, signal : cv2.Mat, lmbda) -> cv2.Mat:
        cnt_rows = signal.shape[0]
        
        h = cv2.Mat(np.zeros((cnt_rows-2, cnt_rows), np.float32))
        i = cv2.Mat(np.eye(cnt_rows, dtype=np.float32))
        t = cv2.Mat(np.zeros((cnt_rows-2, cnt_rows), np.uint8))

        # set diagonal 0,1,2 of h
        for row in range(cnt_rows-2):
            h[row, row] = 1
            h[row, row+1] = -2
            h[row, row+2] = 1

        h = cv2.gemm(h, h, lmbda*lmbda, t, 0, flags=cv2.GEMM_1_T)
        h = cv2.add(i, h)
        h = cv2.invert(h, flags=cv2.DECOMP_LU)[1]
        h = cv2.subtract(i, h)

        s = list(cv2.split(signal))
        for i in range(len(s)):
            s[i] = cv2.Mat(np.array(s[i], dtype=np.float32)).reshape(-1,3)
            s[i] = cv2.gemm(h, s[i], 1, t, 0, flags=0)
        
        # s[0] = cv2.gemm(h, s[0], 1, t, 0, flags=0)
        # s[1] = cv2.gemm(h, s[1], 1, t, 0, flags=0)
        # s[2] = cv2.gemm(h, s[2], 1, t, 0, flags=0)
        
        signal = cv2.merge(s)
        
        return signal

    # Moving average on signal
    def movingAverage(self, signal : cv2.Mat, n : int, kernelSize : int) -> MatLike:
        for _ in range(n):
            signal = cv2.blur(signal, (kernelSize, 1))
        return signal

    def selectGreen(self, signal : cv2.Mat) -> MatLike:
        rgb = cv2.split(signal)[0]
        return rgb[:,1]
    
    def fft_rppg(self, signal : MatLike, do_magnitude) -> MatLike:
        """ Convert time domain signal to frequency domain signal"""
        planes : list[MatLike] = [
            signal,
            cv2.Mat(np.zeros((signal.shape[0], 1), np.float32))
        ]
        signal = cv2.merge(planes)
                
        # Fourier transform
        signal = cv2.dft(signal, flags=cv2.DFT_COMPLEX_OUTPUT)

        if do_magnitude:
            planes = cv2.split(signal)
            signal = cv2.magnitude(planes[0], planes[1])
            
        return signal
    
    def compute_bpm (self, rppg_fft : cv2.Mat, fps : int) -> float:
        rows_cnt = rppg_fft.shape[0]
        
        low = int(rows_cnt * LOW_BPM / SEC_PER_MIN / fps)
        high = int(rows_cnt * HIGH_BPM / SEC_PER_MIN / fps)
        
        mask = cv2.Mat(np.zeros((rows_cnt, 1), np.uint8))
        mask[low:high+1, :] = 1
        
        _,_,_,max_loc = cv2.minMaxLoc(rppg_fft, mask=mask)
        
        bpm = max_loc[1] * fps / rows_cnt * SEC_PER_MIN
        # bpm = max_val * fps / rows_cnt * SEC_PER_MIN
        
        print(bpm, max_loc[1], fps)
        # print(bpm, max_val, max_loc[1], fps)
        
        return bpm
