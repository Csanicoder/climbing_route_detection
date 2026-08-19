import cv2

class Video:
    """
    Class for loading and reading a video file.

    Attributes:
        _cap: private VideoCapture object
        _ret, _frame: private members for reading the video
        VIDEO_PATH (str): relative path to the video file
        WIDTH (int): width of the frame, in pixels
        HEIGHT (int): height of the frame, in pixels
        FRAME_COUNT (int): total number of frames
        FPS (int): number of frames per second
        FRAME_TIME (float): duration of one frame, in second (1 / FPS)
        LENGTH (float): length of the video in seconds
    """

    def __init__(self, video_path : str):
        """
        Load a video from a file and get its parameters.
        Also reads the first frame

        Args:
            video_path (str): relative path to the video file
        """

        self._cap = cv2.VideoCapture(video_path)

        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        self.VIDEO_PATH = video_path
        self.WIDTH = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.HEIGHT = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.FRAME_COUNT = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.FPS = int(self._cap.get(cv2.CAP_PROP_FPS))
        self.FRAME_TIME = 1 / self.FPS
        self.LENGTH = self.FRAME_COUNT * self.FRAME_TIME

        self._ret, self._frame = self._cap.read()
        if not self._ret:
            raise RuntimeError("Could not read first frame from video.")

    def __del__(self):
        """Release the video capture when the instance is destroyed"""
        self._cap.release()

    def get_index(self):
        """Get the index of the current frame"""
        return self._cap.get(cv2.CAP_PROP_POS_FRAMES) - 1

    def next(self):
        """Get the next frame"""
        if self.get_index() + 1 == self.FRAME_COUNT: # If the video is finished, jump to the start
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self._ret, self._frame = self._cap.read()
        if not self._ret:
            raise RuntimeError(f"Could not read frame {self.get_index() + 1} from video.")

        return self._frame


    def current(self):
        """Get the current frame"""
        return self._frame

    def get_frame(self, frame_idx):
        """Get the frame at index `frame_idx`"""
        if frame_idx == 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.FRAME_COUNT - 2)

        else:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
        return self.next()
