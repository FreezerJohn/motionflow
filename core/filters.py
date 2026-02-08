import time

import numpy as np
from OneEuroFilter import OneEuroFilter


class PoseFilter:
    def __init__(self, min_cutoff=1.0, beta=0.005, d_cutoff=1.0, freq=30.0):
        self.filters = {}
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.freq = freq
        self.last_timestamp = None

    def filter(self, keypoints, timestamp=None):
        """
        Filter keypoints.

        :param keypoints: Numpy array of shape (N, 3) or (N, 2) where N is number of keypoints.
        :param timestamp: Current timestamp. If None, uses time.time().
        :return: Filtered keypoints.
        """
        if timestamp is None:
            timestamp = time.time()

        if self.last_timestamp is None:
            self.last_timestamp = timestamp
            # Initialize filters for each coordinate of each keypoint
            # keypoints shape: (17, 3) -> x, y, conf
            # We only filter x and y.
            for i in range(len(keypoints)):
                # OneEuroFilter(freq, mincutoff=1.0, beta=0.0, dcutoff=1.0)
                self.filters[f"x_{i}"] = OneEuroFilter(self.freq,
                                                      mincutoff=self.min_cutoff,
                                                      beta=self.beta,
                                                      dcutoff=self.d_cutoff)
                # Initialize with first value
                self.filters[f"x_{i}"](keypoints[i][0], timestamp)

                self.filters[f"y_{i}"] = OneEuroFilter(self.freq,
                                                      mincutoff=self.min_cutoff,
                                                      beta=self.beta,
                                                      dcutoff=self.d_cutoff)
                # Initialize with first value
                self.filters[f"y_{i}"](keypoints[i][1], timestamp)
            return keypoints

        filtered_keypoints = np.copy(keypoints)

        for i in range(len(keypoints)):
            # Filter X
            filtered_keypoints[i][0] = self.filters[f"x_{i}"](keypoints[i][0], timestamp)
            # Filter Y
            filtered_keypoints[i][1] = self.filters[f"y_{i}"](keypoints[i][1], timestamp)
            # Conf (index 2) is left as is

        self.last_timestamp = timestamp
        return filtered_keypoints
