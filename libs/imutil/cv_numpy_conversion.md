
1. How to represent the cv::Mat size, and numpy array shape

    - cv::Mat : `Size(cols, rows) # (width, height)`
    - numpy array: `height, width, nchannels = array.shape`
    - They are reverse order, it's wrong to directly input numpy's shape into OpenCV Size in api like 'cv2.resize(src, Size)'


