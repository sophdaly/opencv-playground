import cv2
import numpy as np
import argparse

COLOR_RANGE = {
    'blue': ([90, 84, 69], [120, 255, 255]),
    'yellow': ([10, 100, 100], [40, 255, 255]),
    'green': ([40, 80, 32], [70, 255, 255]),
    'red': ([160, 100, 100], [179, 255, 255]),
    'orange': ([160, 100, 47], [179, 255, 255])
}


def create_mask(src):
    """
    Return mask using colors set in COLOR_RANGE dict
    """

    mask = np.zeros(src.shape[:2], dtype="uint8")
    for color, (lower, upper) in COLOR_RANGE.items():
        mask += cv2.inRange(src, np.array(lower), np.array(upper))

    return mask


def set_video():
    """
    Return video input from video or camera
    """

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="Path to video file")
    args = vars(ap.parse_args())

    if not args.get("video", False):
        return cv2.VideoCapture(0)
    else:
        return cv2.VideoCapture(args["video"])


if __name__ == "__main__":

    camera = set_video()

    while True:
        # Convert each frame from BGR to HSV
        _, frame = camera.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold the hsv image to get only outlined colors
        mask = create_mask(hsv)

        # Perform erosions and dilations to solidify color blobs
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Display output
        output = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        cv2.imshow('output', output)

        # Break loop on ESC
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    # Close camera and windows
    camera.release()
    cv2.destroyAllWindows()
