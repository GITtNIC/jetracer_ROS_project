#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2 as cv
import numpy as np

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32


class LineDetectorNode:
    def __init__(self):
        rospy.init_node('line_detector_node')

        self.line_pub = rospy.Publisher(
            '/relative_line_hpos',
            Float32,
            queue_size=10
        )

        self.image_sub = rospy.Subscriber(
            '/csi_cam_0/image_raw/compressed',
            CompressedImage,
            self.image_callback,
            queue_size=1
        )

        rospy.loginfo("line_detector_node started. Waiting for camera images...")

    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv.imdecode(np_arr, cv.IMREAD_COLOR)

            if frame is None:
                rospy.logwarn("Could not decode compressed image.")
                return

            height, width = frame.shape[:2]

            # Look only at the lower part of the image
            roi = frame[int(height * 0.6):height, :]

            # Convert to grayscale
            gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

            # Blur to reduce noise
            blurred = cv.GaussianBlur(gray, (5, 5), 0)

            # Threshold: dark areas become white in the mask
            _, mask = cv.threshold(blurred, 100, 255, cv.THRESH_BINARY_INV)

            # Find all white pixels in the mask
            ys, xs = np.where(mask > 0)

            if len(xs) < 50:
                rospy.logwarn_throttle(2, "No clear line detected.")
                return

            # Use average x position as a simple line estimate
            x_line = int(np.mean(xs))

            center_x = width / 2.0
            error_px = x_line - center_x
            relative = (error_px / center_x) * 100.0

            # Limit value to [-100, 100]
            relative = max(-100.0, min(100.0, relative))

            self.line_pub.publish(Float32(relative))

            rospy.loginfo_throttle(
                1,
                "Line detected: x=%d px | relative_line_hpos=%.1f",
                x_line,
                relative
            )

        except Exception as e:
            rospy.logerr("Error in image_callback: %s", str(e))


if __name__ == '__main__':
    node = LineDetectorNode()
    rospy.spin()


