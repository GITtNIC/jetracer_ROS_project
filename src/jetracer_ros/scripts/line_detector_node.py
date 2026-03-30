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

        self.debug_image_pub = rospy.Publisher(
            '/line_detector/debug_image/compressed',
            CompressedImage,
            queue_size=1
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

            # Look only at lower part of image
            roi_y_start = int(height * 0.6)
            roi = frame[int(height * 0.6):height, :]

            # Convert to grayscale
            gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

            # Blur to reduce noise
            blurred = cv.GaussianBlur(gray, (5, 5), 0)

            # Threshold: dark areas become white in mask
            _, mask = cv.threshold(blurred, 100, 255, cv.THRESH_BINARY_INV)

            # Find all white pixels in mask
            ys, xs = np.where(mask > 0)

            debug_frame = frame.copy()

            # Draw ROI box
            cv.rectangle(debug_frame, (0, roi_y_start), (width - 1, height - 1), (255, 0, 0), 2)

            # Draw image center line
            center_x = width / 2.0
            cv.line(debug_frame, (int(center_x), 0), (int(center_x), height - 1), (0, 255, 255), 2)

            if len(xs) < 50:
                rospy.logwarn_throttle(2, "No clear line detected.")
                self.publish_debug_image(debug_frame)
                return

            # Estimate line position from average x
            x_line = int(np.mean(xs))

            error_px = x_line - center_x
            relative = (error_px / center_x) * 100.0
            relative = max(-100.0, min(100.0, relative))

            self.line_pub.publish(Float32(relative))

            # Draw detected line position on full image
            cv.line(debug_frame, (x_line, roi_y_start), (x_line, height - 1), (0, 255, 0), 3)

            # Draw text
            text = "x=%d  rel=%.1f" % (x_line, relative)
            cv.putText(
                debug_frame,
                text,
                (20, 40),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            self.publish_debug_image(debug_frame)

            rospy.loginfo_throttle(
                1,
                "Line detected: x=%d px | relative_line_hpos=%.1f",
                x_line,
                relative
            )

        except Exception as e:
            rospy.logerr("Error in image_callback: %s", str(e))

    def publish_debug_image(self, frame):
        try:
            success, encoded_image = cv.imencode('.jpg', frame)
            if not success:
                rospy.logwarn("Failed to encode debug image.")
                return

            debug_msg = CompressedImage()
            debug_msg.header.stamp = rospy.Time.now()
            debug_msg.format = "jpeg"
            debug_msg.data = np.array(encoded_image).tostring()

            self.debug_image_pub.publish(debug_msg)

        except Exception as e:
            rospy.logerr("Error publishing debug image: %s", str(e))


if __name__ == '__main__':
    node = LineDetectorNode()
    rospy.spin()
