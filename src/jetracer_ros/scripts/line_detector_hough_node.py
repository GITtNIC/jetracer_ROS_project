#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2 as cv
import numpy as np
import math

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32


class LineDetectorHoughNode:
    def __init__(self):
        rospy.init_node('line_detector_hough_node')

        # --- Publishers ---
        self.line_pub = rospy.Publisher(
            '/relative_line_hpos',
            Float32,
            queue_size=10
        )

        self.angle_pub = rospy.Publisher(
            '/line_angle',
            Float32,
            queue_size=10
        )

        self.debug_image_pub = rospy.Publisher(
            '/line_detector/debug_image/compressed',
            CompressedImage,
            queue_size=1
        )

        # --- Parameters ---
        self.canny_low = rospy.get_param('~canny_low', 50)
        self.canny_high = rospy.get_param('~canny_high', 150)
        self.hough_threshold = rospy.get_param('~hough_threshold', 30)
        self.hough_min_length = rospy.get_param('~hough_min_length', 40)
        self.hough_max_gap = rospy.get_param('~hough_max_gap', 20)
        self.roi_top_fraction = rospy.get_param('~roi_top_fraction', 0.4)

        # --- Subscriber ---
        self.image_sub = rospy.Subscriber(
            '/csi_cam_0/image_raw/compressed',
            CompressedImage,
            self.image_callback,
            queue_size=1,
            buff_size=2**24
        )

        rospy.loginfo("line_detector_hough_node started.")

    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv.imdecode(np_arr, cv.IMREAD_COLOR)

            if frame is None:
                rospy.logwarn("Could not decode image.")
                return

            height, width = frame.shape[:2]
            center_x = width / 2.0

            # --- Region of interest: bottom portion of frame ---
            roi_y_start = int(height * self.roi_top_fraction)
            roi = frame[roi_y_start:height, :]

            # --- Preprocessing ---
            gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
            blurred = cv.GaussianBlur(gray, (5, 5), 0)

            # --- Canny edge detection ---
            edges = cv.Canny(blurred, self.canny_low, self.canny_high)

            # --- Hough Line Transform ---
            lines = cv.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=self.hough_threshold,
                minLineLength=self.hough_min_length,
                maxLineGap=self.hough_max_gap
            )

            debug_frame = frame.copy()

            # Draw ROI boundary
            cv.line(debug_frame, (0, roi_y_start), (width, roi_y_start), (255, 0, 0), 2)

            # Draw image centre
            cv.line(debug_frame, (int(center_x), 0), (int(center_x), height), (0, 255, 255), 1)

            if lines is None or len(lines) == 0:
                rospy.logwarn_throttle(2, "No lines detected.")
                self.publish_debug_image(debug_frame)
                return

            # --- Filter lines: keep only near-vertical lines ---
            good_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)

                # Keep lines where vertical component is larger than horizontal
                if dy > dx * 0.5:
                    good_lines.append(line[0])

                    # Draw detected line segments in red on debug image
                    cv.line(debug_frame,
                            (x1, y1 + roi_y_start),
                            (x2, y2 + roi_y_start),
                            (0, 0, 255), 2)

            if len(good_lines) == 0:
                rospy.logwarn_throttle(2, "No vertical lines found.")
                self.publish_debug_image(debug_frame)
                return

            # --- Compute average line position and angle ---
            all_x = []
            all_angles = []

            for x1, y1, x2, y2 in good_lines:
                mid_x = (x1 + x2) / 2.0
                all_x.append(mid_x)

                # Angle from vertical (0 = perfectly vertical)
                angle = math.atan2(abs(x2 - x1), abs(y2 - y1))
                # Preserve sign: positive = line leans right
                if x2 - x1 < 0:
                    angle = -angle
                all_angles.append(angle)

            avg_x = np.mean(all_x)
            avg_angle_deg = np.degrees(np.mean(all_angles))

            # --- Compute relative position [-100, +100] ---
            error_px = avg_x - center_x
            relative = (error_px / center_x) * 100.0
            relative = max(-100.0, min(100.0, relative))

            # --- Publish ---
            self.line_pub.publish(Float32(relative))
            self.angle_pub.publish(Float32(avg_angle_deg))

            # --- Draw result on debug image ---
            avg_x_int = int(avg_x)
            cv.line(debug_frame,
                    (avg_x_int, roi_y_start),
                    (avg_x_int, height),
                    (0, 255, 0), 3)

            text1 = "pos=%.1f%%" % relative
            text2 = "angle=%.1f deg" % avg_angle_deg
            text3 = "%d lines" % len(good_lines)
            cv.putText(debug_frame, text1, (20, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(debug_frame, text2, (20, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(debug_frame, text3, (20, 90),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            self.publish_debug_image(debug_frame)

            rospy.loginfo_throttle(
                1,
                "Hough: pos=%.1f%% | angle=%.1f deg | %d lines",
                relative, avg_angle_deg, len(good_lines)
            )

        except Exception as e:
            rospy.logerr("Error in image_callback: %s", str(e))

    def publish_debug_image(self, frame):
        try:
            success, encoded = cv.imencode('.jpg', frame)
            if not success:
                return
            debug_msg = CompressedImage()
            debug_msg.header.stamp = rospy.Time.now()
            debug_msg.format = "jpeg"
            debug_msg.data = np.array(encoded).tostring()
            self.debug_image_pub.publish(debug_msg)
        except Exception as e:
            rospy.logerr("Error publishing debug image: %s", str(e))


if __name__ == '__main__':
    node = LineDetectorHoughNode()
    rospy.spin()
