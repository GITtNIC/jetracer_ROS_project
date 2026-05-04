#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Edge-based line detector  (shared v1/v2 perception node, plan D1 / D6).

Pipeline (Szeliski 2022, sections 7.2 and 7.4):
    grayscale  ->  Gaussian blur  ->  Canny  ->  probabilistic Hough
      ->  slope filter (discard near-horizontal segments)
      ->  project each segment to the ROI's bottom row
      ->  centerline x = median of those projections
      ->  lateral_error = (center_x - image_center) / (width/2)   in [-1, 1]

A grayscale / threshold path is intentionally NOT used here (plan D1):
v1 and v2 must share the same perception so that any performance
difference is attributable to the controller.

Publishes:
    /line/lateral_error                      std_msgs/Float32 in [-1, 1]
    /line_detector/debug_image/compressed    sensor_msgs/CompressedImage
"""

import rospy
import cv2 as cv
import numpy as np

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32


class EdgeLineDetectorNode:
    def __init__(self):
        rospy.init_node('linefollow_v2_traj')

        # --- Canny ---
        self.canny_low = rospy.get_param('~canny_low', 50)
        self.canny_high = rospy.get_param('~canny_high', 150)

        # --- Probabilistic Hough ---
        self.hough_threshold = rospy.get_param('~hough_threshold', 30)
        self.hough_min_len = rospy.get_param('~hough_min_len', 20)
        self.hough_max_gap = rospy.get_param('~hough_max_gap', 10)

        # --- Filtering ---
        #   min_abs_slope = minimum |dy/dx| to keep a segment.
        #   Near-horizontal segments (texture, shadows) are rejected.
        self.min_abs_slope = rospy.get_param('~min_abs_slope', 0.3)

        # --- ROI: bottom fraction of the frame ---
        self.roi_top = rospy.get_param('~roi_top', 0.6)

        # --- Gaussian blur kernel (odd) ---
        self.blur_ksize = int(rospy.get_param('~blur_ksize', 5))
        if self.blur_ksize % 2 == 0:
            self.blur_ksize += 1

        self.line_pub = rospy.Publisher(
            '/line/lateral_error', Float32, queue_size=10
        )
        self.debug_image_pub = rospy.Publisher(
            '/line_detector/debug_image/compressed',
            CompressedImage, queue_size=1
        )

        self.image_sub = rospy.Subscriber(
            '/csi_cam_0/image_raw/compressed',
            CompressedImage,
            self.image_callback,
            queue_size=1,
        )

        rospy.loginfo(
            "edge_trajectory_detector_node started. "
            "canny=(%d, %d)  hough_thr=%d  min|slope|=%.2f  roi_top=%.2f",
            self.canny_low, self.canny_high,
            self.hough_threshold, self.min_abs_slope, self.roi_top,
        )

    # ---------------------------------------------------------------------
    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv.imdecode(np_arr, cv.IMREAD_COLOR)
            if frame is None:
                rospy.logwarn("Could not decode compressed image.")
                return

            height, width = frame.shape[:2]
            roi_y0 = int(height * self.roi_top)
            roi = frame[roi_y0:height, :]
            roi_h = roi.shape[0]

            # --- Szeliski 7.2: edge detection ---
            gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
            blurred = cv.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)
            edges = cv.Canny(blurred, self.canny_low, self.canny_high)

            # --- Szeliski 7.4: probabilistic Hough ---
            segments = cv.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180.0,
                threshold=self.hough_threshold,
                minLineLength=self.hough_min_len,
                maxLineGap=self.hough_max_gap,
            )

            # --- Debug overlay setup ---
            debug_frame = frame.copy()
            cv.rectangle(
                debug_frame,
                (0, roi_y0), (width - 1, height - 1),
                (255, 0, 0), 2,
            )
            image_center_x = width / 2.0
            cv.line(
                debug_frame,
                (int(image_center_x), 0), (int(image_center_x), height - 1),
                (0, 255, 255), 2,
            )

            if segments is None or len(segments) == 0:
                rospy.logwarn_throttle(2.0, "No Hough segments this frame.")
                self._publish_debug(debug_frame)
                return

            # --- Project each kept segment onto the ROI bottom row ---
            xs_at_bottom = []
            for seg in segments:
                x1, y1, x2, y2 = seg[0]
                dx = float(x2 - x1)
                dy = float(y2 - y1)

                if abs(dy) < 1e-3:
                    # Purely horizontal; reject by slope filter anyway.
                    continue
                slope = dy / dx if abs(dx) > 1e-3 else float('inf')
                if abs(slope) < self.min_abs_slope:
                    continue  # near-horizontal, reject

                # Parametric extrapolation to y = roi_h (bottom of ROI).
                # x(t) = x1 + (dx/dy) * (y - y1)
                x_bottom = x1 + (dx / dy) * (roi_h - y1)
                xs_at_bottom.append(x_bottom)

                # Draw kept segment in green
                cv.line(
                    debug_frame,
                    (int(x1), int(y1 + roi_y0)),
                    (int(x2), int(y2 + roi_y0)),
                    (0, 255, 0), 2,
                )

            if len(xs_at_bottom) == 0:
                rospy.logwarn_throttle(
                    2.0, "All segments rejected by slope filter."
                )
                self._publish_debug(debug_frame)
                return

            # --- Centerline x = robust median of projections ---
            center_line_x = float(np.median(xs_at_bottom))
            center_line_x = max(0.0, min(float(width - 1), center_line_x))

            # --- Normalized lateral error in [-1, 1] ---
            lateral_error = (center_line_x - image_center_x) / image_center_x
            lateral_error = max(-1.0, min(1.0, lateral_error))

            self.line_pub.publish(Float32(lateral_error))

            # --- Debug overlay: draw centerline + readout ---
            cv.line(
                debug_frame,
                (int(center_line_x), roi_y0),
                (int(center_line_x), height - 1),
                (0, 0, 255), 3,
            )
            cv.putText(
                debug_frame,
                "e=%+.2f  segs=%d" % (lateral_error, len(xs_at_bottom)),
                (20, 40),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2,
            )
            self._publish_debug(debug_frame)

            rospy.loginfo_throttle(
                1.0,
                "Line: x=%.1f px  e=%+.3f  segs_kept=%d",
                center_line_x, lateral_error, len(xs_at_bottom),
            )

        except Exception as ex:
            rospy.logerr("Error in image_callback: %s", str(ex))

    # ---------------------------------------------------------------------
    def _publish_debug(self, frame):
        try:
            ok, encoded = cv.imencode('.jpg', frame)
            if not ok:
                return
            out = CompressedImage()
            out.header.stamp = rospy.Time.now()
            out.format = "jpeg"
            out.data = np.array(encoded).tostring()
            self.debug_image_pub.publish(out)
        except Exception as ex:
            rospy.logerr("Error publishing debug image: %s", str(ex))


if __name__ == '__main__':
    node = EdgeLineDetectorNode()
    rospy.spin()
