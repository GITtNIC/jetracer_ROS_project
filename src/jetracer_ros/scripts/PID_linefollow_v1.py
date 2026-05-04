#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v1 PID line-following controller — self-contained (perception + control).

Perception (Szeliski 2022, sections 7.2 and 7.4):
    grayscale -> Gaussian blur -> Canny -> probabilistic Hough
    -> slope filter -> min-segments check -> project to ROI bottom
    -> centerline_x = median(x_projections)
    -> e = (image_center - centerline_x) / image_center    in [-1, 1]

Sign convention (standard negative-feedback with Kp > 0):
    e > 0  <=>  line is to the LEFT of image center
    u > 0  =>   angular.z > 0  =>  left turn  (ROS REP-103 yaw about +z)
    So +e commands a left turn, i.e. toward the line.  No sign flips
    elsewhere in the control law; u = Kp*e does what the textbook says.

Control (Lynch & Park 2017, eq. 11.41):
    u(t) = Kp*e + Ki*integral(e) + Kd*de/dt
    Variable dt, filtered derivative, conditional anti-windup.

Input:  /csi_cam_0/image_raw/compressed       (sensor_msgs/CompressedImage)
Output: /cmd_vel                               (geometry_msgs/Twist)
        /line_detector/debug_image/compressed  (sensor_msgs/CompressedImage)
"""

import rospy
import cv2 as cv
import numpy as np

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32


class PIDLineFollower:
    def __init__(self):
        rospy.init_node('line_controller_node')

        # --- Perception ---
        self.canny_low       = rospy.get_param('~canny_low',       50)
        self.canny_high      = rospy.get_param('~canny_high',      150)
        self.hough_threshold = rospy.get_param('~hough_threshold', 30)
        self.hough_min_len   = rospy.get_param('~hough_min_len',   20)
        self.hough_max_gap   = rospy.get_param('~hough_max_gap',   10)
        self.min_abs_slope   = rospy.get_param('~min_abs_slope',   0.3)
        self.min_segs        = int(rospy.get_param('~min_segs',    3))
        self.roi_top         = rospy.get_param('~roi_top',         0.6)
        self.blur_ksize      = int(rospy.get_param('~blur_ksize',  5))
        if self.blur_ksize % 2 == 0:
            self.blur_ksize += 1

        # --- PID ---
        self.kp            = rospy.get_param('~kp',            1.0)
        self.ki            = rospy.get_param('~ki',            0.0)
        self.kd            = rospy.get_param('~kd',            0.0)
        self.max_angular   = rospy.get_param('~max_angular',   0.6)
        self.d_alpha       = rospy.get_param('~d_alpha',       0.3)
        self.forward_speed = rospy.get_param('~forward_speed', 0.05)
        self.dt_min        = rospy.get_param('~dt_min',        0.005)
        self.dt_max        = rospy.get_param('~dt_max',        0.2)
        self.timeout       = rospy.get_param('~timeout',       0.5)

        # --- PID state ---
        self.e_prev        = 0.0
        self.integral      = 0.0
        self.d_filtered    = 0.0
        self.t_prev        = None
        self.last_img_time = None

        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.err_pub = rospy.Publisher(
            '/line/lateral_error', Float32, queue_size=10,
        )
        self.debug_pub = rospy.Publisher(
            '/line_detector/debug_image/compressed',
            CompressedImage, queue_size=1,
        )
        self.image_sub = rospy.Subscriber(
            '/csi_cam_0/image_raw/compressed',
            CompressedImage,
            self.image_callback,
            queue_size=1,
        )
        self.watchdog_timer = rospy.Timer(rospy.Duration(0.05), self.watchdog_cb)

        rospy.loginfo(
            "line_controller_node started. Kp=%.3f Ki=%.3f Kd=%.3f v=%.2f",
            self.kp, self.ki, self.kd, self.forward_speed,
        )

    # ------------------------------------------------------------------
    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv.imdecode(np_arr, cv.IMREAD_COLOR)
            if frame is None:
                rospy.logwarn("Could not decode compressed image.")
                return

            now = rospy.Time.now()
            self.last_img_time = now

            height, width = frame.shape[:2]
            roi_y0 = int(height * self.roi_top)
            roi    = frame[roi_y0:, :]
            roi_h  = roi.shape[0]

            # Perception pipeline
            gray    = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
            blurred = cv.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)
            edges   = cv.Canny(blurred, self.canny_low, self.canny_high)
            segments = cv.HoughLinesP(
                edges,
                rho=1, theta=np.pi / 180.0,
                threshold=self.hough_threshold,
                minLineLength=self.hough_min_len,
                maxLineGap=self.hough_max_gap,
            )

            # Debug overlay base
            debug = frame.copy()
            image_center_x = width / 2.0
            cv.rectangle(debug, (0, roi_y0), (width-1, height-1), (255, 0, 0), 2)
            cv.line(debug,
                    (int(image_center_x), 0), (int(image_center_x), height-1),
                    (0, 255, 255), 2)

            if segments is None or len(segments) == 0:
                rospy.logwarn_throttle(2.0, "No Hough segments.")
                self._publish_debug(debug)
                return

            xs_at_bottom = []
            for seg in segments:
                x1, y1, x2, y2 = seg[0]
                dx, dy = float(x2 - x1), float(y2 - y1)
                if abs(dy) < 1e-3:
                    continue
                slope = dy / dx if abs(dx) > 1e-3 else float('inf')
                if abs(slope) < self.min_abs_slope:
                    continue
                x_bottom = x1 + (dx / dy) * (roi_h - y1)
                xs_at_bottom.append(x_bottom)
                cv.line(debug,
                        (int(x1), int(y1 + roi_y0)),
                        (int(x2), int(y2 + roi_y0)),
                        (0, 255, 0), 2)

            if len(xs_at_bottom) < self.min_segs:
                rospy.logwarn_throttle(
                    2.0,
                    "Only %d segments kept (need >= %d) - ignoring frame.",
                    len(xs_at_bottom), self.min_segs,
                )
                self._publish_debug(debug)
                return

            center_x = float(np.median(xs_at_bottom))
            center_x = max(0.0, min(float(width - 1), center_x))
            # +e = line LEFT of image center (see module docstring).
            e = max(-1.0, min(1.0, (image_center_x - center_x) / image_center_x))
            self.err_pub.publish(Float32(data=e))  # observability hook

            # PID — need two samples
            if self.t_prev is None:
                self.t_prev = now
                self.e_prev = e
                self._publish_debug(debug)
                return

            dt = (now - self.t_prev).to_sec()
            if dt < self.dt_min:
                self._publish_debug(debug)
                return
            dt = min(dt, self.dt_max)

            p_term = self.kp * e

            d_raw = (e - self.e_prev) / dt
            self.d_filtered = (self.d_alpha * d_raw
                               + (1.0 - self.d_alpha) * self.d_filtered)
            d_term = self.kd * self.d_filtered

            tentative_integral = self.integral + e * dt
            u_unsat = p_term + self.ki * tentative_integral + d_term
            u = max(-self.max_angular, min(self.max_angular, u_unsat))

            pushing_up   = (u_unsat >  self.max_angular) and (e > 0.0)
            pushing_down = (u_unsat < -self.max_angular) and (e < 0.0)
            if not (pushing_up or pushing_down):
                self.integral = tentative_integral

            cmd = Twist()
            cmd.linear.x  = self.forward_speed
            cmd.angular.z = u
            self.cmd_pub.publish(cmd)

            self.e_prev = e
            self.t_prev = now

            # Finish debug overlay
            cv.line(debug,
                    (int(center_x), roi_y0), (int(center_x), height-1),
                    (0, 0, 255), 3)
            cv.putText(debug,
                       "e=%+.2f  segs=%d" % (e, len(xs_at_bottom)),
                       (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            self._publish_debug(debug)

            rospy.loginfo_throttle(
                0.5,
                "e=%+.3f P=%+.3f I=%+.3f D=%+.3f u=%+.3f dt=%.3f segs=%d",
                e, p_term, self.ki * self.integral, d_term, u, dt,
                len(xs_at_bottom),
            )

        except Exception as ex:
            rospy.logerr("Error in image_callback: %s", str(ex))

    # ------------------------------------------------------------------
    def _publish_debug(self, frame):
        try:
            ok, encoded = cv.imencode('.jpg', frame)
            if not ok:
                return
            out = CompressedImage()
            out.header.stamp = rospy.Time.now()
            out.format = "jpeg"
            out.data = np.array(encoded).tostring()
            self.debug_pub.publish(out)
        except Exception as ex:
            rospy.logerr("Error publishing debug image: %s", str(ex))

    # ------------------------------------------------------------------
    def watchdog_cb(self, _event):
        if self.last_img_time is None:
            return
        if (rospy.Time.now() - self.last_img_time).to_sec() > self.timeout:
            self.cmd_pub.publish(Twist())
            rospy.logwarn_throttle(
                1.0, "No camera image for > %.2fs - zero Twist.", self.timeout,
            )

    # ------------------------------------------------------------------
    def shutdown(self):
        self.cmd_pub.publish(Twist())
        rospy.loginfo("line_controller_node stopping. Zero Twist published.")


if __name__ == '__main__':
    node = PIDLineFollower()
    rospy.on_shutdown(node.shutdown)
    rospy.spin()
