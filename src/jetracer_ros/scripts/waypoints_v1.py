#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v1 waypoint navigator (go-to-goal, Ackermann).

Given a hard-coded (or rosparam) list of 2D goal points in the odom frame,
drive the car through them in order using a continuously-blended
proportional controller:

        linear.x  = clamp( k_v * dist_to_goal,     0, v_max )
        angular.z = clamp( k_w * heading_error,  -w_max, w_max )

where
        dist_to_goal    = sqrt( dx^2 + dy^2 )
        heading_error   = wrap_pi( atan2(dy, dx) - yaw )
        (dx, dy)        = (x_goal - x, y_goal - y)

A waypoint is considered "reached" when dist_to_goal < reach_threshold.
On reach, the next waypoint is popped; when the list is empty, the node
publishes zero Twist and stops.

Why continuous blending (not turn-then-drive):
  Ackermann cars cannot spin in place. At heading errors near +/- pi the
  car simply draws a wider curve; this is physically correct and keeps
  control smooth. It may overshoot corners by a few tens of centimetres;
  that is the expected v1 behaviour and is what the waypoint drift test
  is designed to characterise.

Frame:
  All waypoints are in the "odom" frame as published in /odom.
  Convention: park the car with its rear axle on the (0,0) mark and the
  chassis pointing along +x BEFORE launching jetracer.launch, so that
  the hard-coded square maps to the tape grid on the floor.

Safety features:
  * Watchdog: if /odom stops arriving for > odom_timeout seconds, zero
    Twist is published until it comes back. This protects against EKF
    or network stalls.
  * Mission timeout: if the whole mission has not completed in
    mission_timeout seconds, the node aborts and stops the car.
  * Shutdown hook: on rospy shutdown, publish zero Twist.

Logging:
  Every cycle logs (throttled) the active waypoint index, distance to
  goal, heading error, and the commanded (v, w). A rosbag of /odom and
  /cmd_vel captures the full trajectory for the drift plot.

Subscribes: /odom      (nav_msgs/Odometry)
Publishes:  /cmd_vel   (geometry_msgs/Twist)

Theory: LaValle (2006) Ch. 15.4-15.5 (nonholonomic motion planning);
        Barfoot (2026) for drift characterisation on /odom feedback.
"""

import math

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion


def wrap_pi(angle):
    """Wrap an angle in radians to (-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    return angle


class WaypointNavigator(object):
    def __init__(self):
        rospy.init_node('waypoint_nav_node')

        # --- Waypoints (odom frame, metres). Default = 1 m square. ---
        # Override at launch with: <rosparam param="~waypoints">[[1,0],[1,1],[0,1],[0,0]]</rosparam>
        default_wps = [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]
        wp_param = rospy.get_param('~waypoints', default_wps)
        self.waypoints = [(float(p[0]), float(p[1])) for p in wp_param]
        self.wp_idx = 0

        # --- Controller gains ---
        self.k_v = rospy.get_param('~k_v', 0.5)
        self.k_w = rospy.get_param('~k_w', 1.5)

        # --- Output limits ---
        self.v_max = rospy.get_param('~v_max', 0.15)   # m/s
        self.w_max = rospy.get_param('~w_max', 0.6)    # rad/s

        # --- Reach threshold (metres) ---
        self.reach_thresh = rospy.get_param('~reach_thresh', 0.30)

        # --- Safety ---
        self.odom_timeout = rospy.get_param('~odom_timeout', 0.5)
        self.mission_timeout = rospy.get_param('~mission_timeout', 180.0)

        # --- State ---
        self.last_odom_time = None
        self.done = False
        self.t_start = rospy.Time.now()

        # --- ROS I/O ---
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber(
            '/odom', Odometry, self.odom_callback, queue_size=10
        )
        self.watchdog_timer = rospy.Timer(
            rospy.Duration(0.1), self.watchdog_cb
        )

        rospy.loginfo(
            "waypoint_nav_node started: %d waypoints, k_v=%.2f k_w=%.2f "
            "v_max=%.2f w_max=%.2f reach=%.2f",
            len(self.waypoints), self.k_v, self.k_w,
            self.v_max, self.w_max, self.reach_thresh,
        )
        for i, (wx, wy) in enumerate(self.waypoints):
            rospy.loginfo("  wp[%d] = (%+.2f, %+.2f)", i, wx, wy)

    # ---------------------------------------------------------------------
    def odom_callback(self, msg):
        now = rospy.Time.now()
        self.last_odom_time = now

        if self.done:
            return

        # Mission timeout
        if (now - self.t_start).to_sec() > self.mission_timeout:
            rospy.logwarn("Mission timeout reached. Aborting.")
            self._finish()
            return

        # Extract pose
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        (_, _, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])

        # Active waypoint
        if self.wp_idx >= len(self.waypoints):
            self._finish()
            return
        gx, gy = self.waypoints[self.wp_idx]

        # Errors
        dx = gx - x
        dy = gy - y
        dist = math.hypot(dx, dy)
        heading_err = wrap_pi(math.atan2(dy, dx) - yaw)

        # Reach check
        if dist < self.reach_thresh:
            rospy.loginfo(
                "Reached wp[%d] = (%+.2f, %+.2f). dist=%.2f",
                self.wp_idx, gx, gy, dist,
            )
            self.wp_idx += 1
            if self.wp_idx >= len(self.waypoints):
                rospy.loginfo("All waypoints reached. Stopping.")
                self._finish()
            return

        # Proportional control, saturated
        v = max(0.0, min(self.v_max, self.k_v * dist))
        w = max(-self.w_max, min(self.w_max, self.k_w * heading_err))

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.cmd_pub.publish(cmd)

        rospy.loginfo_throttle(
            0.5,
            "wp[%d] d=%.2f h_err=%+.2f rad v=%.2f w=%+.2f",
            self.wp_idx, dist, heading_err, v, w,
        )

    # ---------------------------------------------------------------------
    def watchdog_cb(self, _event):
        if self.done or self.last_odom_time is None:
            return
        if (rospy.Time.now() - self.last_odom_time).to_sec() > self.odom_timeout:
            self.cmd_pub.publish(Twist())
            rospy.logwarn_throttle(
                1.0,
                "No /odom for > %.2fs - zero Twist.", self.odom_timeout,
            )

    # ---------------------------------------------------------------------
    def _finish(self):
        self.done = True
        self.cmd_pub.publish(Twist())

    # ---------------------------------------------------------------------
    def shutdown(self):
        self.cmd_pub.publish(Twist())
        rospy.loginfo("waypoint_nav_node stopping. Zero Twist published.")


if __name__ == '__main__':
    node = WaypointNavigator()
    rospy.on_shutdown(node.shutdown)
    rospy.spin()
