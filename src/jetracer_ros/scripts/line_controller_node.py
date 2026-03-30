#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy

from std_msgs.msg import Float32
from geometry_msgs.msg import Twist


class LineControllerNode:
    def __init__(self):
        rospy.init_node('line_controller_node')

        self.kp = rospy.get_param('~kp', 1.0)
        self.max_angular = rospy.get_param('~max_angular', 0.6)
        self.forward_speed = rospy.get_param('~forward_speed', 0.05)
        self.deadband = rospy.get_param('~deadband', 8.0)

        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.line_sub = rospy.Subscriber(
            '/relative_line_hpos',
            Float32,
            self.line_callback,
            queue_size=10
        )

        rospy.loginfo("line_controller_node started.")

    def line_callback(self, msg):
        try:
            line_hpos = msg.data

            # Deadband around image center
            if abs(line_hpos) < self.deadband:
                line_hpos = 0.0

            # Convert from [-100, 100] to about [-1, 1]
            error = line_hpos / 100.0

            angular_z = self.kp * error

            # Clamp steering command
            if angular_z > self.max_angular:
                angular_z = self.max_angular
            elif angular_z < -self.max_angular:
                angular_z = -self.max_angular

            cmd = Twist()
            cmd.linear.x = self.forward_speed
            cmd.angular.z = angular_z

            self.cmd_pub.publish(cmd)

            rospy.loginfo_throttle(
                1,
                "line_hpos=%.2f | error=%.2f | linear=%.2f | angular=%.2f",
                msg.data,
                error,
                cmd.linear.x,
                cmd.angular.z
            )

        except Exception as e:
            rospy.logerr("Error in line_callback: %s", str(e))

    def shutdown(self):
        cmd = Twist()
        self.cmd_pub.publish(cmd)
        rospy.loginfo("line_controller_node stopped. Published zero Twist.")


if __name__ == '__main__':
    node = LineControllerNode()
    rospy.on_shutdown(node.shutdown)
    rospy.spin()
