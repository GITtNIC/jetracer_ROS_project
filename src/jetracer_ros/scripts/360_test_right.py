#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""360_test_right.py

Drive the JetRacer in an unbounded right circle at full steering lock for
min-turning-radius calibration. Bag recording is handled by the launch file
(see launch/360_test_right.launch), matching the line_follow_v1 convention.

Run:   roslaunch jetracer 360_test_right.launch
Stop:  Ctrl+C  -- publishes zero cmd_vel before exit.
"""

import rospy
from geometry_msgs.msg import Twist


def main():
    rospy.init_node('test_360_right', anonymous=False)

    linear_speed  = rospy.get_param('~linear_speed',  0.15)   # m/s
    angular_speed = rospy.get_param('~angular_speed', -1.5)   # rad/s, negative = right
    publish_hz    = rospy.get_param('~publish_hz',    20)

    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    def shutdown():
        rospy.loginfo('stopping car...')
        stop = Twist()
        for _ in range(10):
            pub.publish(stop)
            rospy.sleep(0.05)

    rospy.on_shutdown(shutdown)

    rospy.sleep(1.0)  # let publishers/subscribers wire up

    cmd = Twist()
    cmd.linear.x  = linear_speed
    cmd.angular.z = angular_speed

    rospy.loginfo('right circle: v=%.2f m/s, w=%.2f rad/s  --  Ctrl+C to stop',
                  linear_speed, angular_speed)

    rate = rospy.Rate(publish_hz)
    while not rospy.is_shutdown():
        pub.publish(cmd)
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
