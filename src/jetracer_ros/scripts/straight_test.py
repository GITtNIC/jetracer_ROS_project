#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""straight_test.py

Drive the JetRacer forward at constant linear speed for a fixed duration,
then stop. Used to verify /odom linear scale (compare /odom reported
distance to a tape measure on the floor).

Run:    roslaunch jetracer straight_test.launch
        roslaunch jetracer straight_test.launch duration:=10.0
        roslaunch jetracer straight_test.launch linear_speed:=0.10
Stop:   completes on its own; Ctrl+C also works.
"""

import rospy
from geometry_msgs.msg import Twist


def main():
    rospy.init_node('straight_test', anonymous=False)

    linear_speed = rospy.get_param('~linear_speed', 0.15)   # m/s
    duration     = rospy.get_param('~duration',     7.0)    # s
    publish_hz   = rospy.get_param('~publish_hz',   20)

    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    def shutdown():
        rospy.loginfo('stopping car...')
        stop = Twist()
        for _ in range(10):
            pub.publish(stop)
            rospy.sleep(0.05)

    rospy.on_shutdown(shutdown)

    rospy.sleep(1.0)  # let publishers wire up

    cmd = Twist()
    cmd.linear.x  = linear_speed
    cmd.angular.z = 0.0

    rospy.loginfo('straight: v=%.2f m/s for %.1f s', linear_speed, duration)

    rate     = rospy.Rate(publish_hz)
    t_start  = rospy.Time.now()
    t_end    = t_start + rospy.Duration.from_sec(duration)
    while not rospy.is_shutdown() and rospy.Time.now() < t_end:
        pub.publish(cmd)
        rate.sleep()

    rospy.loginfo('done -- stopping')
    # shutdown() runs via on_shutdown when rospy exits;
    # publish a few stops here too for snappier braking
    stop = Twist()
    for _ in range(5):
        pub.publish(stop)
        rospy.sleep(0.05)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
