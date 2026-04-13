#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2 as cv
import numpy as np

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32

class EdgeTrajectoryDetectorNode:
    def __init__(self):
        rospy.init_node('edge_trajectory_detector_node')

        # Parametere for kantdeteksjon og baneplanlegging
        self.canny_low = rospy.get_param('~canny_low', 50)
        self.canny_high = rospy.get_param('~canny_high', 150)
        self.estimated_lane_width = rospy.get_param('~lane_width', 300) # Forventet bredde i piksler hvis én kant mangler

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

        rospy.loginfo("Edge Trajectory Detector startet. Venter på bilder...")

    def image_callback(self, msg):
        try:
            # 1. Dekod bildet
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv.imdecode(np_arr, cv.IMREAD_COLOR)

            if frame is None:
                rospy.logwarn("Kunne ikke dekode bildet.")
                return

            height, width = frame.shape[:2]
            center_x = width / 2.0

            # 2. Definer Region of Interest (ROI) - nederste 40% av bildet
            roi_y_start = int(height * 0.6)
            roi = frame[roi_y_start:height, :]

            # 3. Forberedelse og Canny Edge Detection
            gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
            blurred = cv.GaussianBlur(gray, (5, 5), 0)
            edges = cv.Canny(blurred, self.canny_low, self.canny_high)

            debug_frame = frame.copy()
            cv.rectangle(debug_frame, (0, roi_y_start), (width - 1, height - 1), (255, 0, 0), 2)
            cv.line(debug_frame, (int(center_x), 0), (int(center_x), height - 1), (0, 255, 255), 2)

            # 4. Trajectory Planning
            # Finn alle piksler som er detektert som en kant
            ys, xs = np.where(edges > 0)

            if len(xs) < 20:
                rospy.logwarn_throttle(2, "Ingen tydelige kanter funnet. Fortsetter rett frem.")
                self.publish_debug_image(debug_frame)
                return

            # Separer kantene i "venstre for midten" og "høyre for midten"
            left_edges = xs[xs < center_x]
            right_edges = xs[xs >= center_x]

            target_x = center_x # Standardmål

            # Finn gjennomsnittlig x-posisjon for venstre og høyre kant
            if len(left_edges) > 10 and len(right_edges) > 10:
                # Begge kantene er synlige. Målet er nøyaktig i midten.
                left_x = np.mean(left_edges)
                right_x = np.mean(right_edges)
                target_x = (left_x + right_x) / 2.0
                
                cv.line(debug_frame, (int(left_x), roi_y_start), (int(left_x), height - 1), (0, 165, 255), 2)
                cv.line(debug_frame, (int(right_x), roi_y_start), (int(right_x), height - 1), (0, 165, 255), 2)

            elif len(left_edges) > 10:
                # Bare venstre kant er synlig. Planlegg banen ut fra forventet banebredde.
                left_x = np.mean(left_edges)
                target_x = left_x + (self.estimated_lane_width / 2.0)
                cv.line(debug_frame, (int(left_x), roi_y_start), (int(left_x), height - 1), (0, 165, 255), 2)

            elif len(right_edges) > 10:
                # Bare høyre kant er synlig. Planlegg banen ut fra forventet banebredde.
                right_x = np.mean(right_edges)
                target_x = right_x - (self.estimated_lane_width / 2.0)
                cv.line(debug_frame, (int(right_x), roi_y_start), (int(right_x), height - 1), (0, 165, 255), 2)

            # 5. Kalkuler relativ feil for styringen [-100, 100]
            error_px = target_x - center_x
            relative = (error_px / center_x) * 100.0
            relative = max(-100.0, min(100.0, relative))

            # Publiser feilen til kontrolleren
            self.line_pub.publish(Float32(relative))

            # Tegn inn den planlagte trajectoryen (målstreken) i grønt
            cv.line(debug_frame, (int(target_x), roi_y_start), (int(target_x), height - 1), (0, 255, 0), 4)

            # Legg til tekst i debug-bildet
            text = "Target X=%d  Rel=%.1f" % (int(target_x), relative)
            cv.putText(debug_frame, text, (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            self.publish_debug_image(debug_frame)

        except Exception as e:
            rospy.logerr("Error i image_callback: %s", str(e))

    def publish_debug_image(self, frame):
        try:
            success, encoded_image = cv.imencode('.jpg', frame)
            if not success:
                return

            debug_msg = CompressedImage()
            debug_msg.header.stamp = rospy.Time.now()
            debug_msg.format = "jpeg"
            debug_msg.data = np.array(encoded_image).tobytes() # Bytet ut tostring() med tobytes() for nyere numpy

            self.debug_image_pub.publish(debug_msg)

        except Exception as e:
            rospy.logerr("Error under publisering av debug bilde: %s", str(e))


if __name__ == '__main__':
    node = EdgeTrajectoryDetectorNode()
    rospy.spin()
