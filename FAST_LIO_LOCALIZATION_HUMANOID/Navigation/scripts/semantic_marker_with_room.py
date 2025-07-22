#!/usr/bin/env python3

import rospy
import yaml
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker
from geometry_msgs.msg import Point, Quaternion

class InteractiveSemanticMarker:
    def __init__(self):
        self.server = InteractiveMarkerServer("semantic_marker_server")
        self.semantic_info_file = '/home/nuc1003a/ARTS_TEST/src/hdl_localization/config/map/semantic_map.yaml' 
        self.semantic_info, self.rooms = self.load_semantic_info()

        for semantic in self.semantic_info:
            self.create_interactive_marker(semantic)
        
        for room_name, room_info in self.rooms.items():
            self.create_room_boundary(room_info)
        
        self.server.applyChanges()

    def load_semantic_info(self):
        with open(self.semantic_info_file, 'r') as file:
            data = yaml.safe_load(file)
            return data['semantic_info'], data['rooms']

    def save_semantic_info(self):
        with open(self.semantic_info_file, 'w') as file:
            yaml.dump({'semantic_info': self.semantic_info, 'rooms': self.rooms}, file)

    def process_feedback(self, feedback):
        for semantic in self.semantic_info:
            if str(semantic['id']) == feedback.marker_name:
                semantic['position']['x'] = feedback.pose.position.x
                semantic['position']['y'] = feedback.pose.position.y
                # Ensure orientation is consistent with map
                feedback.pose.orientation = Quaternion(0, 0, 0, 1)
                self.save_semantic_info()
                self.server.setPose(feedback.marker_name, feedback.pose)
                break
        self.server.applyChanges()

    def create_interactive_marker(self, semantic):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "map"
        int_marker.name = str(semantic['id'])
        int_marker.description = semantic['name']
        int_marker.pose.position.x = semantic['position']['x']
        int_marker.pose.position.y = semantic['position']['y']
        # Set default orientation to be aligned with the map
        int_marker.pose.orientation = Quaternion(0, 0, 0, 1)

        # Create markers for the edges of the box
        edge_thickness = 0.01
        size_x = semantic.get('size', {}).get('x', 0.5)
        size_y = semantic.get('size', {}).get('y', 0.5)

        def create_edge_marker(pos_x, pos_y, scale_x, scale_y):
            marker = Marker()
            marker.type = Marker.CUBE
            marker.scale.x = scale_x
            marker.scale.y = scale_y
            marker.scale.z = edge_thickness
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.pose.position.x = pos_x
            marker.pose.position.y = pos_y
            marker.pose.position.z = 0
            return marker

        top_edge = create_edge_marker(0, size_y / 2 - edge_thickness / 2, size_x, edge_thickness)
        bottom_edge = create_edge_marker(0, -size_y / 2 + edge_thickness / 2, size_x, edge_thickness)
        left_edge = create_edge_marker(-size_x / 2 + edge_thickness / 2, 0, edge_thickness, size_y)
        right_edge = create_edge_marker(size_x / 2 - edge_thickness / 2, 0, edge_thickness, size_y)

        # Create a marker for the ID
        id_text_marker = Marker()
        id_text_marker.type = Marker.TEXT_VIEW_FACING
        id_text_marker.scale.z = 0.3
        id_text_marker.color.a = 1.0
        id_text_marker.color.r = 0.5
        id_text_marker.color.g = 0.5
        id_text_marker.color.b = 1.0
        id_text_marker.text = "ID: {}".format(semantic['id'])
        id_text_marker.pose.position.x = 0
        id_text_marker.pose.position.y = 0.2
        id_text_marker.pose.position.z = 0.3  # Position above the frame

        # Create a marker for the text
        name_text_marker = Marker()
        name_text_marker.type = Marker.TEXT_VIEW_FACING
        name_text_marker.scale.z = 0.3
        name_text_marker.color.a = 1.0
        name_text_marker.color.r = 1.0
        name_text_marker.color.g = 0.0
        name_text_marker.color.b = 0.0
        name_text_marker.text = semantic['name']
        name_text_marker.pose.position.z = 0.3  # Position above the frame

        # Create a control that contains all markers
        control = InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append(top_edge)
        control.markers.append(bottom_edge)
        control.markers.append(left_edge)
        control.markers.append(right_edge)
        control.markers.append(id_text_marker)
        control.markers.append(name_text_marker)
        int_marker.controls.append(control)

        self.server.insert(int_marker, self.process_feedback)

    def create_room_boundary(self, room_info):
        # Define softer colors for different room types
        room_colors = {
            "living_room": (0.7, 0.7, 0.7),  # Light gray
            "kitchen": (0.6, 0.8, 0.6),      # Light green
            "resturant": (0.6, 0.6, 0.8),    # Light blue
            "bedroom": (0.8, 0.8, 0.6)       # Light yellow
        }

        # Get color for the current room
        color = room_colors.get(room_info['name'], (0.75, 0.75, 0.75))  # Default to light gray

        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "map"
        int_marker.name = "room_" + str(room_info['id'])
        int_marker.description = room_info['name']
        
        # Create the boundary marker
        boundary_marker = Marker()
        boundary_marker.type = Marker.LINE_STRIP
        boundary_marker.scale.x = 0.05
        boundary_marker.color.a = 1.0
        boundary_marker.color.r = color[0]
        boundary_marker.color.g = color[1]
        boundary_marker.color.b = color[2]

        corners = room_info['corners']
        points = [Point(x=corner['x'], y=corner['y'], z=0) for corner in corners]
        points.append(points[0])  # Close the loop by adding the first point at the end
        boundary_marker.points = points

        # Create the room name text marker
        # room_text_marker = Marker()
        # room_text_marker.type = Marker.TEXT_VIEW_FACING
        # room_text_marker.scale.z = 0.5
        # room_text_marker.color.a = 1.0
        # room_text_marker.color.r = color[0]
        # room_text_marker.color.g = color[1]
        # room_text_marker.color.b = color[2]
        # room_text_marker.text = room_info['name']
        # room_text_marker.pose.position.x = sum(p.x for p in points) / len(points)
        # room_text_marker.pose.position.y = sum(p.y for p in points) / len(points)
        # room_text_marker.pose.position.z = 1.0

        # Create a control that contains all markers
        control = InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append(boundary_marker)
        # control.markers.append(room_text_marker)
        int_marker.controls.append(control)

        self.server.insert(int_marker)

def main():
    rospy.init_node('interactive_semantic_marker')
    ism = InteractiveSemanticMarker()
    rospy.spin()

if __name__ == "__main__":
    main()
