#!/usr/bin/env python

import rospy
from std_msgs.msg import String
import json

def process_llm_output(llm_output):
    try:
        llm_data = json.loads(llm_output)
        object_name = llm_data.get('name', '')
        object_id = llm_data.get('id', '')

        rospy.loginfo("Received LLM output: name=%s, id=%s", object_name, object_id)

        # Publish to /object_name topic
        object_name_pub.publish(String(object_name))

        # Publish to /object_id topic
        object_id_pub.publish(String(object_id))

    except json.JSONDecodeError as e:
        rospy.logerr("Error decoding LLM output JSON: %s", str(e))

def llm_output_callback(msg):
    llm_output = msg.data
    process_llm_output(llm_output)

def main():
    rospy.init_node('navigation_node')

    global object_name_pub, object_id_pub
    object_name_pub = rospy.Publisher('/object_name', String, queue_size=10)
    object_id_pub = rospy.Publisher('/object_id', String, queue_size=10)

    rospy.Subscriber('/llm_output', String, llm_output_callback)

    rospy.spin()

if __name__ == "__main__":
    main()
