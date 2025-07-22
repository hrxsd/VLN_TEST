#!/usr/bin/env python3
import yaml
import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import String
from openai import OpenAI
import json

# Initialize OpenAI API (replace 'your-api-key' with your actual API key)
api_key = "sk-hLnEnofYFMZJUX1zF52dF59097A74323A12c3e407176Bc20"
api_base = "https://pro.aiskt.com/v1"

def load_semantic_map(file_path):
    with open(file_path, 'r') as file:
        semantic_map = yaml.safe_load(file)
    return semantic_map['semantic_info']

def chatgpt_query(system_prompt, user_prompt):    
    client = OpenAI(api_key=api_key, base_url=api_base)
    completion = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}]
    )
    result = json.loads(completion.choices[0].message.content.strip())
    return result

def save_to_yaml(data, filename='data.yaml'):
    with open(filename, 'w', encoding='utf-8') as file:
        yaml.dump(data, file, allow_unicode=True, default_flow_style=False)

def load_from_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def save_to_json(data, filename='/home/nuc1003a/ARTS_TEST/src/hdl_localization/json/data.json'):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def read_data_json(file_path='/home/nuc1003a/ARTS_TEST/src/hdl_localization/json/data.json'):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

class SemanticNavigator:
    def __init__(self, semantic_info):
        self.semantic_info = semantic_info
        self.pub_name = rospy.Publisher('/object_name', String, queue_size=10, latch=True)
        self.pub_id = rospy.Publisher('/object_id', String, queue_size=10, latch=True)
        self.pub_responese = rospy.Publisher('response', String, queue_size=10, latch=True)
        rospy.init_node('semantic_navigator', anonymous=True)
        self.rate = rospy.Rate(1)  # 1 Hz
        rospy.Subscriber("user_commands", String, self.callback)

    def callback(self, msg):
        user_input = msg.data
        system_prompt = (
            "Now imagine that you are a home service robot, "
            "and you now have the ability to output the name and id of a target object based on the user's linguistic commands and a yaml file that contains the semantics of the environment."
            + str(self.semantic_info) + "is the semantic information contained in the environment, which:"
            "- id = 1----object's id"
            "- name = 'bed'----object's name"
            "- position = {x: 3, y: 4}----location of the object on the map"
            "- size = {x: 1, y: 2}----two-dimensional dimensions of the object"
            "Please understand the semantic information provided, "
            "and according to the instruction provided by the user, output the name and id of the target object (output JSON data structure)."
            """For example, if the user's instruction is: “Go to the largest table”, 
            then you need to find the table with the largest size based on the information provided and output it in this format:
            {
                "name": "table",
                "id": "1",
                "response": "Sure, I will go to the largest table."
            }
            Please generate a JSON-formatted response according to the above requirements, and reply only to the JSON itself, not to any other content.
            """
        )
        user_prompt = "The object I want to find is " + user_input
        target_name = chatgpt_query(system_prompt, user_prompt)
        save_to_json(target_name)
        data = read_data_json()
        name = data['name']
        id = data['id']
        response = data['response']
        # rospy.loginfo(f"Response: {response}")
        print(response)
        self.pub_name.publish(name)
        self.pub_id.publish(id)
        self.pub_responese.publish(response)
        self.prompt_user_input()

    def prompt_user_input(self):
        user_input = input("Enter your command: \n")
        if user_input:
            self.pub_user_command.publish(user_input)

    def run(self):
        self.pub_user_command = rospy.Publisher('user_commands', String, queue_size=10)
        rospy.loginfo("Semantic Navigator is running... Can I help you?")
        self.prompt_user_input()
        rospy.spin()

def main():
    semantic_info = load_semantic_map('/home/nuc1003a/ARTS_TEST/src/hdl_localization/config/map/semantic_map.yaml')
    navigator = SemanticNavigator(semantic_info)
    navigator.run()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
