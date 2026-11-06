"""Generate a self-contained Gazebo Harmonic world, without external models."""
from pathlib import Path
import xml.etree.ElementTree as ET
from arena import OBSTACLES, export_map


def robot(name, x, y, color):
    body = f'''<model name="{name}"><pose>{x} {y} 0.16 0 0 0</pose>
    <link name="base"><inertial><mass>3</mass><inertia><ixx>0.05</ixx><iyy>0.08</iyy><izz>0.1</izz></inertia></inertial>
    <collision name="body"><geometry><box><size>0.5 0.32 0.16</size></box></geometry></collision>
    <visual name="body"><geometry><box><size>0.5 0.32 0.16</size></box></geometry><material><ambient>{color} 1</ambient><diffuse>{color} 1</diffuse></material></visual>
    <visual name="nose"><pose>0.24 0 0.09 0 0 0</pose><geometry><box><size>0.07 0.15 0.03</size></box></geometry><material><ambient>1 1 1 1</ambient></material></visual></link>'''
    for side, wy in [('left', 0.21), ('right', -0.21)]:
        body += f'''<link name="{side}_wheel"><pose>0 {wy} -0.04 1.57079632679 0 0</pose>
        <inertial><mass>0.3</mass><inertia><ixx>0.001</ixx><iyy>0.001</iyy><izz>0.002</izz></inertia></inertial>
        <collision name="wheel"><geometry><cylinder><radius>0.12</radius><length>0.05</length></cylinder></geometry></collision>
        <visual name="wheel"><geometry><cylinder><radius>0.12</radius><length>0.05</length></cylinder></geometry><material><ambient>0.08 0.08 0.08 1</ambient></material></visual></link>
        <joint name="{side}_joint" type="revolute"><parent>base</parent><child>{side}_wheel</child><axis><xyz expressed_in="__model__">0 1 0</xyz><limit><lower>-1e16</lower><upper>1e16</upper></limit></axis></joint>'''
    for i, cx in enumerate([-0.20, 0.20]):
        body += f'''<link name="caster{i}"><pose>{cx} 0 -0.12 0 0 0</pose>
        <inertial><mass>0.05</mass><inertia><ixx>0.0001</ixx><iyy>0.0001</iyy><izz>0.0001</izz></inertia></inertial>
        <collision name="ball"><geometry><sphere><radius>0.04</radius></sphere></geometry><surface><friction><ode><mu>0</mu><mu2>0</mu2></ode></friction></surface></collision>
        <visual name="ball"><geometry><sphere><radius>0.04</radius></sphere></geometry></visual></link>
        <joint name="caster_joint{i}" type="fixed"><parent>base</parent><child>caster{i}</child></joint>'''
    return body + f'''<plugin filename="gz-sim-diff-drive-system" name="gz::sim::systems::DiffDrive">
    <left_joint>left_joint</left_joint><right_joint>right_joint</right_joint><wheel_separation>0.42</wheel_separation><wheel_radius>0.12</wheel_radius>
    <topic>/model/{name}/cmd_vel</topic><odom_topic>/model/{name}/odometry</odom_topic><odom_publish_frequency>30</odom_publish_frequency>
    </plugin></model>'''


def generate():
    world = '''<?xml version="1.0"?><sdf version="1.9"><world name="formation">
    <physics name="physics" type="ignored"><max_step_size>0.001</max_step_size><real_time_factor>1</real_time_factor></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <gui fullscreen="false"><plugin filename="MinimalScene" name="3D View"><gz-gui><title>Three robot RL formation</title><property type="string" key="state">docked</property></gz-gui><engine>ogre2</engine><scene>scene</scene><ambient_light>0.5 0.5 0.5</ambient_light><background_color>0.15 0.18 0.22</background_color><camera_pose>-7 -9 9 0 0.65 0.85</camera_pose></plugin><plugin filename="GzSceneManager" name="Scene Manager"/><plugin filename="InteractiveViewControl" name="View control"/></gui>
    <light name="sun" type="directional"><pose>0 0 10 0 0 0</pose><diffuse>0.9 0.9 0.9 1</diffuse><specular>0.2 0.2 0.2 1</specular><direction>-0.5 0.2 -1</direction></light>
    <model name="ground"><static>true</static><link name="ground"><collision name="ground"><geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry></collision><visual name="ground"><geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry><material><ambient>0.45 0.48 0.5 1</ambient><diffuse>0.45 0.48 0.5 1</diffuse></material></visual></link></model>'''
    world += robot('leader', 0, 0, '1 0.7 0.1')
    world += robot('follower_1', -1.5, 0.9, '0.1 0.6 1')
    world += robot('follower_2', -1.5, -0.9, '0.6 0.2 1')
    path = Path(__file__).resolve().parents[1] / 'worlds' / 'formation.sdf'
    root = ET.fromstring(world + '</world></sdf>')
    scene = root.find('world')
    ET.SubElement(scene, 'plugin', filename='gz-sim-sensors-system', name='gz::sim::systems::Sensors').append(ET.fromstring('<render_engine>ogre2</render_engine>'))
    ET.SubElement(scene, 'plugin', filename='gz-sim-contact-system', name='gz::sim::systems::Contact')
    leader = scene.find("model[@name='leader']")
    # Scale every physical dimension, wheel spacing, inertia and mass consistently.
    scale = 1.7
    leader.find('pose').text = '0 0 0.272 0 0 0'
    for link in leader.findall('link'):
        for pose in link.findall('.//pose'):
            values = list(map(float, pose.text.split()))
            pose.text = ' '.join(map(str, [v*scale for v in values[:3]]+values[3:]))
        for size in link.findall('.//size'):
            size.text = ' '.join(str(float(v)*scale) for v in size.text.split())
        for tag in ('radius', 'length'):
            for element in link.findall('.//'+tag):
                element.text = str(float(element.text)*scale)
        link.find('inertial/mass').text = str(float(link.find('inertial/mass').text)*scale**3)
        for inertia in link.findall('inertial/inertia/*'):
            inertia.text = str(float(inertia.text)*scale**5)
    plugin = leader.find('plugin')
    plugin.find('wheel_radius').text = str(.12*scale)
    plugin.find('wheel_separation').text = str(.42*scale)
    base = leader.find("link[@name='base']")
    base.append(ET.fromstring('''<sensor name="lidar" type="gpu_lidar"><pose>0 0 0.32 0 0 0</pose><topic>/leader/scan</topic><update_rate>10</update_rate><always_on>true</always_on><lidar><scan><horizontal><samples>360</samples><resolution>1</resolution><min_angle>-3.14159265</min_angle><max_angle>3.14159265</max_angle></horizontal></scan><range><min>0.12</min><max>12</max><resolution>0.02</resolution></range></lidar></sensor>'''))
    base.append(ET.fromstring('''<sensor name="camera" type="camera"><pose>0.40 0 0.22 0 0 0</pose><topic>/leader/camera</topic><update_rate>5</update_rate><always_on>true</always_on><camera><horizontal_fov>1.4</horizontal_fov><image><width>320</width><height>240</height><format>R8G8B8</format></image><clip><near>0.05</near><far>25</far></clip></camera></sensor>'''))
    base.append(ET.fromstring('''<visual name="lidar_housing"><pose>0 0 0.25 0 0 0</pose><geometry><cylinder><radius>0.10</radius><length>0.12</length></cylinder></geometry><material><ambient>0.1 0.1 0.1 1</ambient></material></visual>'''))
    for name in ('leader', 'follower_1', 'follower_2'):
        scene.find(f"model[@name='{name}']/link[@name='base']").append(ET.fromstring(f'''<sensor name="bumper" type="contact"><topic>/model/{name}/contacts</topic><update_rate>20</update_rate><always_on>true</always_on><contact><collision>body</collision></contact></sensor>'''))
    blocks = list(OBSTACLES) + [(7, -7.2, 24, .4, 1), (7, 7.2, 24, .4, 1), (-5.2, 0, .4, 14, 1), (19.2, 0, .4, 14, 1)]
    for i, (x, y, w, d, h) in enumerate(blocks):
        scene.append(ET.fromstring(f'''<model name="obstacle_{i}"><static>true</static><pose>{x} {y} {h/2} 0 0 0</pose><link name="block"><collision name="block"><geometry><box><size>{w} {d} {h}</size></box></geometry></collision><visual name="block"><geometry><box><size>{w} {d} {h}</size></box></geometry><material><ambient>0.8 0.18 0.08 1</ambient><diffuse>0.8 0.18 0.08 1</diffuse></material></visual></link></model>'''))
    scene.find('gui/plugin/camera_pose').text = '3 -16 18 0 0.9 1.3'
    for name in ('leader', 'follower_1', 'follower_2'):
        model = scene.find(f"model[@name='{name}']")
        model.find('plugin/odom_topic').text = f'/model/{name}/wheel_odometry'
        model.append(ET.fromstring(f'''<plugin filename="gz-sim-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher"><odom_topic>/model/{name}/odometry</odom_topic><odom_frame>world</odom_frame><dimensions>3</dimensions><odom_publish_frequency>30</odom_publish_frequency></plugin>'''))
    for visual in root.findall(".//visual"):
        visual.set("name", visual.get("name")+"_visual")
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding='unicode', xml_declaration=True)
    export_map()
    return path


if __name__ == '__main__':
    print(generate())
