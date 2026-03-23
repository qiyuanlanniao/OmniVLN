import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import json
import os

def update_json_file(x, y, z):
    """
    修改本地 JSON 文件中 ID 为 -1 的物体坐标
    """
    json_path = "/home/iot/hm/ros2_ws/maps/scene_objects.json"

    try:
        # 1. 读取原始数据
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 2. 查找并更新 ID 为 -1 的项
        found = False
        for obj in data:
            if obj.get("id") == -1:
                obj["position"] = [float(x), float(y), float(z)]
                found = True
                break

        # 3. 写回文件
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        print(f"📝 Moved to: [{x:.2f}, {y:.2f}, {z:.2f}]")

    except Exception as e:
        print(f"❌ {e}")

def teleport(x, y, z):
    # --- 1. 修改 ROS 2 实时位姿 ---
    rclpy.init()
    node = Node('teleporter')
    # 使用 latched 模式发布，确保后续节点启动也能收到
    pub = node.create_publisher(PoseStamped, '/mavros/vision_pose/pose', 10)
    
    msg = PoseStamped()
    msg.header.frame_id = "map"
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.pose.position.x = float(x)
    msg.pose.position.y = float(y)
    msg.pose.position.z = float(z)
    msg.pose.orientation.w = 1.0
    
    # 连续发布确保成功
    for _ in range(5):
        pub.publish(msg)

    # --- 2. 修改持久化 JSON 文件 ---
    update_json_file(x, y, z)

    rclpy.shutdown()

if __name__ == "__main__":
    # 目标坐标
    teleport(-4.015016257762909,
            8.39906919002533,
            0.03786160983145239)

    # teleport(2.568601703643799,
    #         0.9642015099525456,
    #         -0.1938526779413224)
    