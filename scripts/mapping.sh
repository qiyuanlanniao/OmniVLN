#!/bin/bash

# --- 路径配置 ---
WS_ROOT="$HOME/hm/ros2_ws"
SEMANTIC_SRC="$WS_ROOT/src/semantic"

# --- 环境初始化 ---
source /opt/ros/humble/setup.bash
if [ -f "$WS_ROOT/install/setup.bash" ]; then
    source "$WS_ROOT/install/setup.bash"
fi

export PYTHONPATH=$PYTHONPATH:$SEMANTIC_SRC

# --- 强力清理函数 ---
cleanup() {
    echo -e "\n[清理中] 正在关闭所有节点和可视化工具..."
    kill 0
    exit
}
trap cleanup SIGINT SIGTERM

# --- 1. 启动 Foxglove Bridge (后台运行) ---
echo "Launching Foxglove Bridge..."
ros2 launch foxglove_bridge foxglove_bridge_launch.xml &
# 稍微等待一下，确保 Bridge 端口已开启
sleep 2

# --- 2. 启动 Foxglove Studio (后台运行) ---
echo "Launching Foxglove Studio..."
# 使用 nohup 可以防止终端关闭导致 Foxglove 关闭，同时将输出丢弃
nohup foxglove-studio "foxglove://open?ds=foxglove-websocket&ds.url=ws://localhost:8765/" > /dev/null 2>&1 &

# --- 3. 启动静态 TF 发布器 ---
echo "Launching Static TF Publishers..."
ros2 run tf2_ros static_transform_publisher --x 0 --y 0 --z 0 --yaw 0 --pitch 0 --roll 0 --frame-id map --child-frame-id lidar --ros-args -p use_sim_time:=true &
ros2 run tf2_ros static_transform_publisher --x 0 --y 0 --z 0 --yaw 0 --pitch 0 --roll 0 --frame-id lidar --child-frame-id camera --ros-args -p use_sim_time:=true &

# --- 4. 启动拓扑管理节点 ---
echo "Starting Topology Manager..."
cd $SEMANTIC_SRC
python3 -m semantic_mapping.topology_manager \
    --ros-args -p use_sim_time:=true &

# --- 5. 启动感知节点 (前台运行) ---
echo "Starting Semantic Mapping Node..."
python3 -m semantic_mapping.mapping_ros2_node \
    --config config/mapping_mecanum_real.yaml \
    --ros-args -p use_sim_time:=true

# 保持脚本不退出
wait
