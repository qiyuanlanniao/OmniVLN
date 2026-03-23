#!/bin/bash

# --- 路径配置 ---
WS_ROOT="$HOME/hm/ros2_ws"
SEMANTIC_SRC="$WS_ROOT/src/semantic"

# --- 环境初始化 ---
source /opt/ros/humble/setup.bash
if [ -f "$WS_ROOT/install/setup.bash" ]; then
    source "$WS_ROOT/install/setup.bash"
fi

# 关键：设置 PYTHONPATH，让 Python 能找到 semantic_mapping 包
export PYTHONPATH=$PYTHONPATH:$SEMANTIC_SRC

# --- 强力清理函数 ---
# 当按下 Ctrl+C 时，确保关闭所有后台进程
cleanup() {
    echo -e "\n[清理中] 正在关闭所有节点和发布器..."
    # kill 0 表示关闭当前进程组中的所有进程
    kill 0
    exit
}
trap cleanup SIGINT SIGTERM

# --- 1. 启动静态 TF 发布器 (后台运行) ---
echo "Launching Static TF Publishers..."
# map -> lidar
ros2 run tf2_ros static_transform_publisher --x 0 --y 0 --z 0 --yaw 0 --pitch 0 --roll 0 --frame-id map --child-frame-id lidar --ros-args -p use_sim_time:=true &
# lidar -> camera
ros2 run tf2_ros static_transform_publisher --x 0 --y 0 --z 0 --yaw 0 --pitch 0 --roll 0 --frame-id lidar --child-frame-id camera --ros-args -p use_sim_time:=true &


# --- 2. 启动 RViz2 (后台运行) ---
echo "Starting RViz2..."
rviz2 --ros-args -p use_sim_time:=true &


# --- 3. 启动拓扑管理节点 ---
echo "Starting Topology Manager..."
cd $SEMANTIC_SRC
python3 -m semantic_mapping.topology_manager \
    --ros-args -p use_sim_time:=true &


# --- 4. 启动感知节点 (前台运行) ---
echo "Starting Semantic Mapping Node..."
# 注意：cd 已经在上面执行过了，这里直接运行
python3 -m semantic_mapping.mapping_ros2_node \
    --config config/mapping_mecanum_real.yaml \
    --ros-args -p use_sim_time:=true

# 保持脚本不退出
wait
