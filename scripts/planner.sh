#!/bin/bash

# --- 1. 路径定义 (请根据你的实际绝对路径核对) ---
WS_ROOT="/home/iot/hm/ros2_ws"
# language_planner 源码根目录
PLANNER_SRC="/home/iot/hm/ros2_ws/src/language_planner"
# semantic_mapping 源码根目录 (captioner 在它下面)
SEMANTIC_SRC="$WS_ROOT/src/semantic/semantic_mapping"

# --- 2. 环境变量设置 ---
export MISTRAL_API_KEY="tnOkCfCWOdrDc73zCkUWlISnawxp3Cf6"
export OPENAI_API_KEY="sk-cHT66N3kwGaD7mwO227dBe0aF98041De8e01260bB5Df687a"

# 【核心修复】同时添加 planner 路径和 semantic_mapping 路径
# 这样 Python 就能同时找到 language_planner 和 captioner 两个包了
export PYTHONPATH=$PYTHONPATH:$PLANNER_SRC:$SEMANTIC_SRC

# --- 3. 环境初始化 ---
source /opt/ros/humble/setup.bash
if [ -f "$WS_ROOT/install/setup.bash" ]; then
    source "$WS_ROOT/install/setup.bash"
fi

# --- 4. 启动大脑节点 (后台) ---
echo "Starting Language Planner Node (Brain)..."
# 注意：cd 到 PLANNER_SRC 运行
cd $PLANNER_SRC
python3 -m language_planner.language_planner_node \
    --model mistral \
    --platform mecanum \
    --ros-args -p use_sim_time:=false &

sleep 5

# --- 5. 启动交互终端 (前台) ---
echo "------------------------------------------------------"
echo "Planner Ready. Enter your command (e.g., 'go to the chair'):"
echo "------------------------------------------------------"
python3 -m language_planner.language_query_publisher

# 捕获退出信号
trap "kill 0" SIGINT SIGTERM
wait
