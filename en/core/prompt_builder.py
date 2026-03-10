# iot@iot:~/hm/ros2_ws/src/en/core/prompt_builder.py

import json
import numpy as np
import ast
from .geometry import OctantAnalyzer
from .extractor import TargetExtractor

# 模拟房间中心坐标（对应贡献 F: 认知持久化中的房间锚点）
ROOM_CENTERS = {
    0: [5.0, 5.0, 1.0],
    1: [-5.0, 5.0, 1.0],
    2: [-5.0, -5.0, 1.0],
    3: [5.0, -5.0, 1.0]
}

class PromptBuilder:
    def __init__(self):
        self.analyzer = OctantAnalyzer()
        self.extractor = TargetExtractor()
        self.focal_distance = 3.0
        
        self.system_context = (
            "SYSTEM RULE: Respond ONLY with 'go_near(ID)'. "
            "DO NOT explain. DO NOT add object names. "
            "Example: go_near(15)"
        )

    def build_baseline_prompt(self, objects, instruction):
        """
        Baseline: 全量堆叠 (适配 D1-D7)
        """
        obj_list_str = []
        for obj in objects:
            if obj['id'] == -1: continue
            info = (
                f"- ID: {obj['id']}, Name: {obj['name']}, "
                f"Pos: {obj['position']}, Room: {obj['room']}, "
                f"Description: {obj['caption']}"
            )
            obj_list_str.append(info)

        return f"{self.system_context}\n\n=== Environment Objects ===\n" + \
               "\n".join(obj_list_str) + f"\n\n=== User Instruction ===\n{instruction}"

    def build_ours_prompt(self, robot_pos, objects, instruction):
        """
        Ours: 多分辨率空间注意力 (支持跨房间瞬移逻辑)
        """
        target_category = self.extractor.extract(instruction)
        
        # 确定机器人初始所在的房间
        initial_room = 0
        for obj in objects:
            if obj['id'] == -1:
                initial_room = obj.get('room', 0)
                break

        focal_tier = []       # TIER 1: 高细节 (当前视野内或瞬移后的目标)
        peripheral_tier = []   # TIER 2: 中细节 (当前房间远景)
        room_summaries = {}    # TIER 3: 低细节 (其他房间汇总)

        for obj in objects:
            if obj['id'] == -1: continue
            
            obj_id = obj['id']
            obj_room = obj.get('room', 0)
            obj_name = obj['name'].lower()
            is_target = (target_category and target_category in obj_name) or (str(obj_id) in instruction)

            # --- 核心逻辑判断 ---
            if obj_room == initial_room:
                # 逻辑 A：物体在当前房间 (D1-D6 逻辑)
                spatial = self.analyzer.analyze(robot_pos, obj['position'])
                if spatial['distance'] < self.focal_distance or is_target:
                    info = f"- {obj['name']}(ID:{obj_id}): {spatial['octant']}, {spatial['distance']}m. Detail: {obj['caption']}"
                    focal_tier.append(info)
                else:
                    info = f"- {obj['name']}(ID:{obj_id}): {spatial['octant']}, {spatial['distance']}m"
                    peripheral_tier.append(info)
            
            else:
                # 逻辑 B：物体在其他房间 (D7-D9 逻辑)
                if is_target:
                    # 如果是目标候选，模拟“瞬移”到该房间中心获取细粒度信息
                    target_room_center = ROOM_CENTERS.get(obj_room, [0,0,0])
                    spatial = self.analyzer.analyze(target_room_center, obj['position'])
                    # 标记是从哪个房间中心观察到的
                    info = (f"- {obj['name']}(ID:{obj_id}): [In Room {obj_room}] "
                            f"Observation from room center: {spatial['octant']}, {spatial['distance']}m. "
                            f"Detail: {obj['caption']}")
                    focal_tier.append(info)
                else:
                    # 非目标的跨房间物体，仅进入汇总桶 (TIER 3)
                    if obj_room not in room_summaries:
                        room_summaries[obj_room] = []
                    room_summaries[obj_room].append(obj['name'])

        # 构造 TIER 3 文本
        memory_tier = []
        for rid, items in room_summaries.items():
            counts = {name: items.count(name) for name in set(items)}
            summary = ", ".join([f"{count} {name}(s)" for name, count in counts.items()])
            memory_tier.append(f"- Room {rid}: Contains {summary}")

        # 组装 H-CoT Prompt
        prompt = (
            f"{self.system_context}\n\n"
            f"### TIER 1: IMMEDIATE FOCAL SIGHT & TARGET INSPECTION ###\n"
            "(Detailed information of nearby objects and potential targets in other rooms)\n"
            + ("\n".join(focal_tier) if focal_tier else "None") + "\n\n"
            
            f"### TIER 2: PERIPHERAL AWARENESS (Current Room {initial_room}) ###\n"
            "(Landmarks in your current area without details)\n"
            + ("\n".join(peripheral_tier) if peripheral_tier else "None") + "\n\n"
            
            f"### TIER 3: GLOBAL TOPOLOGICAL MEMORY ###\n"
            "(Summarized overview of distant rooms)\n"
            + ("\n".join(memory_tier) if memory_tier else "No other rooms detected.") + "\n\n"
            
            f"=== User Instruction ===\n{instruction}\n\n"
            "Reasoning Path: 1. Identify which Room the target belongs to. "
            "2. Match the detail in TIER 1 with the user's specific description. "
            "3. Confirm the ID and call go_near."
        )
        return prompt