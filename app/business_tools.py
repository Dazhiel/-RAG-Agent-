"""扫地/拖地机器人客服业务工具。

这些工具没有副作用，只返回结构化建议，供模型组织最终回答时参考。
"""
import json
from typing import Any

from langchain_core.tools import tool


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _split_text(value: str) -> list[str]:
    if not value:
        return []
    items = [value]
    for separator in [",", "，", ";", "；", "\n", "、"]:
        next_items = []
        for item in items:
            next_items.extend(item.split(separator))
        items = next_items
    return [item.strip() for item in items if item.strip()]


@tool
def diagnose_fault(error_code: str = "", symptoms: str = "", context: str = "") -> str:
    """根据错误码、故障现象和使用场景，诊断扫地/拖地机器人故障并给出排查步骤。"""
    symptom_items = _split_text(symptoms)
    code = error_code.strip().upper()
    possible_causes = []
    check_steps = []
    need_service = False
    service_reason = "优先完成基础排查。"

    if code:
        possible_causes.append(f"设备报告错误码 {code}，需要结合说明书和知识库确认具体含义。")

    combined = f"{code} {symptoms} {context}"
    if any(word in combined for word in ["转圈", "导航", "地图", "定位", "避障"]):
        possible_causes.extend(["传感器脏污或被遮挡", "地图异常", "轮组或万向轮卡滞"])
        check_steps.extend(["清洁前方、侧边和底部传感器", "检查左右轮和万向轮是否缠绕毛发", "重启后尝试重新建图"])
    if any(word in combined for word in ["吸力", "吸不动", "灰尘", "滤网", "尘盒"]):
        possible_causes.extend(["尘盒已满", "滤网堵塞或未干透", "主刷或风道堵塞"])
        check_steps.extend(["清空尘盒", "清洗并完全晾干滤网", "检查主刷、边刷和吸入口是否堵塞"])
    if any(word in combined for word in ["不出水", "拖布", "水箱", "拖地"]):
        possible_causes.extend(["水箱缺水或安装不到位", "出水口堵塞", "拖布过脏"])
        check_steps.extend(["重新安装水箱", "检查出水口并清理堵塞", "清洗或更换拖布"])
    if any(word in combined for word in ["充电", "回充", "电池", "不开机"]):
        possible_causes.extend(["充电触点脏污", "充电座位置不合适", "电池或电源异常"])
        check_steps.extend(["擦拭机器人和充电座触点", "确认充电座两侧和前方留有空间", "更换插座后再次测试"])
    if any(word in combined for word in ["进水", "烧焦", "冒烟", "异响严重", "电池鼓包"]):
        need_service = True
        service_reason = "存在安全风险或硬件损坏迹象，应停止使用并联系售后。"

    if not possible_causes:
        possible_causes = ["故障信息不足，需要补充错误码、故障现象和已尝试步骤。"]
    if not check_steps:
        check_steps = ["记录错误码和出现频率", "重启设备观察是否复现", "检查尘盒、滤网、主刷、边刷、轮组和传感器"]

    return _json(
        {
            "工具": "故障诊断",
            "错误码": code or "未提供",
            "故障现象": symptom_items or [symptoms or "未提供"],
            "可能原因": list(dict.fromkeys(possible_causes)),
            "排查步骤": list(dict.fromkeys(check_steps)),
            "是否建议售后": {
                "需要售后": need_service,
                "原因": service_reason,
            },
        }
    )


@tool
def generate_maintenance_plan(
    location: str = "",
    weather: str = "",
    usage_frequency: str = "",
    has_pets: bool = False,
    floor_type: str = "",
    context: str = "",
) -> str:
    """根据地区、天气、使用频率、宠物情况和地面材质，生成扫地/拖地机器人保养计划。"""
    combined = f"{location} {weather} {usage_frequency} {floor_type} {context}"
    today = ["清空尘盒", "检查主刷和边刷是否缠绕毛发", "擦拭传感器表面"]
    this_week = ["清洗拖布", "检查滤网并按需清理", "检查轮组和万向轮是否卡滞"]
    this_month = ["深度清洁主刷、边刷、尘盒和水箱", "检查耗材磨损情况", "整理一次地图和禁区设置"]
    warnings = []

    if has_pets or any(word in combined for word in ["宠物", "猫", "狗", "毛发"]):
        today.append("重点清理主刷和边刷上的宠物毛发")
        this_week.append("增加滤网和尘盒清洁频率")
        warnings.append("宠物家庭不要等到吸力明显下降才清理毛发。")

    if any(word in combined for word in ["下雨", "潮湿", "湿度", "梅雨", "回南天"]):
        today.extend(["取下拖布并充分晾干", "确认滤网完全干透后再装回"])
        this_week.append("检查水箱和拖布支架是否有异味或霉斑")
        warnings.append("潮湿天气不要装回未干透的滤网，避免影响吸力或产生异味。")

    if any(word in combined for word in ["每天", "高频", "频繁"]):
        this_week.append("每周至少清洁一次滤网和主刷")
    if any(word in combined for word in ["木地板", "地板"]):
        warnings.append("木地板拖地建议控制出水量，避免长时间残留水渍。")

    return _json(
        {
            "工具": "保养计划",
            "今日建议": list(dict.fromkeys(today)),
            "本周建议": list(dict.fromkeys(this_week)),
            "本月建议": list(dict.fromkeys(this_month)),
            "注意事项": list(dict.fromkeys(warnings)),
        }
    )


@tool
def recommend_robot_type(
    home_area: str = "",
    floor_type: str = "",
    budget: str = "",
    has_pets: bool = False,
    has_children_or_elderly: bool = False,
    mopping_required: bool = True,
    automation_preference: str = "",
    context: str = "",
) -> str:
    """根据家庭面积、地面材质、预算、宠物、老人小孩和拖地需求，推荐扫地/拖地机器人类型。"""
    combined = f"{home_area} {floor_type} {budget} {automation_preference} {context}"
    recommended_type = "扫拖一体款" if mopping_required else "基础扫地款"
    reasons = []
    avoid = []
    follow_up = []

    if has_pets or any(word in combined for word in ["宠物", "猫", "狗", "毛发"]):
        recommended_type = "宠物家庭强化吸力扫拖一体款"
        reasons.extend(["宠物家庭需要更强吸力", "防缠绕主刷和大容量尘盒更重要"])
        avoid.append("低吸力且无防缠绕设计的基础款")

    if any(word in combined for word in ["大户型", "120", "150", "200"]) or home_area.strip() in ["120", "150", "200"]:
        reasons.append("大户型更需要长续航和稳定导航")
        follow_up.append("是否需要自动集尘或自动上下水？")

    if any(word in combined for word in ["少维护", "省事", "自动", "懒"]):
        recommended_type = "自动集尘扫拖一体款"
        reasons.append("自动集尘能减少日常维护频率")

    if any(word in combined for word in ["地毯", "毯"]):
        reasons.append("有地毯时应关注地毯识别、增压和拖布抬升能力")
        avoid.append("拖布不可抬升且地毯识别弱的机型")

    if has_children_or_elderly:
        reasons.append("有老人或小孩时建议选择避障稳定、噪音较低的机型")

    if not reasons:
        reasons = ["日常清洁场景优先选择导航稳定、续航足够、耗材易买的机型"]
        follow_up.append("请补充家庭面积、预算、是否有宠物和是否需要拖地。")

    return _json(
        {
            "工具": "选购推荐",
            "推荐类型": recommended_type,
            "推荐理由": list(dict.fromkeys(reasons)),
            "不建议选择": list(dict.fromkeys(avoid)),
            "追问问题": list(dict.fromkeys(follow_up)),
        }
    )


def get_business_tools():
    return [
        diagnose_fault,
        generate_maintenance_plan,
        recommend_robot_type,
    ]
