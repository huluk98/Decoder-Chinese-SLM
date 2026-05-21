from __future__ import annotations

import re

STRUCTURED_SPACE_PATTERN = re.compile(r"\s*([(),:=\[\]{}])\s*")
COMMAND_PREFIX_PATTERN = re.compile(r"^(?:好的|好|收到|可以|没问题|行)[,，:：\s]*")
COMMAND_SUFFIX_PATTERN = re.compile(r"[\s。．.！!？?]+$")
ACTION_SPLIT_PATTERN = re.compile(r"[;；]\s*")
SPACE_PATTERN = re.compile(r"\s+")

ROOMS = ("客厅", "卧室", "书房", "厨房", "餐厅")
DEVICES = ("空调", "电视", "灯", "灯光", "音箱")
MODE_ALIASES = (
    ("舒睡模式一", ("舒睡模式一", "舒睡1", "舒睡模式1", "舒服模式1")),
    ("舒睡模式二", ("舒睡模式二", "舒睡2", "舒睡模式2", "舒服模式2")),
    ("睡眠模式", ("睡眠模式",)),
    ("舒睡模式", ("舒睡模式", "舒睡")),
    ("学习模式", ("学习模式",)),
    ("夜灯模式", ("夜灯模式",)),
    ("普通模式", ("普通模式",)),
    ("制冷模式", ("制冷模式",)),
    ("制热模式", ("制热模式",)),
    ("抽湿模式", ("抽湿模式", "除湿模式")),
    ("送风模式", ("送风模式",)),
    ("自动模式", ("自动模式",)),
    ("环保模式", ("环保模式",)),
    ("游戏模式", ("游戏模式",)),
    ("电影模式", ("电影模式",)),
    ("体育模式", ("体育模式",)),
    ("新闻模式", ("新闻模式",)),
    ("儿童模式", ("儿童模式",)),
)
LIGHT_COLORS = ("冷色光", "暖色光", "自然光")
DIRECTIONS = ("上下左右", "上下", "左右", "不动", "左", "右", "上", "下")
CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def canonicalize_command_response(text: str) -> str:
    """Canonicalize short smart-home responses for exact command evaluation.

    This intentionally stays conservative: common wording variants such as
    "灯已开启" and "已打开灯" collapse to the same command, but device, room,
    mode, direction, and on/off polarity differences remain visible.
    """
    text = _clean_command_text(text)
    actions = [_canonicalize_action(action) for action in ACTION_SPLIT_PATTERN.split(text) if action.strip()]
    if not actions:
        return text
    return " | ".join(sorted(actions))


def _clean_command_text(text: str) -> str:
    text = STRUCTURED_SPACE_PATTERN.sub(r"\1", str(text))
    text = text.replace('"', "'")
    text = SPACE_PATTERN.sub("", text)
    text = COMMAND_PREFIX_PATTERN.sub("", text)
    text = COMMAND_SUFFIX_PATTERN.sub("", text)
    text = _normalize_numbers(text)
    return text.strip()


def _canonicalize_action(action: str) -> str:
    action = _clean_command_text(action)
    device = _device_for(action)

    timer = _timer_for(action)
    if timer:
        op = _operation_for(action)
        return f"{device or '设备'}:定时:{timer}:{op or '执行'}"

    mode = _mode_for(action)
    if mode:
        op = "关闭" if _is_mode_close(action) else "切换"
        return f"{device or '设备'}:模式:{mode}:{op}"

    if device in {"灯", "灯光"} or any(word in action for word in ("亮度", "调亮", "调暗", "最亮", "最暗")):
        light = _light_action_for(action)
        if light:
            return f"灯:{light}"

    if device == "空调" or any(word in action for word in ("风向", "风速", "温度")):
        ac = _aircon_action_for(action)
        if ac:
            return f"空调:{ac}"

    if device and "电视" in device:
        tv = _tv_action_for(action)
        if tv:
            return f"{device}:{tv}"

    op = _operation_for(action)
    if op:
        return f"{device or '设备'}:{op}"

    return action


def _normalize_numbers(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        value = _parse_cn_number(match.group(1))
        return str(value) if value is not None else match.group(1)

    return re.sub(r"([零〇一二两三四五六七八九十]+)(?=(?:点|时|小时|分钟|分|度|档|台|个|后))", replace, text)


def _parse_cn_number(value: str) -> int | None:
    if not value:
        return None
    if value.isdigit():
        return int(value)
    if value == "十":
        return 10
    if "十" in value:
        left, _, right = value.partition("十")
        tens = CN_DIGITS.get(left, 1) if left else 1
        ones = CN_DIGITS.get(right, 0) if right else 0
        return tens * 10 + ones
    if len(value) == 1:
        return CN_DIGITS.get(value)
    return None


def _device_for(action: str) -> str:
    for room in ROOMS:
        if f"{room}电视" in action or (room in action and "电视" in action):
            return f"{room}电视"
    if "电视" in action or "频道" in action or "字幕" in action or "画面" in action:
        return "电视"
    if "空调" in action or "冷气" in action:
        return "空调"
    if "灯光" in action:
        return "灯"
    if "灯" in action or "亮度" in action:
        return "灯"
    if "音箱" in action:
        return "音箱"
    return ""


def _operation_for(action: str) -> str:
    if any(word in action for word in ("已关闭", "关闭", "关掉", "关上")):
        return "关闭"
    if any(word in action for word in ("已打开", "已开启", "打开", "开启", "启动", "运行")):
        return "打开"
    return ""


def _timer_for(action: str) -> str:
    if "后" not in action:
        return ""
    match = re.search(r"(?:设置|定时|定)?([0-9半]+(?:点半|点|小时|时|分钟|分)?)后(?:开启|打开|关闭|运行)", action)
    if not match:
        return ""
    timer = match.group(1)
    if "点" not in timer and timer.endswith("时") and not timer.endswith("小时"):
        timer = f"{timer[:-1]}小时"
    return timer


def _mode_for(action: str) -> str:
    for canonical, aliases in MODE_ALIASES:
        if any(alias in action for alias in aliases):
            return canonical
    return ""


def _is_mode_close(action: str) -> bool:
    return any(word in action for word in ("关闭", "关掉", "结束", "退出", "停止", "不要"))


def _light_action_for(action: str) -> str:
    for color in LIGHT_COLORS:
        if color in action:
            return f"色温:{color}"
    if any(word in action for word in ("最亮", "最高")):
        return "亮度:最高"
    if any(word in action for word in ("最暗", "最低")):
        return "亮度:最低"
    if any(word in action for word in ("调亮", "调高", "提高")):
        return "亮度:提高"
    if any(word in action for word in ("调暗", "调低", "降低")):
        return "亮度:降低"
    return ""


def _aircon_action_for(action: str) -> str:
    if "风向" in action:
        for direction in DIRECTIONS:
            if direction in action:
                return f"风向:{direction}"
    if "风速" in action:
        if any(word in action for word in ("调高", "提高", "升高", "高一档")):
            return "风速:提高"
        if any(word in action for word in ("调低", "降低", "低一档")):
            return "风速:降低"
    if "温度" in action:
        exact = re.search(r"温度(?:调到|设为|设置为)?([0-9]+)度", action)
        if exact:
            return f"温度:{exact.group(1)}度"
        if any(word in action for word in ("调高", "升高", "提高")):
            return "温度:提高"
        if any(word in action for word in ("调低", "降低")):
            return "温度:降低"
    return ""


def _tv_action_for(action: str) -> str:
    if "字幕" in action:
        return f"字幕:{_operation_for(action) or '切换'}"
    if "静音" in action:
        return f"静音:{_operation_for(action) or '打开'}"
    if "音量" in action:
        if any(word in action for word in ("调高", "提高", "大")):
            return "音量:提高"
        if any(word in action for word in ("调低", "降低", "小")):
            return "音量:降低"
    if "上一个频道" in action or "上一" in action:
        return "频道:上一个"
    if "下一个频道" in action or "下一" in action:
        return "频道:下一个"
    if "暂停" in action:
        return "播放:暂停"
    if "播放" in action:
        return "播放:开始"
    return ""
