import re
from typing import Any
import string
from mathruler.grader import grade_answer
from word2number import w2n
from typing import Any, Dict, Generator, List, Tuple, Union

# Metadata
REWARD_NAME = "infoseek_rerank_name"
REWARD_TYPE = "sequential"


def entity_format_reward(response: str) -> float:
    response_clean = response.replace('assistant:', '')
    response_clean = response_clean.replace('user:', '')
    reason_pattern = r"<reason>.*?</reason>"
    entity_pattern = r"<entity>.*?</entity>"
    search_pattern = r"<search>.*?</search>"
    infomation_pattern = r"<information>.*?</information>"
    answer_pattern = r"<answer>.*?</answer>"

    # 编译单元表达式（用于匹配顺序）
    unit_pattern = f"({reason_pattern})|({entity_pattern})|({search_pattern})|({infomation_pattern})|({answer_pattern})"
    unit_re = re.compile(unit_pattern, re.DOTALL)

    # 匹配整个输入
    matches = list(unit_re.finditer(response_clean))
    
    # 如果没有匹配项，或拼接起来和原文不一致（中间出现非法内容），直接返回 0
    reconstructed = ''.join(m.group(0) for m in matches)
    if re.sub(r'\s+', '', reconstructed) != re.sub(r'\s+', '', response_clean):
        return 0.0

    # 提取结构标签序列
    sequence = []
    for m in matches:
        if m.group(1):
            sequence.append('R')
        elif m.group(2):
            sequence.append('E')
        elif m.group(3):
            sequence.append('S')
        elif m.group(4):
            sequence.append('I')
        elif m.group(5):
            sequence.append('A')

    if sequence[0] == 'R' and sequence[1] == 'E' and len(sequence) == 2:
        return 1.0
    else:
        return 0.0

def retrieval_format_reward(response: str) -> float:
    response_clean = response.replace('assistant:', '')
    response_clean = response_clean.replace('user:', '')
    reason_pattern = r"<reason>.*?</reason>"
    entity_pattern = r"<entity>.*?</entity>"
    search_pattern = r"<search>.*?</search>"
    infomation_pattern = r"<information>.*?</information>"
    answer_pattern = r"<answer>.*?</answer>"

    # 编译单元表达式（用于匹配顺序）
    unit_pattern = f"({reason_pattern})|({entity_pattern})|({search_pattern})|({infomation_pattern})|({answer_pattern})"
    unit_re = re.compile(unit_pattern, re.DOTALL)

    # 匹配整个输入
    matches = list(unit_re.finditer(response_clean))
    
    # 如果没有匹配项，或拼接起来和原文不一致（中间出现非法内容），直接返回 0
    reconstructed = ''.join(m.group(0) for m in matches)
    if re.sub(r'\s+', '', reconstructed) != re.sub(r'\s+', '', response_clean):
        return 0.0

    # 提取结构标签序列
    sequence = []
    for m in matches:
        if m.group(1):
            sequence.append('R')
        elif m.group(2):
            sequence.append('E')
        elif m.group(3):
            sequence.append('S')
        elif m.group(4):
            sequence.append('I')
        elif m.group(5):
            sequence.append('A')

     # 规则检测
    if len(sequence) < 2 or sequence[0] != 'R':
        return 0.0 


    if sequence[-1] != 'A':
        return 0.0

    # S前一个必须是R，后一个必须是I
    if 'S' in sequence:
        s_idx = sequence.index('S')
        # 必须不是开头(为了看前一个) 且 必须不是结尾(为了看后一个)
        if s_idx > 0 and s_idx < len(sequence) - 1:
            if sequence[s_idx - 1] != 'R' or sequence[s_idx + 1] != 'I':
                return 0.0
        else:
            # 如果S出现在开头或结尾，说明它不完整，也应该判错(返回0.0)
            # 或者是你认为这种情况不应该发生？根据你的业务逻辑决定
            return 0.0

    if sequence.count('A') != 1:
        return 0.0  # 不止一个 answer

    # E不能出现在序列中
    if 'E' in sequence:
        return 0.0

    return 1.0

def overall_format_reward(response: str) -> float:
    response_clean = response.replace('assistant:', '')
    response_clean = response_clean.replace('user:', '')
    reason_pattern = r"<reason>.*?</reason>"
    entity_pattern = r"<entity>.*?</entity>"
    search_pattern = r"<search>.*?</search>"
    infomation_pattern = r"<information>.*?</information>"
    answer_pattern = r"<answer>.*?</answer>"

    # 编译单元表达式（用于匹配顺序）
    unit_pattern = f"({reason_pattern})|({entity_pattern})|({search_pattern})|({infomation_pattern})|({answer_pattern})"
    unit_re = re.compile(unit_pattern, re.DOTALL)

    # 匹配整个输入
    matches = list(unit_re.finditer(response_clean))
    
    # 如果没有匹配项，或拼接起来和原文不一致（中间出现非法内容），直接返回 0
    reconstructed = ''.join(m.group(0) for m in matches)
    if re.sub(r'\s+', '', reconstructed) != re.sub(r'\s+', '', response_clean):
        return 0.0

    # 提取结构标签序列
    sequence = []
    for m in matches:
        if m.group(1):
            sequence.append('R')
        elif m.group(2):
            sequence.append('E')
        elif m.group(3):
            sequence.append('S')
        elif m.group(4):
            sequence.append('I')
        elif m.group(5):
            sequence.append('A')

     # 至少有两个元素，且第一个元素一定是reason
    if len(sequence) < 2 or sequence[0] != 'R':
        return 0.0 

    # E前一个必须是R
    if 'E' in sequence and sequence.index('E') > 0 and sequence[sequence.index('E') - 1] != 'R':
        return 0.0

    # S前一个必须是R，后一个必须是I
    if 'S' in sequence:
        s_idx = sequence.index('S')
        # 必须不是开头(为了看前一个) 且 必须不是结尾(为了看后一个)
        if s_idx > 0 and s_idx < len(sequence) - 1:
            if sequence[s_idx - 1] != 'R' or sequence[s_idx + 1] != 'I':
                return 0.0
        else:
            # 如果S出现在开头或结尾，说明它不完整，也应该判错(返回0.0)
            # 或者是你认为这种情况不应该发生？根据你的业务逻辑决定
            return 0.0

    return 1.0

def search_reward(response: str) -> float:
    try:
        search_match = re.search(r"<search>(.*?)</search>", response)
        search_predict = search_match.group(1).strip() if search_match else ''
        # 如果检测到search，并且不为空，则返回1.0
        if search_predict.strip().lower() != '':
            return 1.0

    except Exception:
        pass

    return 0.0


def rerank_reward(response: str, entity_gt: str) -> float:
    try:
        entity_match = re.search(r"<entity>(.*?)</entity>", response)
        entity_predict = entity_match.group(1).strip() if entity_match else response.strip()

        if entity_predict.strip().lower() == entity_gt.strip().lower():
            return 1.0 

    except Exception:
        pass

    return 0.0


def normalize_answer(text: str) -> str:
    """Normalize a given text by removing articles, punctuation, and white spaces, and converting to lowercase."""
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())

    def remove_punctuation(text: str) -> str:
        return ''.join(ch for ch in text if ch not in set(string.punctuation))

    def lowercase(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punctuation(lowercase(text))))

def string_accuracy_reward(predict_str: str, ground_truth: list[str]) -> float:
    try:
        answer_match = re.search(r"<answer>(.*?)</answer>", predict_str)
        given_answer = answer_match.group(1).strip() if answer_match else predict_str.strip()
        for answer in ground_truth:
            if grade_answer(normalize_answer(given_answer), normalize_answer(answer)):
                return 1.0
    except Exception:
        pass
    return 0.0

def replace_number_words(text):
    # 定义一个正则表达式模式，用于匹配可能的数字词组
    pattern = re.compile(r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|'
                         r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|'
                         r'eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|'
                         r'eighty|ninety|hundred|thousand|million|billion|trillion|'
                         r'point|and|[-\s])+\b', re.IGNORECASE)

    def convert(match):
        try:
            # 尝试将匹配的数字词组转换为数字
            return str(w2n.word_to_num(match.group()))
        except ValueError:
            # 如果转换失败，返回原始匹配内容
            return match.group()

    # 使用正则表达式替换文本中的数字词组
    return pattern.sub(convert, text)

def find_all(s: str, c: str) -> Generator[int, None, None]:
    """Find all occurrences of a character in a string and return their indices.

    Args:
        s: The input string to search.
        c: The character to search for.

    Yields:
        int: The index of the next occurrence of the character.
    """
    idx = s.find(c)
    while idx != -1:
        yield idx
        idx = s.find(c, idx + 1)

def clean_str_range(text: str) -> str:
    """Clean range expression in a string (e.g., '9-10' --> '9 - 10').

    Args:
        text: The input string containing the range expression.

    Returns:
        str: The cleaned string with proper spacing around the hyphen.
    """
    # try:
    idx_list = list(find_all(text, '-'))
    idx_replace = [
        idx for idx in idx_list if idx >= 1 and text[idx - 1].isdigit()
    ]
    new_str = ''.join(
        ' - ' if idx in idx_replace else s for idx, s in enumerate(text)
    )
    return new_str

def process_numerical_answer(string_number: str) -> Union[float, List[float]]:
    """Parses numerical answer string into numbers (a single number or a range).

    1) Clean the string and extract numbers;
    2) if there are 2 numbers, return a range as [minimum value, maximum value]
        else if there is 1 number, return a single number
        else return [0, 0]

    Args:
        string_number: A string representing a numerical answer.

    Returns:
        A single digit or a list with 2 numbers.
    """
    # Clean string
    string_number = replace_number_words(string_number)
    string_number = clean_str_range(string_number)
    numerical_numbers_tmp = re.findall(
        r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?', string_number
    )
    numerical_numbers_tmp = [
        n.replace(',', '').strip('.') for n in numerical_numbers_tmp
    ]
    numerical_numbers = []
    for n in numerical_numbers_tmp:
        if n.count('.') > 1:
            n = n.split('.')[0]
            numerical_numbers.append(float(n))
        else:
            numerical_numbers.append(float(n))

    # Use the first 2 numbers
    if len(numerical_numbers) > 2:
        numerical_numbers = numerical_numbers[:2]

    if len(numerical_numbers) == 2:
        first_val = numerical_numbers[0]
        second_val = numerical_numbers[1]
        return [first_val, second_val] if first_val <= second_val else first_val
    elif len(numerical_numbers) == 1:
        return numerical_numbers[0]
    else:
        return [0, 0]

def safe_division(x: float, y: float) -> float:
    """Divide x by y, returning 0 if y is 0."""
    return x / y if y != 0 else 0

def range_intersection_over_union(
        x_list: List[float], y_list: List[float]
    ) -> float:
    """Calculate the intersection over union (IOU) of two ranges."""
    min_1, max_1 = min(x_list), max(x_list)
    min_2, max_2 = min(y_list), max(y_list)

    overlap = max(0.0, min(max_1, max_2) - max(min_1, min_2))
    length_x = (max_1 - min_1) + 1e-12
    length_y = (max_2 - min_2) + 1e-12
    iou = safe_division(overlap, length_x + length_y - overlap)
    return iou

def numerical_accuracy_reward(predict_str: str, ground_truth: list) -> float:
    """Calculate the accuracy reward for numerical answers in the InfoSeek task."""
    try:
        content_match = re.findall(r"<answer>(.*?)</answer>", predict_str)
        given_answer = content_match[-1].strip() if content_match else predict_str.strip()
        # given_answer = content_match[-1].strip() if content_match else ''
        # 将数字词组转换为数字
        # given_answer = replace_number_words(given_answer)

        if given_answer is None or given_answer == "":
            return 0.0
        
        given_answer = process_numerical_answer(given_answer)

        ground_truth = [
            replace_number_words(answer) for answer in ground_truth
        ]
        min_value = min([float(answer) for answer in ground_truth])
        max_value = max([float(answer) for answer in ground_truth])
        ground_truth = [min_value, max_value]
        # given_answer = float(given_answer)
        
        if isinstance(given_answer, list):
            if ground_truth[0] <= given_answer[0] <= ground_truth[1] and ground_truth[0] <= given_answer[1] <= ground_truth[1]:
                return 1
            else:
                iou = range_intersection_over_union(given_answer, ground_truth)
                return 1 if iou >= 0.5 - 1e-12 else 0
            
        if min_value <= given_answer <= max_value:
            return 1.0
        
        return 0.0
    except Exception as e:
        print(f"Error processing numerical answer: {e}")
        return 0.0

# def compute_score(reward_input: dict[str, Any], format_weight: float = 0.5) -> dict[str, float]:
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    # breakpoint()
    import ipdb; ipdb.set_trace()
    ground_truth = reward_input["ground_truth"]
    data_type = ground_truth["data_type"]
    if data_type == "string":
        accuracy_score = string_accuracy_reward(reward_input["response"], ground_truth["answer"])
    elif data_type == "numerical":
        accuracy_score = numerical_accuracy_reward(reward_input["response"], ground_truth["answer"])
    else:
        raise ValueError(f"Invalid data type: {data_type}")

    # import ipdb; ipdb.set_trace()
    task_type = reward_input["task_type"]

    rerank_score = rerank_reward(reward_input["response"], ground_truth["entity"])
    search_score = search_reward(reward_input["response"])
    
    if task_type == "entity":
        format_score = entity_format_reward(reward_input["response"])
        overall_score = 0.5 * rerank_score + 0.5 * format_score
    elif task_type == "retrieval":
        format_score = retrieval_format_reward(reward_input["response"])
        overall_score = 0.5 * accuracy_score + 0.5 * format_score
    else:
        format_score = overall_format_reward(reward_input["response"])
        overall_score = 0.5 * rerank_score + 0.5 * accuracy_score + 0.5 * format_score

    return {
        "overall": overall_score,
        "format": format_score,
        "search": search_score,
        "rerank": rerank_score,
        "accuracy": accuracy_score,
    }
