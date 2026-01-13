import re
from typing import Any
import string
from mathruler.grader import grade_answer
from word2number import w2n
from typing import Any, Dict, Generator, List, Tuple, Union

def format_reward(response: str) -> float:
    """
    格式奖励（正向判定）：
    - <think> 与 </think> 必须成对（可都不出现）
    - 必须存在至少一段非空的 <answer>...</answer>
    - <answer> 与 </answer> 必须成对

    满足要求返回 1.0，否则返回 0.0。
    """
    response_clean = response.replace("assistant:", "").replace("user:", "")

    # 1) <think> 标签必须成对（允许完全没有）
    if response_clean.count("<reason>") != response_clean.count("</reason>"):
        return 0.0

    # 2) 只在最后一个 </think> 之后查找 <answer>（与 deepeyes 一致）
    predict_no_think = (
        response_clean.split("</reason>")[-1].strip() if "</reason>" in response_clean else response_clean.strip()
    )

    # 3) <answer> 标签必须成对，且必须能抽取到非空答案
    if predict_no_think.count("<answer>") != predict_no_think.count("</answer>"):
        return 0.0

    answer_match = re.search(r"<answer>(.*?)</answer>", predict_no_think, re.DOTALL)
    if not answer_match:
        return 0.0

    answer_text = answer_match.group(1).strip()
    if not answer_text:
        return 0.0

    return 1.0

def tool_call_reward(response: str) -> float:
    try:
        tool_call_match = re.search(r"<search_call>(.*?)</search_call>", response, re.DOTALL)
        tool_call_predict = tool_call_match.group(1).strip() if tool_call_match else ''
        if tool_call_predict.strip().lower() != '':
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

def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    # breakpoint()
    extra_info = extra_info or {}
    response = solution_str
    data_type = ground_truth["data_type"]
    answers = ground_truth["answer"]
    if data_type == "string":
        accuracy_score = string_accuracy_reward(response, answers)
    elif data_type == "numerical":
        accuracy_score = numerical_accuracy_reward(response, answers)
    else:
        raise ValueError(f"Invalid data type: {data_type}")

    format_score = format_reward(response)
    tool_call_score = tool_call_reward(response)
    overall_score = 0.5 * accuracy_score + 0.5 * format_score
    # NaiveRewardManager expects key "score" if returning dict.
    return {
        "score": float(overall_score),
        "format": float(format_score),
        "tool_call": float(tool_call_score),
        "accuracy": float(accuracy_score),
    }

if __name__ == "__main__":
    data_source = "searchR1_infoseek"
    solution_str = """<reason>
        The question asks who "found" the Castle of Óbidos. This is a historical structure, and "found" likely refers to its original builders or the people responsible for its construction. The Castle of Óbidos is a medieval castle located in Portugal, and its origins date back to the early Middle Ages. It was likely built by the Romans or early Christian kingdoms in the region, but more specifically, it was constructed by the Portuguese nobility during the 11th and 12th centuries to defend against Moorish invasions.

        To confirm the exact builders or the historical context, I will search for information about the construction of the Castle of Óbidos.
        </reason>
        <search_call>
        {"name": "search", "arguments": {"query_list": ["who built the Castle of Óbidos Portugal", "history of Castle of Óbidos construction"]}}
        </search_call>user
        <search_response>

        {"result": "Doc 1 (Title: Castle of Óbidos)\nThe Castle of Óbidos () is a well-preserved medieval castle located in the civil parish of Santa Maria, São Pedro e Sobral da Lagoa, in the municipality of Óbidos, Portuguese District of Leiria.## History.Óbidos had its foundation in the existence of a fortified settlement, likely over the Luso-Roman castro and Roman oppidum (then civitas) on the hilltop, that was the mysterious \"Eburobrittium\", once cited by Pliny the Elder as being situated between \"Collipo\" (present-day Golpilheira near Leiria) and \"Olisipo\" (Lisbon). Archeological surveys later resulted in the discovery of a forum, baths and other Roman structures near the settlement. The area was later settled by the Lusitanos (since the 4th century BC) and Romans in the first century, but later occupied by Visigoths during the 5th\n\nDoc 2 (Title: Castle of Óbidos)\nJoão Gonçalves, in the Battle of Aljubarrota.A settlement started building to the west and southern part of the cas...(truncated)...ettlement. The area was later settled by the Lusitanos (since the 4th century BC) and Romans in the first century, but later occupied by Visigoths during the 5th\n\nDoc 3 (Title: Castle of Óbidos)\n( the \"Torre de D. Fernando\" or D. Fernando Tower) constructed by the Queen. As part of the construction, a connection was made between the dungeons and the Church of Santiago gallery.During the context of the 1383–1385 Portuguese succession crises, the alcalde (against the wishes of its residents) allied with John I of Castile, resulting in an assault by the forces loyal to the Master of Avis (future King John). Óbidos and its castle were transferred to John (following his election by the Cortes in Coimbra as King) by Vasco Gonçalves Teixeira, following the death of his father and former-alcalde, João Gonçalves, in the Battle of Aljubarrota.A settlement started building to the west and southern part of the castle within the walls sometime in the 15th century. During the reign of King D."}

        </search_response>
        assistant
        <reason>
        From the search results, it is clear that the Castle of Óbidos has a long and layered history. The earliest foundations of the site date back to the Roman period, with evidence of a Roman settlement called "Eburobrittium" on the hilltop. The area was later occupied by the Lusitanians and Romans, and later by the Visigoths. However, the castle as a fortified medieval structure was likely built by the Portuguese nobility during the 11th and 12th centuries to defend against Moorish invasions.

        The search results also mention that during the 1383–1385 Portuguese succession crisis, the castle was transferred to King John I of Portugal, but this refers to a later political event, not its original construction. The castle's construction was likely initiated by local Portuguese nobles or rulers during the medieval period, but the exact individual or group responsible for its initial building is not explicitly mentioned in the results.

        Therefore, the castle was not "found" by a single person or group, but rather constructed over time by various groups, starting with the Romans and later by the Portuguese nobility during the medieval period.
        </reason>
        <answer>
        The Castle of Óbidos was not "found" by a single person or group, but rather constructed over time. Its earliest foundations date back to the Roman period, with a settlement called "Eburobrittium."
    """
    ground_truth = {
        "data_type": "string",
        "answer": ["Afonso Henriques", "O fundador", "O conquistador", "O grande", "Afonso I of Portugal"],
        "entity": "Castle of Óbidos",
    }

    print(compute_score(data_source, solution_str, ground_truth))