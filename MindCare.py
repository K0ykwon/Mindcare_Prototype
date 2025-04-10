import os
import re
import json
import hashlib
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

emotion_mapping = {
    1: "공포",
    2: "놀람",
    3: "중립",
    4: "혐오",
    5: "분노",
    6: "슬픔",
    7: "행복"
}

def get_openai_client():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=api_key)

def extract_integer(text):
    match = re.search(r"\d+", text)
    return int(match.group()) if match else None

def preprocess_input_with_openai(text):
    client = get_openai_client()
    prompt = f"다음 문장을 정리해 주세요: {text}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def get_sentiment_score(text):
    client = get_openai_client()
    prompt = (
        f"다음 문장의 감정을 1(공포), 2(놀람), 3(중립), 4(혐오), 5(분노), 6(슬픔), 7(행복)으로 "
        f"정수형으로 분류해 주세요. 결과는 정수형 숫자만 출력하세요: {text}"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return extract_integer(response.choices[0].message.content.strip())

def is_negative_emotion(score):
    return score in [1, 4, 5, 6]

def is_warning_signal(user_input):
    client = get_openai_client()
    prompt = (
        "다음 문장에서 자해, 자살, 극단적인 감정과 관련된 위험 신호가 있는 경우 'true', 없으면 'false'만 답하세요: "
        f"{user_input}"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    result = response.choices[0].message.content.strip().lower()
    return "true" in result

def encrypt_name(name):
    return hashlib.sha256(name.encode()).hexdigest()

def log_interaction(user_input, sentiment_score, response_text, username=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_data = {
        "timestamp": timestamp,
        "sentiment_score": emotion_mapping[sentiment_score],
        "response": response_text,
        "user_input": user_input,
        "username_hash": encrypt_name(username) if username else None
    }
    os.makedirs("logs", exist_ok=True)
    with open(f"logs/{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

def load_previous_logs(username, limit=3):
    """
    username을 해시 처리하여 해당 사용자 로그만 불러오고,
    최근 limit개 만큼 반환합니다.
    반환값은 dict: {timestamp: {"input": ..., "response": ...}}
    """
    if not username:
        return {}

    username_hash = encrypt_name(username)
    logs = {}

    # 최신 로그부터 정렬
    for filename in sorted(os.listdir("logs"), reverse=True):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join("logs", filename)
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            if data.get("username_hash") == username_hash:
                timestamp = data.get("timestamp", filename.replace(".json", ""))
                logs[timestamp] = {
                    "input": data.get("user_input", ""),
                    "response": data.get("response", "")
                }
                if len(logs) >= limit:
                    break

    # 최신순 → 오래된 순으로 정렬해서 반환
    return dict(sorted(logs.items()))


def generate_response(sentiment_score, user_input, username=None):
    client = get_openai_client()
    dominant_emotion = emotion_mapping.get(sentiment_score, "알 수 없음")

    previous_logs = load_previous_logs(username)
    context_lines = []

    if previous_logs:
        context_lines.append("이전 상담 대화 기록은 다음과 같습니다:")
        for timestamp, log in previous_logs.items():
            context_lines.append(f"- [{timestamp}] 사용자: {log['input']}\n  응답: {log['response']}")
        context_lines.append("")

    if is_negative_emotion(sentiment_score):
        if is_warning_signal(user_input):
            prompt = (
                "자살, 자해 등 위험 신호가 감지되었습니다. 이에 대한 적절한 응답을 생성해 주세요. "
                "특히 상담센터 정보와 같은 도움 되는 정보를 제공해 주세요."
            )
        else:
            prompt = "사용자가 부정적인 감정을 느끼고 있습니다. 이에 대한 적절한 응답을 생성해 주세요."
    else:
        prompt = "사용자의 감정에 적절한 공감 메시지를 생성해 주세요."

    full_prompt = "\n".join(context_lines + [
        prompt,
        f"현재 사용자는 특히 '{dominant_emotion}' 감정을 느끼고 있습니다.",
        f"이번 사용자 입력: {user_input}"
    ])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": full_prompt}]
    )
    return response.choices[0].message.content.strip()


def analyze_user_input(user_input):
    cleaned_input = preprocess_input_with_openai(user_input)
    return get_sentiment_score(cleaned_input)

def run_pipeline(user_input, username=None):
    sentiment_score = analyze_user_input(user_input)
    if sentiment_score is None: sentiment_score = 3
    response = generate_response(sentiment_score, user_input, username=username)
    log_interaction(user_input, sentiment_score, response, username)
    return response


def generate_farewell_message(username=None):
    client = get_openai_client()

    name_part = f"사용자 이름은 {username}입니다. " if username else ""
    prompt = (
        f"{name_part}지금까지 감정에 대해 충분히 이야기했습니다. "
        "이제 대화를 마치려고 합니다. 사용자의 감정을 존중하고 따뜻하게 배웅하는 인사말을 한국어로 자연스럽게 작성해 주세요. "
        "2문장 이내로 해주세요."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def main():
    print("감정 상담 챗봇에 오신 걸 환영합니다. 언제든지 감정을 자유롭게 표현해 주세요.")
    username = input("사용자 이름(선택, 사용자 이름은 암호화되어 저장됩니다): ").strip()

    while True:
        user_input = input("\n당신의 감정을 말해 주세요 (종료하려면 '그만할게요', '안녕' 등 입력): ").strip()

        if user_input.lower() in ["그만할게요", "수고했어요", "고마워요", "안녕", "종료", "bye"]:
            farewell = generate_farewell_message(username=username if username else None)
            print("\n" + farewell)
            break

        response = run_pipeline(user_input, username=username)
        print(f"\n챗봇: {response}")

if __name__ == "__main__":
    main()