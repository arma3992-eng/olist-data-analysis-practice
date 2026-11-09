
from pathlib import Path
import pandas as pd

# 1. 타이타닉 데이터 불러오기
df = pd.read_csv("data/train.csv")

# 2. 기본 통계
total = len(df)
survived = int(df["Survived"].sum())
deceased = total - survived

facts = [
    ("전체 분석 대상", f"분석 대상은 {total:,}명이다."),
    ("생존자 수", f"생존자는 {survived:,}명이다."),
    ("사망자 수", f"사망자는 {deceased:,}명이다."),
    ("전체 생존율", f"전체 생존율은 {survived / total:.1%}이다."),
]

# 3. 성별 생존율
for sex, label in [("female", "여성"), ("male", "남성")]:
    group = df[df["Sex"] == sex]
    facts.append((
        f"{label} 생존율",
        f"{label} {len(group)}명 중 {int(group['Survived'].sum())}명이 "
        f"생존했으며 생존율은 {group['Survived'].mean():.1%}이다."
    ))

# 4. 객실 등급별 생존율
for pclass in [1, 2, 3]:
    group = df[df["Pclass"] == pclass]
    facts.append((
        f"{pclass}등급 생존율",
        f"{pclass}등급 {len(group)}명 중 "
        f"{int(group['Survived'].sum())}명이 생존했으며 "
        f"생존율은 {group['Survived'].mean():.1%}이다."
    ))

# 5. 결측치 통계
for column, label in [
    ("Age", "나이"),
    ("Cabin", "객실 번호"),
    ("Embarked", "승선 항구"),
]:
    facts.append((
        f"{label} 결측치",
        f"{label} 정보가 없는 승객은 "
        f"{df[column].isna().sum()}명이다."
    ))

# 6. RAG용 Markdown 사실표 저장
lines = [
    "# 타이타닉 데이터 분석 사실표",
    "",
    "이 자료는 Kaggle Titanic train.csv 학습용 표본을 분석한 결과이다.",
    "타이타닉 전체 탑승객에 대한 통계가 아니다.",
    "",
]

for i, (title, content) in enumerate(facts, start=1):
    lines.extend([
        f"## F{i:02d} {title}",
        f"- 내용: {content}",
        "- 기준: Kaggle Titanic train.csv 학습용 데이터",
        "- 출처: data/train.csv",
        "",
    ])

output = Path("rag/titanic_facts.md")
output.parent.mkdir(exist_ok=True)
output.write_text("\n".join(lines), encoding="utf-8")

print(f"분석 대상: {total:,}명")
print(f"생존자: {survived:,}명")
print(f"사실표: {len(facts)}개")
print(f"저장 완료: {output}")
