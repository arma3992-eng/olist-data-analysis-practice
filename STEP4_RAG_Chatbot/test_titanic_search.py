
from pathlib import Path
import os
import re
from getpass import getpass

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

# 1. 타이타닉 사실표 읽기
base = Path(__file__).resolve().parents[1]
fact_path = base / "rag" / "titanic_facts.md"

text = fact_path.read_text(encoding="utf-8")

# 2. 사실표를 근거 ID 단위로 분리
blocks = re.split(r"(?=^## F\d{2}\s)", text, flags=re.MULTILINE)

docs = []

for block in blocks:
    match = re.match(r"## (F\d{2})\s", block)
    if match:
        docs.append(
            Document(
                page_content=block.strip(),
                metadata={"id": match.group(1)}
            )
        )

print(f"등록할 근거: {len(docs)}개")

# 3. OpenAI API 키 불러오기
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    api_key = getpass("OpenAI API 키 입력(화면에 표시되지 않음): ")

# 4. 임베딩 및 벡터 저장소 생성
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key
)

store = InMemoryVectorStore.from_documents(docs, embeddings)

print("벡터 저장소 생성 완료")

# 5. 질문과 관련된 근거 검색
question = "타이타닉 승객의 국적별 생존율은 얼마야?"

results = store.similarity_search(question, k=3)

print(f"\n질문: {question}")

for doc in results:
    print(f"\n근거 ID: {doc.metadata['id']}")
    print(doc.page_content)


# 6. 검색된 근거를 이용한 AI 답변
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

context = "\n\n".join(
    f"[{doc.metadata['id']}]\n{doc.page_content}"
    for doc in results
)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "너는 타이타닉 데이터 분석 챗봇이다.\n"
     "반드시 제공된 근거만 사용해 답변하라.\n"
     "답변에 근거 ID를 [F05] 형태로 표시하라.\n"
     "자료는 Kaggle 학습용 표본이며 전체 탑승객이 아니다.\n"
     "근거가 부족하면 확인할 수 없다고 답하라."),
    ("human", "근거:\n{context}\n\n질문: {question}")
])

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=api_key,
    temperature=0
)

response = (prompt | llm).invoke({
    "context": context,
    "question": question
})

print("\n=== RAG 챗봇 답변 ===")
print(response.content)
