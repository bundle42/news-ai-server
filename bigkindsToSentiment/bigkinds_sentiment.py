import os
import glob
import pandas as pd
from transformers import pipeline

# =========================
# 종목별 폴더 설정
# =========================
stock_folders = {
    "삼성전자": r"C:\springboot_img\bigkinds\삼성전자",
    "SK하이닉스": r"C:\springboot_img\bigkinds\SK하이닉스",
    "현대차": r"C:\springboot_img\bigkinds\현대차"
}

# =========================
# 감정분석 모델 로드 (한 번만)
# =========================
print("감정분석 모델 로드 중...")
sentiment_pipeline = pipeline(
    "text-classification",
    model="snunlp/KR-FinBert-SC"
)
print("모델 로드 완료!")

# =========================
# 설정
# =========================
BATCH_SIZE = 16          # 모델 추론 배치
SAVE_INTERVAL = 100      # 중간 저장 간격(기사 수)

# =========================
# 종목별 처리
# =========================
for query, input_folder in stock_folders.items():
    print("\n" + "#" * 70)
    print(f"[{query}] 처리 시작")

    output_folder = os.path.join(input_folder, "csv")
    os.makedirs(output_folder, exist_ok=True)

    excel_files = glob.glob(os.path.join(input_folder, "NewsResult_*.xlsx"))

    if not excel_files:
        print(f"[{query}] 엑셀 파일 없음")
        continue

    print(f"[{query}] 총 {len(excel_files)}개 파일 발견")

    for input_file in excel_files:
        try:
            base_name = os.path.basename(input_file)
            date_part = base_name.replace("NewsResult_", "").replace(".xlsx", "")
            output_file = os.path.join(output_folder, f"news_sentiment_{date_part}.csv")
            temp_file = os.path.join(output_folder, f"news_sentiment_{date_part}_temp.csv")

            # 이미 최종 결과 파일이 존재하면 스킵
            if os.path.exists(output_file):
                print(f"[{query}] 이미 처리 완료, 스킵: {base_name}")
                continue

            print(f"\n처리 중: {base_name}")

            # 엑셀 읽기
            df = pd.read_excel(input_file)

            required_cols = ["일자", "제목", "본문", "URL"]
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"'{col}' 컬럼 없음")

            # 전처리
            df["일자"] = df["일자"].astype(str).str.strip()
            df["제목"] = df["제목"].fillna("").astype(str).str.strip()
            df["본문"] = df["본문"].fillna("").astype(str).str.strip()
            df["URL"] = df["URL"].fillna("").astype(str).str.strip()

            # 컬럼명 변경
            df = df.rename(columns={
                "일자": "date",
                "제목": "title",
                "본문": "contents",
                "URL": "newsLink"
            })

            # 날짜 형식 통일
            df["date"] = pd.to_datetime(
                df["date"], format="%Y%m%d", errors="coerce"
            ).dt.strftime("%Y-%m-%d")

            # 제목 리스트 준비
            titles = df["title"].fillna("").astype(str).str[:512].tolist()
            cleaned_titles = [t if t.strip() else "중립 기사" for t in titles]

            labels = []
            confidences = []
            sentiment_scores = []

            total = len(cleaned_titles)
            print(f"총 기사 수: {total}")

            # =========================
            # 배치 단위 처리 + 중간 저장
            # =========================
            for start in range(0, total, BATCH_SIZE):
                end = min(start + BATCH_SIZE, total)
                batch_titles = cleaned_titles[start:end]

                results = sentiment_pipeline(batch_titles, batch_size=BATCH_SIZE)

                for result in results:
                    label = result["label"].lower()
                    confidence = float(result["score"])

                    if label == "positive":
                        sentiment_score = confidence
                    elif label == "negative":
                        sentiment_score = -confidence
                    else:
                        sentiment_score = 0.0

                    labels.append(label)
                    confidences.append(confidence)
                    sentiment_scores.append(sentiment_score)

                # 진행률 출력
                print(f"[{query}] {base_name} 진행률: {end}/{total}")

                # 중간 저장
                if end % SAVE_INTERVAL == 0 or end == total:
                    temp_df = df.iloc[:end].copy()
                    temp_df["label"] = labels
                    temp_df["confidence"] = confidences
                    temp_df["sentiment_score"] = sentiment_scores
                    temp_df["searchQuery"] = query

                    temp_result_df = temp_df[[
                        "date", "title", "label", "confidence", "sentiment_score",
                        "contents", "newsLink", "searchQuery"
                    ]]

                    temp_result_df.to_csv(temp_file, index=False, encoding="utf-8-sig")
                    print(f"중간 저장 완료: {temp_file} ({end}/{total})")

            # =========================
            # 최종 저장
            # =========================
            df["label"] = labels
            df["confidence"] = confidences
            df["sentiment_score"] = sentiment_scores
            df["searchQuery"] = query

            result_df = df[[
                "date", "title", "label", "confidence", "sentiment_score",
                "contents", "newsLink", "searchQuery"
            ]]

            result_df.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"최종 저장 완료: {output_file}")

            # temp 파일 삭제 (선택)
            if os.path.exists(temp_file):
                os.remove(temp_file)

        except Exception as e:
            print(f"[{query}] 오류 - {os.path.basename(input_file)}: {e}")

print("\n전체 종목 처리 완료!")