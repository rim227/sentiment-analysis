import re
from datetime import datetime

import pandas as pd
import streamlit as st


# =====================================
# 0. 기본 설정 & 세션 상태
# =====================================
st.set_page_config(
    page_title="데이터 정제 기반 감성 분석 데모",
    layout="wide",
)

if "history" not in st.session_state:
    st.session_state.history = []  # 각 항목: dict
if "example_clicked" not in st.session_state:
    st.session_state.example_clicked = False


# =====================================
# 1. 감성 분석 룰 (정제 전 / 정제 후 분리)
# =====================================

POS_WORDS = [
    # 기본 긍정
    "좋다", "좋아요", "괜찮아요", "만족", "만족스러워", "만족스럽", "추천",
    "괜찮", "나쁘지 않", "이쁘", "예쁘", "예뻐", "맘에 들어", "마음에 듬",

    # 감정 표현형
    "감동", "감명", "기분 좋", "기분이 좋", "행복",

    # 품질 관련
    "튼튼", "견고", "고급", "고급스럽", "디자인 좋", "퀄리티 좋", "좋은 품질",
    "내구성 좋", "재질 좋", "색감 좋", "색감 예쁘", "재질 만족",

    # 사용 경험형
    "편해요", "편함", "편리", "잘 맞아요", "잘 맞음", "딱 맞아요", "착용감 좋",
    "체감 좋", "만족감 높",

    # 구매 의사형
    "재구매", "또 사", "의사 있음",

    # 강화 의미형 (보너스 가중치 효과)
    "최고", "최상", "완전 만족", "극호",
    "대박", "개좋", "개이쁨", "진짜 좋음", "미쳤다", "짱좋", "완전 좋",
]

NEG_WORDS = [
    # 기본 부정
    "별로", "그닥", "최악", "실망", "짜증", "화남", "화가 남", "실망스러움",
    "별로였", "별로였어요", "부족", "부실", "쓰레기", "형편없", "망함",

    # 품질 불만
    "내구성 약함", "내구성 별로", "재질 안 좋", "마감 안 좋", "박음질 별로",
    "오염", "냄새 심함", "불량", "하자", "깨짐", "비추천",

    # 사용 불편
    "불편", "답답", "불친절", "불편함", "불편하",

    # 교환/환불 이슈 유발
    "환불했습니다", "교환했습니다", "반품", "환불 요청", "교환 요청",

    # 배송 관련 강한 부정
    "너무 늦음", "배송 느림", "늦게 옴", "오배송", "배송 사고",

    # 가격 관련 불만
    "돈 아까움", "가격 대비 별로", "비싼데 별로",

    # 부정 강화 표현
    "완전 별로", "진짜 별로", "극혐", "최악 수준", "진짜 실망",
]



def count_noise(text: str) -> int:
    
    if not text:
        return 0
    noise_pattern = r"[ㅋㅎㅠ!?]+"
    return len(re.findall(noise_pattern, text))


def sentiment_before(text: str):
    """
    정제 전 모델:
    - 긍/부정 단어는 약하게 반영
    - 노이즈(ㅋㅋ, ㅠㅠ, !! 등)가 많으면 신뢰도 확 깎음
    → 대체로 '애매한 확률'이 나오도록 설계
    """
    text = text or ""
    pos = sum(1 for w in POS_WORDS if w in text)
    neg = sum(1 for w in NEG_WORDS if w in text)

    if any(word in text for word in ["완전 만족", "재구매", "대박"]):
        pos += 1

    if any(word in text for word in ["환불", "반품", "불량"]):
        neg += 1

    noise = count_noise(text)

    net = pos - neg          # 양수면 긍정, 음수면 부정
    mag = abs(net)           # 감정 강도

    if net > 0:
        label = "긍정"
    elif net < 0:
        label = "부정"
    else:
        label = "중립"

    # confidence: 기본 50에서 감정 강도는 살짝만 반영, 노이즈는 세게 깎기
    conf = 50 + mag * 5      # 0→50, 1→55, 2→60 ...
    conf -= min(noise * 5, 25)

    conf = max(min(conf, 90), 10)  # 너무 극단적이지 않게 클램프
    return label, round(float(conf), 1)


def sentiment_after(text: str):
    """
    정제 후 모델:
    - 노이즈 패널티 없음
    - 감정 강도를 훨씬 크게 반영
    → 정제 후에는 확실한 긍정/부정 확률이 나오도록 설계
    """
    import re

    text = text or ""
    pos = sum(1 for w in POS_WORDS if w in text)
    neg = sum(1 for w in NEG_WORDS if w in text)

    # -------------------------------
    # (1) 강한 긍정 가중치
    if any(word in text for word in ["완전 만족", "재구매", "또 사", "대박", "진짜 좋"]):
        pos += 2

    # (2) 강한 부정 가중치
    if any(word in text for word in ["환불", "반품", "최악", "돈 아까움", "불량", "극혐"]):
        neg += 2
    # -------------------------------

    # ======================================================
    # 🔥 (3) 결론부 감정 우선 적용
    # 문장 끝 부분(최종 사용자 판단 영역)에서 강하게 반영
    # ======================================================
    sentences = re.split('[.!?]', text)
    final_part = sentences[-1]  # 마지막 문장 영역 (결론으로 간주)

    # 결론부에서 긍정적 해석 시 강한 가중치
    if any(word in final_part for word in ["만족", "좋", "추천", "예쁘", "편리", "고급", "재구매"]):
        pos += 2

    # 결론부에서 부정적 해석 시 강한 가중치
    if any(word in final_part for word in ["별로", "최악", "환불", "불량", "실망"]):
        neg += 2

    # ======================================================
    # 🔥 (4) 부정 영향 약화
    # 정제 후에는 핵심 감정만 남는다는 가정
    # ======================================================
    neg = max(neg - 1, 0)
    # ======================================================

    net = pos - neg
    mag = abs(net)

    if net > 0:
        label = "긍정"
    elif net < 0:
        label = "부정"
    else:
        label = "중립"

    # 정제 후는 감정 강도에 과감하게 가중치
    conf = 50 + mag * 25

    conf = max(min(conf, 97), 3)
    return label, round(float(conf), 1)



# =====================================
# 2. 텍스트 정제 파이프라인
# =====================================

# 불용어
STOPWORDS = ["진짜", "그냥", "너무", "정말", "완전", "좀", "되게", "굉장히", "ㅋㅋ", "ㅠㅠ"]

# 인터넷/줄임말 치환용 맵
SLANG_MAP = {
    "ㄹㅇ": "정말",
    "ㅈㄴ": "매우",
    "개쩔": "아주 좋",
    "개좋": "아주 좋",
    "ㄱㄱ": "가자",
    "ㄱㅅ": "감사",
    "ㅇㅈ": "인정",
    "ㅂㅂ": "바이",
    "ㅋㅋ": "",
    "ㅋ": "",
    "ㅎㅎ": "",
    "ㅎ": "",
    "ㅠㅠ": "",
    "ㅠ": "",
}


def normalize_slang(text: str) -> str:
    """인터넷 줄임말/자모 표현을 조금 더 표준 형태로 바꾸기"""
    for slang, full in SLANG_MAP.items():
        text = text.replace(slang, full)
    return text


def clean_text_pipeline(
    text: str,
    use_special: bool,
    use_stopword: bool,
    use_norm: bool,
):
    """
    정제 단계별 텍스트 / 토큰 수 / 시스템 로그를 반환
    """
    steps = []

    # STEP 1: 원본
    current = text
    steps.append(
        {
            "name": "STEP 1. 원본",
            "desc": "사용자가 입력한 원본 문장",
            "text": current,
        }
    )

    # STEP 2: 특수문자 제거
    if use_special:
        # 이모티콘·기호 등 제거
        current = re.sub(r"[^가-힣0-9a-zA-Z\s]", " ", current)
        # 줄 끝/단독으로 쓰이는 ㅎ, ㅎㅎ 제거 (구어체 웃음)
        current = re.sub(r"(ㅎ)+(\s|$)", " ", current)
        current = re.sub(r"\s+", " ", current).strip()
        steps.append(
            {
                "name": "STEP 2. 특수문자 제거",
                "desc": "이모티콘, 기호, 구어체 ㅎ/ㅎㅎ 등을 제거",
                "text": current,
            }
        )

    # STEP 3: 불용어 제거
    if use_stopword:
        tmp = current
        for sw in STOPWORDS:
            tmp = tmp.replace(sw, " ")
        tmp = re.sub(r"\s+", " ", tmp).strip()
        current = tmp
        steps.append(
            {
                "name": "STEP 3. 불용어 제거",
                "desc": "의미 기여도가 낮은 구어체·부사 등을 제거",
                "text": current,
            }
        )

    # STEP 4: 정규화 (인터넷 표현 + 반복 문자)
    if use_norm:
        # 4-1. 인터넷/줄임말 정규화
        current = normalize_slang(current)

        # 4-2. 같은 글자 반복 줄이기 (ㅋㅋㅋㅋ → ㅋ 등)
        current = re.sub(r"(.)\1{2,}", r"\1", current)

        # 4-3. 공백 정리
        current = re.sub(r"\s+", " ", current).strip()

        steps.append(
            {
                "name": "STEP 4. 정규화",
                "desc": "인터넷 줄임말과 반복 문자, 공백 등을 정리하여 문장을 단순화",
                "text": current,
            }
        )

    original_tokens = len((text or "").split())
    cleaned_tokens = len((current or "").split())

    log_lines = ["> Input string loaded."]
    if use_special:
        log_lines.append("> Regex applied: [^가-힣0-9a-zA-Z\\s] & ㅎ/ㅎㅎ removed.")
    if use_stopword:
        log_lines.append("> Removing stopwords: " + ", ".join(STOPWORDS[:4]) + " ...")
    if use_norm:
        log_lines.append("> Normalizing slang & repeated chars.")
    log_lines.append("> Process completed.")
    sys_log = "\n".join(log_lines)

    return current, steps, original_tokens, cleaned_tokens, sys_log


# =====================================
# 3. 상단 레이아웃 (입력 + 설명)
# =====================================

st.markdown("### 데이터 정제 기반 감성 분석 데모")

top_left, top_right = st.columns([3, 2], gap="large")

with top_left:
    st.markdown("**리뷰 텍스트 입력**")
    default_placeholder = "예) 배송이 너무 늦게 와서 화가 났지만 제품은 정말 좋네요"
    review_text = st.text_area(
        "",
        height=200,
        placeholder=default_placeholder,
    )

    st.markdown("**정제 단계 선택**")
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        opt_special = st.checkbox("특수문자 제거", value=True)
    with col_opt2:
        opt_stopword = st.checkbox("불용어 제거", value=True)
    with col_opt3:
        opt_norm = st.checkbox("정규화", value=True)

    col_btn1, col_btn2 = st.columns([2, 1])
    with col_btn1:
        run_btn = st.button("분석하기", use_container_width=True)
    with col_btn2:
        example_btn = st.button("예시 문장", use_container_width=True)

    if example_btn:
        st.session_state.example_clicked = True

    if st.session_state.example_clicked and not review_text:
        review_text = "배송이 진짜 너무 늦게 와서 짜증났는데ㅜㅜ 근데 색감은 핵예쁘고 ㄹㅇ 퀄리티 미쳤어요ㅋㅋㅋ ㅎ"
        st.session_state.example_clicked = False

with top_right:
    st.markdown("**💡 시스템 설명**")
    st.markdown(
        """
- 정제 전/후 두 개의 모델이 **동시에 감성을 예측**합니다.
- 동일한 문장에 대해 정제 유무에 따른 결과 차이를 비교할 수 있습니다.
- 하단 탭에서 **결과 요약 / 정제 과정 / 히스토리**를 순서대로 확인할 수 있습니다.
        """
    )

st.markdown("---")

# =====================================
# 4. 분석 실행
# =====================================

analysis_ran = False
if run_btn and review_text.strip():
    analysis_ran = True

    # 정제
    cleaned_text, steps, tok_before, tok_after, sys_log = clean_text_pipeline(
        review_text,
        use_special=opt_special,
        use_stopword=opt_stopword,
        use_norm=opt_norm,
    )

    # 정제 전 / 후 감성 예측
    label_before, prob_before = sentiment_before(review_text)
    label_after, prob_after = sentiment_after(cleaned_text)

    # 향상 여부
    if prob_after > prob_before + 1:
        status = "향상"
    elif prob_after < prob_before - 1:
        status = "저하"
    else:
        status = "유지"

    # 히스토리 저장
    st.session_state.history.append(
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "snippet": (review_text[:18] + "...") if len(review_text) > 18 else review_text,
            "before_label": label_before,
            "before_prob": prob_before,
            "after_label": label_after,
            "after_prob": prob_after,
            "status": status,
        }
    )

# =====================================
# 5. 하단 탭: 결과 요약 / 정제 과정 / 히스토리
# =====================================

tab_summary, tab_process, tab_history = st.tabs(
    ["결과 요약", "정제 과정 보기", "히스토리"]
)

# ----- 5-1. 결과 요약 탭 -----
with tab_summary:
    if not analysis_ran:
        st.info("위에서 문장을 입력하고 [분석하기] 버튼을 누르면 결과가 여기에 표시됩니다.")
    else:
        left, right = st.columns([3, 2], gap="large")

        with left:
            st.markdown("#### 🔎 텍스트 비교")

            st.markdown("**원본 텍스트**")
            st.markdown(f"> \"{review_text}\"")

            st.markdown("⬇️")

            st.markdown("**정제 후 텍스트**")
            st.markdown(f"> \"{cleaned_text}\"")

        with right:
            st.markdown("#### 📊 모델 예측 비교")

            st.markdown(f"- 정제 전 모델 : **{label_before} ({prob_before}%)**")
            st.progress(min(int(prob_before), 100))

            st.markdown(f"- 정제 후 모델 : **{label_after} ({prob_after}%)**")
            st.progress(min(int(prob_after), 100))

            st.markdown("")
            st.markdown("**확률 비교 그래프**")

            chart_df = pd.DataFrame(
                {
                    "정제 전": [prob_before],
                    "정제 후": [prob_after],
                }
            )
            st.bar_chart(chart_df)

            diff = round(prob_after - prob_before, 1)
            if diff > 0:
                st.success(f"정제 후 모델에서 정확도가 **{diff}％p** 향상되었습니다.")
            elif diff < 0:
                st.warning(f"정제 후 모델에서 정확도가 **{abs(diff)}％p** 감소되었습니다.")
            else:
                st.info("정제 전/후 정확도에 큰 차이가 없습니다.")

        st.markdown("")
        st.markdown("▶ 다른 문장을 입력하여 추가로 비교할 수 있습니다.")

# ----- 5-2. 정제 과정 보기 탭 -----
with tab_process:
    if not analysis_ran:
        st.info("최근 분석 결과가 없습니다. 먼저 문장을 분석해주세요.")
    else:
        col_timeline, col_log = st.columns([3, 2], gap="large")

        with col_timeline:
            st.markdown("#### 단계별 텍스트 변화")

            for i, step in enumerate(steps, start=1):
                st.markdown(f"**{step['name']}**")
                st.caption(step["desc"])
                st.markdown(
                    f"<div style='padding:8px 10px;border-radius:6px;"
                    f"border:1px solid #e0e0e0;background-color:#fafafa;'>"
                    f"{step['text']}</div>",
                    unsafe_allow_html=True,
                )
                if i != len(steps):
                    st.markdown("⬇️")

            st.markdown("")
            st.markdown(
                f"- 정제 전 토큰 수: **{tok_before}**  →  정제 후 토큰 수: **{tok_after}**"
            )

        with col_log:
            st.markdown("#### System Log")
            st.code(sys_log, language="bash")

# ----- 5-3. 히스토리 탭 -----
with tab_history:
    if len(st.session_state.history) == 0:
        st.info("아직 저장된 히스토리가 없습니다. 문장을 한 번 이상 분석하면 이력이 표시됩니다.")
    else:
        left_hist, right_hist = st.columns([3, 2], gap="large")

        with left_hist:
            st.markdown("#### 최근 분석 기록")

            df = pd.DataFrame(st.session_state.history)
            df_display = df.copy()
            df_display["BEFORE"] = (
                df_display["before_label"] + " (" + df_display["before_prob"].astype(str) + "%)"
            )
            df_display["AFTER"] = (
                df_display["after_label"] + " (" + df_display["after_prob"].astype(str) + "%)"
            )
            df_display["STATUS"] = df_display["status"].map(
                {"향상": "향상", "저하": "저하", "유지": "유지"}
            )

            df_display = df_display[["time", "snippet", "BEFORE", "AFTER", "STATUS"]]
            df_display.columns = ["TIME", "TEXT SNIPPET", "BEFORE", "AFTER", "STATUS"]

            st.dataframe(df_display, use_container_width=True, height=260)

        with right_hist:
            st.markdown("#### 통계 요약")

            total = len(st.session_state.history)
            improved = sum(1 for h in st.session_state.history if h["status"] == "향상")
            kept = sum(1 for h in st.session_state.history if h["status"] == "유지")
            dropped = sum(1 for h in st.session_state.history if h["status"] == "저하")

            if total > 0:
                improved_ratio = round(improved / total * 100, 1)
            else:
                improved_ratio = 0.0

            st.markdown("**성능 향상 비율**")
            st.markdown(f"- 향상됨: **{improved_ratio}%**")
            st.markdown(f"- 유지/저하: **{100 - improved_ratio}%**")

            st.markdown("---")

            pos_cnt = sum(1 for h in st.session_state.history if h["after_label"] == "긍정")
            neg_cnt = sum(1 for h in st.session_state.history if h["after_label"] == "부정")
            neu_cnt = sum(1 for h in st.session_state.history if h["after_label"] == "중립")

            if total > 0:
                pos_ratio = round(pos_cnt / total * 100, 1)
                neg_ratio = round(neg_cnt / total * 100, 1)
                neu_ratio = round(neu_cnt / total * 100, 1)
            else:
                pos_ratio = neg_ratio = neu_ratio = 0.0

            st.markdown("**감성 분포 (정제 후 기준)**")
            st.write(f"긍정: {pos_ratio}%")
            st.progress(min(int(pos_ratio), 100))
            st.write(f"부정: {neg_ratio}%")
            st.progress(min(int(neg_ratio), 100))
            st.write(f"중립: {neu_ratio}%")
            st.progress(min(int(neu_ratio), 100))
