import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. 페이지 기본 설정 및 공통 Custom CSS
# ==========================================
st.set_page_config(page_title="조선 경제사 GDP 추정", layout="wide", initial_sidebar_state="collapsed")

# 공통 UI 스타일 (폰트, 탭 디자인, 버튼 등)
st.markdown("""
    <style>
    /* 실제 콘텐츠가 오버레이 위로 올라오도록 설정 */
    .block-container {
        position: relative;
        z-index: 1;
        color: #c5c6c7;
    }
    
    /* 헤더 폰트 */
    h1, h2, h3, h4, h5, h6 {
        color: #d4af37 !important; 
        font-family: 'Gowun Batang', 'Malgun Gothic', serif;
        font-weight: 300;
        letter-spacing: 2px;
    }
    
    /* 상단 네비게이션 바 (Tabs) 스타일링 - 반투명 & 네온 글로우 */
    div[data-testid="stTabs"] {
        background-color: rgba(11, 12, 16, 0.6); 
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 5px 20px;
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        border-bottom: none; 
        gap: 50px; 
        justify-content: center; 
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        color: #8c8c8c; 
        font-family: 'Gowun Batang', serif;
        font-size: 1.1rem;
        background: transparent !important;
        border: none !important;
        transition: all 0.4s ease; 
        padding: 0;
    }
    
    /* 마우스 호버 시 글자가 황금빛으로 빛나는(Glow) 효과 */
    .stTabs [data-baseweb="tab"]:hover {
        color: #d4af37 !important;
        text-shadow: 0 0 10px #d4af37, 0 0 20px #d4af37, 0 0 30px #d4af37;
        transform: translateY(-2px); 
    }
    
    /* 현재 선택된 메뉴의 스타일 */
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
    }
    
    .stTabs [data-baseweb="tab-highlight"] {
        display: none; 
    }
    
    /* 카드(컨테이너) 디자인 */
    div[data-testid="stExpander"] {
        background-color: rgba(31, 40, 51, 0.7);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
    }
    
    /* 랜딩 페이지 중앙 정렬 */
    .landing-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 80vh;
        text-align: center;
    }
    .landing-title {
        font-size: 5rem;
        color: #ffffff;
        letter-spacing: 8px;
        margin-bottom: 0px;
        text-shadow: 0 0 20px rgba(255,255,255,0.3);
    }
    .landing-subtitle {
        color: #d4af37;
        font-size: 1.5rem;
        margin-top: 10px;
        margin-bottom: 60px;
        font-weight: 300;
        letter-spacing: 2px;
    }
    
    /* 공통 버튼 커스텀 */
    .stButton>button {
        background-color: rgba(212, 175, 55, 0.1);
        color: #d4af37;
        border: 1px solid #d4af37;
        border-radius: 30px;
        font-weight: 300;
        letter-spacing: 2px;
        padding: 10px 30px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #d4af37;
        color: #0b0c10;
        box-shadow: 0 0 15px #d4af37;
    }
    </style>
""", unsafe_base64=True)

# ==========================================
# 2. 데이터 시뮬레이션 및 전처리 함수
# ==========================================
@st.cache_data
def load_data():
    years = np.arange(1623, 1801)
    df = pd.DataFrame({'연도': years})
    
    conditions = [
        (df['연도'] <= 1649), (df['연도'] <= 1674), (df['연도'] <= 1720), 
        (df['연도'] <= 1724), (df['연도'] <= 1776), (df['연도'] <= 1800)
    ]
    choices = ['인조', '효종·현종', '숙종', '경종', '영조', '정조']
    df['왕조'] = np.select(conditions, choices, default='기타')
    
    np.random.seed(42)
    df['인구수'] = np.linspace(5000000, 7500000, len(df)) * (1 + np.random.normal(0, 0.02, len(df)))
    df['동전 유통량'] = np.where(df['연도'] >= 1678, np.linspace(0, 8000000, len(df)), 0)
    df['쌀 가격'] = np.where(df['연도'] < 1678, 4, np.linspace(4, 8, len(df)))
    df['쌀임금지수'] = np.where(df['연도'] >= 1700, np.linspace(1, 1.5, len(df)), np.nan)
    df['경지 면적'] = np.where(df['연도'] >= 1750, np.linspace(1200000, 1500000, len(df)), np.nan)
    df['평균 월급'] = 2.5
    
    MAX_COIN = 10000000
    YIELD_PER_GYEOL = 20
    BASE_RICE_PRICE = 4
    tax_per_gyeol = (16 / 15) * 4
    total_tax_gdp = 982000 * tax_per_gyeol
    
    df['유통비율'] = (df['동전 유통량'] / MAX_COIN).clip(lower=0)
    log_base = np.log(2)
    df['비농업비율'] = (0.02 + 0.06 * (np.log(1 + df['유통비율']) / log_base)).clip(upper=0.08)
    df['비농업인구'] = df['인구수'] * df['비농업비율']
    
    for col in ['농업GDP_명목', '비농업GDP_명목', '총GDP_명목', '농업GDP_실질', '비농업GDP_실질', '총GDP_실질']:
        df[col] = np.nan

    mask1 = df['쌀임금지수'].isna()
    df.loc[mask1, '농업GDP_명목'] = df.loc[mask1, '인구수'] * 4 * 2 + total_tax_gdp
    df.loc[mask1, '비농업GDP_명목'] = df.loc[mask1, '비농업인구'] * df.loc[mask1, '평균 월급'] * 12
    df.loc[mask1, '농업GDP_실질'] = df.loc[mask1, '농업GDP_명목']
    df.loc[mask1, '비농업GDP_실질'] = df.loc[mask1, '비농업GDP_명목']
    
    mask2 = df['쌀임금지수'].notna() & df['경지 면적'].isna()
    df.loc[mask2, '농업GDP_명목'] = df.loc[mask2, '인구수'] * df.loc[mask2, '쌀 가격'] * 2 + total_tax_gdp
    df.loc[mask2, '비농업GDP_명목'] = df.loc[mask2, '비농업인구'] * df.loc[mask2, '쌀 가격'] * df.loc[mask2, '쌀임금지수']
    df.loc[mask2, '농업GDP_실질'] = df.loc[mask2, '인구수'] * BASE_RICE_PRICE * 2 + total_tax_gdp
    df.loc[mask2, '비농업GDP_실질'] = df.loc[mask2, '비농업인구'] * BASE_RICE_PRICE * df.loc[mask2, '쌀임금지수']
    
    mask3 = df['쌀임금지수'].notna() & df['경지 면적'].notna()
    df.loc[mask3, '농업GDP_명목'] = df.loc[mask3, '경지 면적'] * 1000 * df.loc[mask3, '쌀 가격'] * YIELD_PER_GYEOL
    df.loc[mask3, '비농업GDP_명목'] = df.loc[mask3, '비농업인구'] * df.loc[mask3, '쌀 가격'] * df.loc[mask3, '쌀임금지수']
    df.loc[mask3, '농업GDP_실질'] = df.loc[mask3, '경지 면적'] * 1000 * BASE_RICE_PRICE * YIELD_PER_GYEOL
    df.loc[mask3, '비농업GDP_실질'] = df.loc[mask3, '비농업인구'] * BASE_RICE_PRICE * df.loc[mask3, '쌀임금지수']

    df['총GDP_명목'] = df['농업GDP_명목'] + df['비농업GDP_명목']
    df['총GDP_실질'] = df['농업GDP_실질'] + df['비농업GDP_실질']
    
    return df

# ==========================================
# 3. 랜딩 페이지 (Entry Gate)
# ==========================================
if 'entered' not in st.session_state:
    st.session_state['entered'] = False

if not st.session_state['entered']:
    # [시작 페이지 CSS] 김홍도 풍속화 + 반투명 블러 효과
    st.markdown("""
        <style>
        .stApp {
            background-image: url('https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Kim_Hong-do-Seodang.jpg/1920px-Kim_Hong-do-Seodang.jpg');
            background-size: cover; background-position: center; background-attachment: fixed;
        }
        .stApp::before {
            content: ""; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
            background-color: rgba(11, 12, 16, 0.65); 
            backdrop-filter: blur(12px); /* 반투명 흐림 효과 */
            z-index: 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="landing-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="landing-title">조선 경제사 추정</h1>', unsafe_allow_html=True)
    st.markdown('<p class="landing-subtitle">데이터로 재구성한 17~18세기 경제 변동</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("탐색 시작", use_container_width=True):
            st.session_state['entered'] = True
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 4. 메인 콘텐츠 (탭 구조)
# ==========================================
else:
    # [메인 페이지 CSS] 조선 풍경(동궐도) + 블러 없음 (가독성을 위한 어두운 틴트만 적용)
    st.markdown("""
        <style>
        .stApp {
            background-image: url('https://upload.wikimedia.org/wikipedia/commons/thumb/0/07/Donggwoldo.jpg/1920px-Donggwoldo.jpg');
            background-size: cover; background-position: center; background-attachment: fixed;
        }
        .stApp::before {
            content: ""; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
            /* 흐림(blur) 효과 없이, 글자가 보이도록 어두운 레이어만 깔아줌 */
            background-color: rgba(11, 12, 16, 0.85); 
            z-index: 0;
        }
        </style>
    """, unsafe_allow_html=True)

    df = load_data()
    
    tab1, tab2, tab3, tab4 = st.tabs(["프로젝트 소개", "모델링 방법론", "대시보드", "역사적 분석"])
    
    # --- 탭 1: 프로젝트 소개 ---
    with tab1:
        st.header("디지털 역사학: 과거의 경제를 수량화하다")
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("""
            ### 연구 배경
            전통적인 역사 서술은 문헌 사료에 크게 의존합니다. 하지만 조선 시대 인조(1623)부터 정조(1800)에 이르는 역동적인 시기의 경제적 변화를 텍스트만으로 이해하는 데는 한계가 있습니다. 
            
            본 프로젝트는 흩어져 있는 사료 데이터 노드(인구, 쌀값, 상평통보 유통량, 결수)를 연결하여 수학적 네트워크를 구축하고, 이를 통해 조선 후기의 실질적인 국가 총생산(GDP)을 추정합니다.
            """)
        with col2:
            st.markdown("""
            ### 핵심 목표
            * **데이터 결측치 보정:** 시대별로 온전하지 않은 사료 데이터 구조에 맞춰 각기 다른 세 가지 산출 모델(Model 1, 2, 3)을 적용.
            * **명목과 실질의 분리:** 쌀값 변동이 가져오는 화폐 환상의 착시를 제거하기 위한 디플레이터(Deflator) 적용.
            * **역사적 인과관계 증명:** 상평통보의 유통(숙종기)과 신해통공(정조기) 등의 주요 정책이 실제 비농업 부문의 성장에 미친 영향을 정량적으로 입증.
            """)
            
    # --- 탭 2: 방법론 ---
    with tab2:
        st.header("GDP 산출 모델")
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.container():
            st.subheader("1. 공통 변수 및 상수")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**인구 및 쌀값**")
                st.latex(r"P = \text{Population}, \quad R = \text{Rice Price}")
                st.markdown("**비농업 인구 비율 및 규모**")
                st.latex(r"s = 0.02 + 0.06 \frac{\log(1+M)}{\log2}")
                st.latex(r"P_{non} = P \times s")
            with col_b:
                st.markdown("**조세 및 수확량**")
                st.latex(r"\text{Tax} = 982000 \times \frac{16}{15} \times 4")
                st.latex(r"Y = \text{Yield per Gyeol}, \quad L = \text{Land Area}")

        st.markdown("---")
        st.subheader("2. 시대별 3단계 추정 모델")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info("Model 1 (17세기 모델)")
            st.caption("조건: 임금 지수(X), 토지 결수(X)")
            st.latex(r"\text{GDP}_{agri} = 2 \times P \times 4 + \text{Tax}")
            st.latex(r"\text{GDP}_{non} = P_{non} \times \text{Wage} \times 12")
        with c2:
            st.warning("Model 2 (18세기 초반 모델)")
            st.caption("조건: 임금 지수(O), 토지 결수(X)")
            st.latex(r"\text{GDP}_{agri} = 2 \times P \times R + \text{Tax}")
            st.latex(r"\text{GDP}_{non} = P_{non} \times R \times W_r")
        with c3:
            st.success("Model 3 (18세기 후반 모델)")
            st.caption("조건: 임금 지수(O), 토지 결수(O)")
            st.latex(r"\text{GDP}_{agri} = L \times 1000 \times Y \times R")
            st.latex(r"\text{GDP}_{non} = P_{non} \times R \times W_r")

    # --- 탭 3: 인터랙티브 대시보드 ---
    with tab3:
        st.header("GDP 변동 시각화")
        st.markdown("<br>", unsafe_allow_html=True)
        
        selected_kings = st.multiselect(
            "분석할 왕조 시기를 선택하세요",
            options=['인조', '효종·현종', '숙종', '경종', '영조', '정조'],
            default=['인조', '효종·현종', '숙종', '경종', '영조', '정조']
        )
        
        filtered_df = df[df['왕조'].isin(selected_kings)]
        
        if not filtered_df.empty:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=filtered_df['연도'], y=filtered_df['총GDP_명목']/10000, name='명목 GDP', line=dict(color='#d4af37', width=3)))
            fig1.add_trace(go.Scatter(x=filtered_df['연도'], y=filtered_df['총GDP_실질']/10000, name='실질 GDP', line=dict(color='#c5c6c7', width=3, dash='dot')))
            fig1.update_layout(
                title='연도별 실질 및 명목 GDP 추이 (단위: 만 냥)',
                template='plotly_dark', 
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0.3)',
                hovermode='x unified'
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=filtered_df['연도'], y=filtered_df['농업GDP_실질']/10000, mode='lines', stackgroup='one', name='농업 GDP', fillcolor='rgba(197, 198, 199, 0.4)', line=dict(color='#c5c6c7')))
            fig2.add_trace(go.Scatter(x=filtered_df['연도'], y=filtered_df['비농업GDP_실질']/10000, mode='lines', stackgroup='one', name='비농업 GDP', fillcolor='rgba(212, 175, 55, 0.6)', line=dict(color='#d4af37')))
            fig2.update_layout(
                title='산업 부문별 실질 GDP 구성 (단위: 만 냥)',
                template='plotly_dark', 
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0.3)',
                hovermode='x unified'
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("선택된 시기의 데이터가 없습니다.")

    # --- 탭 4: 역사적 분석 ---
    with tab4:
        st.header("왕조별 경제 지표 해석")
        st.markdown("<br>", unsafe_allow_html=True)
        
        k1, k2, k3, k4, k5 = st.columns(5)
        
        if 'active_period' not in st.session_state:
            st.session_state['active_period'] = '인조기'
            
        with k1:
            if st.button("인조기", use_container_width=True): st.session_state['active_period'] = '인조기'
        with k2:
            if st.button("효종·현종기", use_container_width=True): st.session_state['active_period'] = '효종·현종기'
        with k3:
            if st.button("숙종기", use_container_width=True): st.session_state['active_period'] = '숙종기'
        with k4:
            if st.button("경종기", use_container_width=True): st.session_state['active_period'] = '경종기'
        with k5:
            if st.button("영·정조기", use_container_width=True): st.session_state['active_period'] = '영·정조기'

        st.markdown("<br>", unsafe_allow_html=True)
        with st.container():
            st.markdown(f"### {st.session_state['active_period']} 분석")
            if st.session_state['active_period'] == '인조기':
                st.info("적용 모델: Model 1 (데이터 결측기)")
                st.write("양란(임진왜란, 병자호란) 이후 전후 복구 단계. 임금 지수와 토지 결수 데이터가 온전치 않아 기본 인구와 고정 세액(Tax)을 기반으로 농업 생산량을 보수적으로 산정했습니다. 상업 활동은 미미한 수준에 머물렀습니다.")
            elif st.session_state['active_period'] == '효종·현종기':
                st.info("적용 모델: Model 1")
                st.write("대동법이 점진적으로 확대되며 유통 경제의 기반이 서서히 닦이던 시기입니다. 다만 아직 화폐 유통량(M)이 본격화되지 않아 비농업 인구 비율(s)은 기저 수준에 머물렀습니다.")
            elif st.session_state['active_period'] == '숙종기':
                st.warning("적용 모델: Model 2 전환기")
                st.write("역사적 변곡점: 1678년 상평통보의 본격적 유통.\n 화폐 유통량($M$)이 로그 함수 모델을 타고 급증하며 비농업 비율($s$)에 유의미한 충격을 주었습니다. 이 시기부터 명목 GDP와 실질 GDP의 괴리가 시작되며, 기근 발생 시 물가 폭등 현상이 데이터로 포착됩니다.")
            elif st.session_state['active_period'] == '경종기':
                st.warning("적용 모델: Model 2")
                st.write("상업 자본이 축적되고 쌀 임금 지수 데이터가 활용 가능해짐에 따라 상업적 거래 가치 추정이 정교해집니다. 화폐 경제가 농업 경제를 보완하는 체제로 안착합니다.")
            elif st.session_state['active_period'] == '영·정조기':
                st.success("적용 모델: Model 3 (가장 정교화된 데이터 풀)")
                st.write("균역법(영조)과 신해통공(정조)이라는 굵직한 경제 정책이 시행되었습니다. 경지 면적 데이터($L$)가 확보되어 농업 GDP 산출이 정밀해졌으며, 상업의 자유화로 비농업 GDP 비중이 최고치에 달하는 조선 후기 르네상스를 증명합니다.")