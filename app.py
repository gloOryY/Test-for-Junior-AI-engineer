import os
from pathlib import Path
import streamlit as st
from financial_assistant import FinancialAssistant

st.set_page_config(page_title="Financial AI Assistant MVP", page_icon="📊", layout="wide")
st.title("Financial AI Assistant")
st.caption("MVP-прототип для тестового задания Junior AI Engineer")

DEFAULT_CSV = Path("financial_data.csv")

with st.sidebar:
    st.header("Настройки")
    api_key = st.text_input("OPENAI_API_KEY", type="password", help="Опционально: если не указать ключ, приложение использует fallback-ответы.")
    model = st.text_input("Модель", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    base_url = st.text_input("Base URL", value=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    uploaded = st.file_uploader("CSV с финансовыми данными", type=["csv"])

csv_path = None
if uploaded is not None:
    temp_path = Path("uploaded_financial_data.csv")
    temp_path.write_bytes(uploaded.getvalue())
    csv_path = temp_path
elif DEFAULT_CSV.exists():
    csv_path = DEFAULT_CSV

if not csv_path:
    st.warning("Положите financial_data.csv рядом с приложением или загрузите CSV через sidebar.")
    st.stop()

assistant = FinancialAssistant(csv_path=csv_path, api_key=api_key or None, model=model, base_url=base_url)
df = assistant.df
best = df.loc[df['revenue_growth_pct'].idxmax()]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Период", f"{int(df['year'].min())}-{int(df['year'].max())}")
c2.metric("Выручка 2024", f"${df.iloc[-1]['revenue']:,.0f}")
c3.metric("Чистая прибыль 2024", f"${df.iloc[-1]['net_income']:,.0f}")
c4.metric("Макс. рост выручки", f"{int(best['year'])}", f"{best['revenue_growth_pct']:.2f}%")

left, right = st.columns([1.15, 1])
with left:
    st.subheader("Финансовые данные")
    st.dataframe(df, use_container_width=True)
with right:
    st.subheader("Примеры вопросов")
    presets = [
        "В каком году был самый быстрый рост выручки?",
        "Как изменялась прибыльность компании со временем?",
        "Объясните динамику операционной маржи компании.",
        "Сравни 2020 и 2024 годы по ключевым метрикам."
    ]
    selected = st.selectbox("Выберите пример", options=[""] + presets)

st.subheader("Вопрос пользователю")
question = st.text_area("Введите вопрос", value=selected)

if st.button("Получить ответ", type="primary"):
    if not question.strip():
        st.error("Введите вопрос.")
    else:
        result = assistant.answer(question)
        st.markdown("### Ответ ассистента")
        st.write(result['answer'])
        with st.expander("Grounding-контекст"):
            st.json(result['context'])
        st.caption(f"Режим ответа: {result['mode']}")
