import io
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


@dataclass
class FinancialAssistant:
    csv_path: str
    api_key: str | None = None
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"

    def __post_init__(self):
        self.df = self._load_data(self.csv_path)
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url) if self.api_key and OpenAI else None

    def _load_data(self, path: str) -> pd.DataFrame:
        raw = open(path, "rb").read()
        text = None

        for enc in ["utf-8-sig", "utf-16", "utf-16le", "utf-16be", "cp1251"]:
            try:
                candidate = raw.decode(enc)
                if "year" in candidate and "revenue" in candidate:
                    text = candidate
                    break
            except Exception:
                continue

        if text is None:
            raise ValueError("Не удалось декодировать CSV")

        df = pd.read_csv(io.StringIO(text)).sort_values("year").reset_index(drop=True)
        df["revenue_growth_pct"] = df["revenue"].pct_change() * 100
        df["operating_income"] = df["revenue"] - df["cogs"] - df["operating_expenses"]
        df["operating_margin_pct"] = df["operating_income"] / df["revenue"] * 100
        df["net_margin_pct"] = df["net_income"] / df["revenue"] * 100
        return df

    def _money(self, x: float) -> str:
        return f"${x:,.0f}"

    def _pct(self, x: float) -> str:
        return f"{x:.2f}%"

    def _find_years(self, question: str) -> List[int]:
        return [int(y) for y in re.findall(r"(?<!\d)(20\d{2})(?!\d)", question)]

    def _build_context(self, question: str) -> Dict[str, Any]:
        df = self.df
        best_growth = df.loc[df["revenue_growth_pct"].idxmax()]
        years = self._find_years(question)

        selected_years = []
        for year in years:
            row = df[df["year"] == year]
            if not row.empty:
                selected_years.append(row.iloc[0].round(2).to_dict())

        return {
            "question": question,
            "summary": {
                "period": f"{int(df['year'].min())}-{int(df['year'].max())}",
                "best_revenue_growth_year": int(best_growth["year"]),
                "best_revenue_growth_pct": round(float(best_growth["revenue_growth_pct"]), 2),
                "revenue_start": self._money(df.iloc[0]["revenue"]),
                "revenue_end": self._money(df.iloc[-1]["revenue"]),
                "net_income_start": self._money(df.iloc[0]["net_income"]),
                "net_income_end": self._money(df.iloc[-1]["net_income"]),
            },
            "selected_years": selected_years,
            "last_5_years": df.tail(5).round(2).to_dict(orient="records"),
        }

    def _fallback_answer(self, question: str) -> str:
        q = question.lower()
        df = self.df

        if "сам" in q and "рост" in q and "выруч" in q:
            row = df.loc[df["revenue_growth_pct"].idxmax()]
            prev = df[df["year"] == row["year"] - 1].iloc[0]
            return (
                f"Наибольший рост выручки был в {int(row['year'])} году: "
                f"{self._pct(row['revenue_growth_pct'])} относительно {int(prev['year'])} года. "
                f"Выручка выросла с {self._money(prev['revenue'])} до {self._money(row['revenue'])}."
            )

        if "прибыль" in q or "прибыльност" in q:
            best = df.loc[df["net_margin_pct"].idxmax()]
            return (
                f"В долгосрочном периоде прибыльность росла: чистая прибыль увеличилась "
                f"с {self._money(df.iloc[0]['net_income'])} до {self._money(df.iloc[-1]['net_income'])}, "
                f"а лучшая чистая маржа была в {int(best['year'])} году и составила "
                f"{self._pct(best['net_margin_pct'])}."
            )

        if "операцион" in q and "марж" in q:
            return (
                f"Операционная маржа выросла с {self._pct(df.iloc[0]['operating_margin_pct'])} "
                f"в {int(df.iloc[0]['year'])} году до {self._pct(df.iloc[-1]['operating_margin_pct'])} "
                f"в {int(df.iloc[-1]['year'])} году. Это говорит об улучшении операционной эффективности."
            )

        years = self._find_years(question)
        if len(years) == 2:
            a = df[df["year"] == years[0]].iloc[0]
            b = df[df["year"] == years[1]].iloc[0]
            return (
                f"Сравнение {years[0]} и {years[1]} годов: выручка выросла с "
                f"{self._money(a['revenue'])} до {self._money(b['revenue'])}, "
                f"чистая прибыль — с {self._money(a['net_income'])} до {self._money(b['net_income'])}, "
                f"чистая маржа — с {self._pct(a['net_margin_pct'])} до {self._pct(b['net_margin_pct'])}."
            )

        return "Я могу ответить на вопросы о росте выручки, прибыли, марже и сравнении лет."

    def _llm_answer(self, context: Dict[str, Any]) -> str:
        if self.client is None:
            raise RuntimeError("LLM client is not configured")

        system_prompt = (
            "Ты финансовый AI-ассистент. "
            "Отвечай только на основе переданных данных. "
            "Не придумывай числа. "
            "Кратко объясняй выводы."
        )

        user_prompt = (
            "Используй только этот JSON-контекст и ответь структурированно "
            "на вопрос пользователя:\n\n"
            + json.dumps(context, ensure_ascii=False, indent=2)
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        return resp.choices[0].message.content.strip()

    def answer(self, question: str) -> Dict[str, Any]:
        context = self._build_context(question)

        if self.client is not None:
            try:
                return {"answer": self._llm_answer(context), "mode": "llm", "context": context}
            except Exception:
                pass

        return {"answer": self._fallback_answer(question), "mode": "fallback", "context": context}
