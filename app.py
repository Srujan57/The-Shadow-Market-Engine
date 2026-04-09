"""Shadow-Market Engine: single-file Streamlit ABM simulator.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# Engine & Configuration Layer
# ============================================================


@dataclass(frozen=True)
class Config:
    """Centralized simulation constants and defaults."""

    APP_TITLE: str = "Shadow-Market"
    DEFAULT_AGENT_COUNT: int = 10_000
    MIN_AGENT_COUNT: int = 1_000
    MAX_AGENT_COUNT: int = 50_000
    DECISION_BATCH_SIZE: int = 1_000
    THREAD_WORKERS: int = 8
    DEFAULT_VOLATILITY: float = 0.12
    DEFAULT_INTEREST_RATE: float = 4.0
    DEFAULT_STIMULUS: float = 50.0
    DEFAULT_PRICE: float = 10.0
    MIN_PRICE: float = 1.0
    MAX_PRICE: float = 250.0
    MAX_TICKS: int = 2_000


CFG = Config()


class SafeExecution:
    """Safety wrapper for simulation logic.

    Catches numeric/logic issues and records them as in-memory events.
    """

    def __init__(self, event_sink: list[str]) -> None:
        self.event_sink = event_sink

    def run(self, fn: Callable, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (OverflowError, FloatingPointError, RecursionError) as exc:
            self.event_sink.append(f"[SAFE-EXEC] Numerical/logic guard triggered: {exc}")
            return None
        except Exception as exc:  # broad fallback to keep UI responsive
            self.event_sink.append(f"[SAFE-EXEC] Unexpected guarded exception: {exc}")
            return None


class PriceElasticityTracker:
    """Tracks transaction-driven synthetic sentiment and elasticity effects."""

    def __init__(self) -> None:
        self.synthetic_sentiment = 1.0

    def update(self, successful_transactions: int, population: int, volatility: float) -> float:
        tx_rate = successful_transactions / max(population, 1)
        # Sentiment increases with transaction momentum, then mean-reverts.
        drift = (tx_rate - 0.5) * 0.08
        noise = np.random.normal(0.0, volatility * 0.02)
        self.synthetic_sentiment = float(np.clip(self.synthetic_sentiment * (1 + drift + noise), 0.5, 1.7))
        return self.synthetic_sentiment


@dataclass
class Agent:
    id: int
    wealth: float
    personality: str
    need_level: float

    def utility_function(self, market_stimulus: float, interest_rate: float, sentiment: float) -> float:
        base_need = self.need_level * 100
        personality_bias = {
            "Aggressive": 15.0,
            "Conservative": -10.0,
            "Neutral": 0.0,
        }[self.personality]
        wealth_signal = np.log1p(max(self.wealth, 0)) * 6
        macro_effect = market_stimulus * 0.4 - interest_rate * 1.8
        return base_need + personality_bias + wealth_signal + macro_effect + sentiment * 7


class ShadowMarketEngine:
    def __init__(self, n_agents: int, volatility: float) -> None:
        self.volatility = volatility
        self.tick = 0
        self.elasticity = PriceElasticityTracker()

        personalities = np.random.choice(
            ["Aggressive", "Conservative", "Neutral"],
            size=n_agents,
            p=[0.34, 0.30, 0.36],
        )
        wealths = np.random.lognormal(mean=3.25, sigma=0.55, size=n_agents)
        needs = np.random.beta(a=2.0, b=2.3, size=n_agents)

        self.agents = [
            Agent(id=i, wealth=float(wealths[i]), personality=str(personalities[i]), need_level=float(needs[i]))
            for i in range(n_agents)
        ]

        self.history: list[dict] = []
        self.events: list[str] = ["[INIT] Shadow-Market initialized."]
        self.active_strategy: dict = {
            "idea": "",
            "segment": "General",
            "quality": 50.0,
            "marketing": 50.0,
            "price_fit": 50.0,
            "competition": 50.0,
            "idea_score": 0.5,
        }

    @staticmethod
    def calculate_purchase_probability(k: float, utility: float, price: float) -> float:
        """Sigmoid decision function: P(buy) = 1 / (1 + e^{-k(U - P)})."""
        x = np.clip(k * (utility - price), -60, 60)
        return float(1.0 / (1.0 + np.exp(-x)))

    def _process_batch(
        self,
        batch: list[Agent],
        price: float,
        interest_rate: float,
        stimulus: float,
        sentiment: float,
        strategy: dict,
    ) -> tuple[int, float, float]:
        tx = 0
        total_happiness = 0.0
        liquidity = 0.0
        demand_boost = 1 + ((strategy["marketing"] - 50) / 200) + ((strategy["quality"] - 50) / 220)
        competition_drag = 1 - ((strategy["competition"] - 50) / 230)
        price_fit_bonus = 1 + ((strategy["price_fit"] - 50) / 240)

        for agent in batch:
            utility = agent.utility_function(stimulus, interest_rate, sentiment)
            utility *= demand_boost * competition_drag * price_fit_bonus

            if strategy["segment"] == "Budget" and agent.personality == "Conservative":
                utility += 8
            elif strategy["segment"] == "Premium" and agent.personality == "Aggressive":
                utility += 8
            elif strategy["segment"] == "Mass Market":
                utility += 4

            # Idea relevance affects utility via agent need fit.
            utility += (strategy["idea_score"] - 0.5) * 20 * (0.7 + agent.need_level)
            k = 0.09 if agent.personality == "Aggressive" else 0.06 if agent.personality == "Neutral" else 0.04
            p_buy = self.calculate_purchase_probability(k=k, utility=utility, price=price)

            if np.random.random() < p_buy and agent.wealth >= price:
                spend = price * np.random.uniform(0.9, 1.1)
                agent.wealth -= spend
                agent.wealth += np.random.uniform(0.0, stimulus * 0.015)  # policy leakage
                tx += 1
                total_happiness += min(1.0, p_buy + 0.1)
            else:
                # Conservative growth from savings + rates
                growth = agent.wealth * ((interest_rate / 100) / 365)
                agent.wealth += growth
                total_happiness += p_buy * 0.4

            # clamp wealth to avoid runaway overflow loops
            agent.wealth = float(np.clip(agent.wealth, 0.0, 1e7))
            liquidity += agent.wealth

        return tx, total_happiness, liquidity

    def step(self, price: float, interest_rate: float, stimulus: float) -> dict | None:
        if self.tick >= CFG.MAX_TICKS:
            self.events.append("[INFO] Max ticks reached; pausing simulation.")
            return None

        self.tick += 1
        sentiment = self.elasticity.synthetic_sentiment

        agent_count = len(self.agents)
        batches = [
            self.agents[i : i + CFG.DECISION_BATCH_SIZE]
            for i in range(0, agent_count, CFG.DECISION_BATCH_SIZE)
        ]

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=CFG.THREAD_WORKERS) as pool:
            futures = [
                pool.submit(self._process_batch, batch, price, interest_rate, stimulus, sentiment, self.active_strategy)
                for batch in batches
            ]

        tx_total = 0
        happiness_total = 0.0
        liquidity_total = 0.0
        for future in futures:
            tx, happiness, liquidity = future.result()
            tx_total += tx
            happiness_total += happiness
            liquidity_total += liquidity

        new_sentiment = self.elasticity.update(
            successful_transactions=tx_total,
            population=agent_count,
            volatility=self.volatility,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        mean_happiness = happiness_total / max(agent_count, 1)

        if tx_total < agent_count * 0.15:
            self.events.append(f"[ALERT t={self.tick}] Market Crash Detected: low transaction velocity ({tx_total}).")
        elif tx_total > agent_count * 0.65:
            self.events.append(f"[TREND t={self.tick}] Viral Trend Emerging: demand surge ({tx_total}).")

        snapshot = {
            "tick": self.tick,
            "transactions": tx_total,
            "synthetic_sentiment": new_sentiment,
            "mean_happiness": mean_happiness,
            "total_liquidity": liquidity_total,
            "mean_wealth": liquidity_total / max(agent_count, 1),
            "p50_wealth": float(np.median([a.wealth for a in self.agents])),
            "p90_wealth": float(np.quantile([a.wealth for a in self.agents], 0.90)),
            "frame_ms": elapsed_ms,
            "target_60fps": elapsed_ms <= 16.7,
            "price": price,
        }
        self.history.append(snapshot)

        return snapshot

    def inject_black_swan(self) -> None:
        shocked = 0
        for agent in self.agents:
            agent.wealth *= 0.5
            shocked += 1
        self.events.append(f"[BLACK SWAN t={self.tick}] Wealth shock applied to {shocked} agents (-50%).")

    def set_user_strategy(
        self,
        idea: str,
        segment: str,
        quality: float,
        marketing: float,
        price_fit: float,
        competition: float,
    ) -> None:
        keywords = {
            "ai": 0.08,
            "automation": 0.06,
            "saas": 0.07,
            "health": 0.05,
            "fintech": 0.06,
            "gaming": 0.05,
            "education": 0.04,
            "eco": 0.05,
            "sustainable": 0.05,
            "marketplace": 0.05,
        }
        lower_idea = idea.lower().strip()
        score = 0.45
        for key, weight in keywords.items():
            if key in lower_idea:
                score += weight
        score += (quality - 50) / 250 + (marketing - 50) / 300 - (competition - 50) / 350
        idea_score = float(np.clip(score, 0.1, 0.95))

        self.active_strategy = {
            "idea": idea.strip(),
            "segment": segment,
            "quality": quality,
            "marketing": marketing,
            "price_fit": price_fit,
            "competition": competition,
            "idea_score": idea_score,
        }
        self.events.append(
            f"[STRATEGY t={self.tick}] Strategy applied: {segment} | quality={quality:.0f}, "
            f"marketing={marketing:.0f}, competition={competition:.0f}, idea_score={idea_score:.2f}"
        )

    def analytics_report(self) -> pd.DataFrame:
        wealth = np.array([a.wealth for a in self.agents], dtype=float)
        n = len(wealth)
        if n == 0:
            return pd.DataFrame()

        sorted_w = np.sort(wealth)
        index = np.arange(1, n + 1)
        gini = float((np.sum((2 * index - n - 1) * sorted_w)) / (n * np.sum(sorted_w) + 1e-9))

        latest_happiness = self.history[-1]["mean_happiness"] if self.history else 0.0
        liquidity = float(np.sum(wealth))

        report = pd.DataFrame(
            {
                "metric": ["Gini Coefficient", "Mean Happiness", "Total Liquidity"],
                "value": [gini, latest_happiness, liquidity],
            }
        )

        describe_df = pd.DataFrame({"wealth": wealth}).describe().T.reset_index().rename(columns={"index": "metric"})
        describe_df["source"] = "pandas.describe"

        return pd.concat([report, describe_df], ignore_index=True)

    def scenario_summary(self) -> dict:
        """Generate plain-language and technical scenario outcome summaries."""
        if not self.history:
            return {
                "status": "not_started",
                "simple": "No simulation data yet. Apply a scenario and run some ticks to see how your idea performs.",
                "technical": (
                    "Insufficient time-series observations: run ticks to produce transaction velocity, "
                    "sentiment, and liquidity trajectories."
                ),
            }

        recent_window = self.history[-20:] if len(self.history) >= 20 else self.history
        recent_df = pd.DataFrame(recent_window)
        base_df = pd.DataFrame(self.history[: min(20, len(self.history))])

        tx_recent = float(recent_df["transactions"].mean())
        tx_base = float(base_df["transactions"].mean())
        sentiment_recent = float(recent_df["synthetic_sentiment"].mean())
        sentiment_base = float(base_df["synthetic_sentiment"].mean())
        happiness_recent = float(recent_df["mean_happiness"].mean())
        liquidity_recent = float(recent_df["total_liquidity"].iloc[-1])
        frame_recent = float(recent_df["frame_ms"].mean())
        idea_score = float(self.active_strategy.get("idea_score", 0.5))

        tx_delta_pct = ((tx_recent - tx_base) / max(tx_base, 1.0)) * 100
        sentiment_delta = sentiment_recent - sentiment_base

        if tx_delta_pct > 8 and sentiment_delta > 0:
            verdict = "positive"
            simple_outcome = "Your scenario looks promising: demand is improving and market mood is strengthening."
            implication = (
                "This suggests your business decision is market-aligned. You can likely scale marketing gradually, "
                "protect margins, and test broader distribution while monitoring conversion quality."
            )
            actions = [
                "Increase rollout budget in controlled stages.",
                "A/B test pricing to improve margin without hurting demand.",
                "Expand to adjacent customer segments.",
            ]
        elif tx_delta_pct < -8 and sentiment_delta < 0:
            verdict = "negative"
            simple_outcome = "Your scenario is struggling: demand and sentiment are both trending down."
            implication = (
                "This indicates your current business decision may not fit market conditions yet. "
                "You likely need to refine product-market fit, reposition pricing, or reduce competitive pressure."
            )
            actions = [
                "Revisit target segment and core value proposition.",
                "Lower price or increase perceived product quality.",
                "Strengthen retention and referral loops before scaling spend.",
            ]
        else:
            verdict = "mixed"
            simple_outcome = "Your scenario is mixed: results are stable but not clearly accelerating."
            implication = (
                "Your business decision appears viable but uncertain. You should run additional experiments before "
                "committing major capital or expansion."
            )
            actions = [
                "Run multiple scenario variants to reduce uncertainty.",
                "Test sharper marketing messages for one segment at a time.",
                "Track sentiment and transaction trends for longer windows.",
            ]

        simple = (
            f"{simple_outcome} In recent ticks, the market processed about {tx_recent:,.0f} purchases per step, "
            f"with average sentiment around {sentiment_recent:.2f}. "
            f"Estimated customer satisfaction (happiness proxy) is {happiness_recent:.2f}."
        )
        technical = (
            f"Scenario diagnostics (recent vs baseline): transaction velocity change {tx_delta_pct:+.1f}%, "
            f"sentiment shift {sentiment_delta:+.3f}, recent mean happiness {happiness_recent:.3f}, "
            f"terminal liquidity {liquidity_recent:,.0f}, mean frame time {frame_recent:.2f}ms. "
            f"Current idea_score={idea_score:.2f}, segment={self.active_strategy.get('segment', 'General')}."
        )

        return {
            "status": "ok",
            "verdict": verdict,
            "simple": simple,
            "technical": technical,
            "implication": implication,
            "actions": actions,
            "tx_delta_pct": tx_delta_pct,
            "sentiment_delta": sentiment_delta,
            "happiness_recent": happiness_recent,
        }


# ============================================================
# Streamlit UI Layer
# ============================================================


st.set_page_config(page_title=CFG.APP_TITLE, layout="wide")
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.1rem;
            font-weight: 750;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: #8EA1B5;
            margin-bottom: 1rem;
        }
        .kpi-card {
            padding: 0.7rem 0.9rem;
            border-radius: 0.8rem;
            background: rgba(45, 73, 110, 0.14);
            border: 1px solid rgba(130, 158, 192, 0.2);
        }
        .feed-shell {
            border-radius: 0.8rem;
            border: 1px solid rgba(130, 158, 192, 0.2);
            padding: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🕶️ Shadow-Market Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Interactive synthetic-economy ABM with safe in-memory execution.</div>',
    unsafe_allow_html=True,
)

if "engine" not in st.session_state:
    st.session_state.engine = ShadowMarketEngine(CFG.DEFAULT_AGENT_COUNT, CFG.DEFAULT_VOLATILITY)
if "safe_exec" not in st.session_state:
    st.session_state.safe_exec = SafeExecution(st.session_state.engine.events)
if "autoplay" not in st.session_state:
    st.session_state.autoplay = False

with st.sidebar:
    st.header("⚙️ Controls")
    interest_rate = st.slider("Initial Interest Rate (%)", 0.0, 20.0, CFG.DEFAULT_INTEREST_RATE, 0.1)
    population = st.slider("Agent Population", CFG.MIN_AGENT_COUNT, CFG.MAX_AGENT_COUNT, CFG.DEFAULT_AGENT_COUNT, 1000)
    stimulus = st.slider("Market Stimulus", 0.0, 100.0, CFG.DEFAULT_STIMULUS, 1.0)
    product_price = st.slider("Product Price", CFG.MIN_PRICE, CFG.MAX_PRICE, CFG.DEFAULT_PRICE, 0.5)
    run_rate = st.select_slider("Auto-Run Speed", options=[1, 2, 5, 10, 20], value=1)

    st.session_state.autoplay = st.toggle("Auto-Run", value=st.session_state.autoplay)

    if st.button("Reset Simulation", use_container_width=True):
        st.session_state.engine = ShadowMarketEngine(population, CFG.DEFAULT_VOLATILITY)
        st.session_state.safe_exec = SafeExecution(st.session_state.engine.events)
        st.success("Simulation reset.")

    if st.button("⚠️ Black Swan Event (-50% Wealth)", use_container_width=True):
        st.session_state.safe_exec.run(st.session_state.engine.inject_black_swan)

engine: ShadowMarketEngine = st.session_state.engine
safe_exec: SafeExecution = st.session_state.safe_exec

if st.session_state.autoplay:
    for _ in range(run_rate):
        safe_exec.run(engine.step, product_price, interest_rate, stimulus)
    time.sleep(0.05)
    st.rerun()

history_df = pd.DataFrame(engine.history)

action_row = st.columns([1, 1, 1, 1.2])
if action_row[0].button("Run 1 Tick", use_container_width=True):
    safe_exec.run(engine.step, product_price, interest_rate, stimulus)
if action_row[1].button("Run 10 Ticks", use_container_width=True):
    for _ in range(10):
        safe_exec.run(engine.step, product_price, interest_rate, stimulus)
if action_row[2].button("Run 100 Ticks", use_container_width=True):
    for _ in range(100):
        safe_exec.run(engine.step, product_price, interest_rate, stimulus)
action_row[3].caption("Tip: use Auto-Run in sidebar for continuous simulation.")

tabs = st.tabs(["📈 Dashboard", "🧪 Live Feed", "📊 Analytics & Export", "💼 Business Scenario", "📘 Instructions"])

with tabs[0]:
    st.subheader("Simulation Status")
    if history_df.empty:
        status_cols = st.columns(4)
        status_cols[0].metric("Tick", engine.tick)
        status_cols[1].metric("Transactions", 0)
        status_cols[2].metric("Sentiment", f"{engine.elasticity.synthetic_sentiment:.3f}")
        status_cols[3].metric("Frame Time (ms)", "—")
        st.info("Run ticks to generate live charts and market telemetry.")
    else:
        latest = history_df.iloc[-1]
        status_cols = st.columns(4)
        status_cols[0].metric("Tick", int(latest["tick"]))
        status_cols[1].metric("Transactions", int(latest["transactions"]))
        status_cols[2].metric("Sentiment", f"{latest['synthetic_sentiment']:.3f}")
        status_cols[3].metric(
            "Frame Time (ms)",
            f"{latest['frame_ms']:.2f}",
            "60fps target met" if latest["target_60fps"] else "Above 16.7ms",
        )

    st.subheader("Market Wealth Distribution (Real-Time)")
    if history_df.empty:
        st.info("No time-series data yet.")
    else:
        chart_df = history_df.set_index("tick")[["mean_wealth", "p50_wealth", "p90_wealth"]]
        st.line_chart(chart_df, height=320)

    st.subheader("Needs vs. Wealth (Agent Scatter)")
    sample_n = min(4000, len(engine.agents))
    sample_agents = random.sample(engine.agents, k=sample_n) if sample_n else []
    scatter_df = pd.DataFrame(
        {
            "needs": [a.need_level for a in sample_agents],
            "wealth": [a.wealth for a in sample_agents],
            "personality": [a.personality for a in sample_agents],
        }
    )
    if scatter_df.empty:
        st.warning("Unable to render scatter plot: no agent sample available.")
    else:
        scatter_fig = px.scatter(
            scatter_df,
            x="needs",
            y="wealth",
            color="personality",
            opacity=0.68,
            title="2D Needs vs Wealth Space",
        )
        scatter_fig.update_layout(margin=dict(l=20, r=20, t=45, b=20))
        st.plotly_chart(scatter_fig, use_container_width=True)

with tabs[1]:
    st.subheader("Event Terminal")
    st.markdown('<div class="feed-shell">', unsafe_allow_html=True)
    live_feed = "\n".join(engine.events[-20:])
    st.text_area("Live Feed", live_feed, height=370)
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Includes critical market alerts, trend signals, and SafeExecution guard events.")

with tabs[2]:
    st.subheader("Post-Sim Report")
    report_df = engine.analytics_report()
    if report_df.empty:
        st.info("Post-simulation metrics will appear after initialization.")
    else:
        st.dataframe(report_df, use_container_width=True, height=320)

    log_df = pd.DataFrame(engine.history)
    if not log_df.empty:
        csv_buffer = io.StringIO()
        log_df.to_csv(csv_buffer, index=False)
        binary_buffer = io.BytesIO(csv_buffer.getvalue().encode("utf-8"))
        st.download_button(
            label="Download Results (CSV)",
            data=binary_buffer,
            file_name="shadow_market_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.caption("Run the simulation to enable CSV export.")

with tabs[3]:
    st.subheader("Interactive Business Scenario Lab")
    st.caption("Enter a business idea and strategy assumptions, then apply and simulate market response.")
    with st.form("idea_form"):
        idea_text = st.text_area(
            "Business Idea / Decision",
            value=engine.active_strategy.get("idea", ""),
            placeholder="Example: AI-powered budgeting app for students with viral referral rewards.",
            height=120,
        )
        form_cols = st.columns(2)
        segment = form_cols[0].selectbox("Target Segment", ["General", "Budget", "Premium", "Mass Market"])
        competition = form_cols[1].slider("Competition Intensity", 0.0, 100.0, 50.0, 1.0)
        quality = st.slider("Product Quality", 0.0, 100.0, 60.0, 1.0)
        marketing = st.slider("Marketing Strength", 0.0, 100.0, 55.0, 1.0)
        price_fit = st.slider("Price-Market Fit", 0.0, 100.0, 50.0, 1.0)
        apply_strategy = st.form_submit_button("Apply Scenario to Agents", use_container_width=True)

    if apply_strategy:
        safe_exec.run(engine.set_user_strategy, idea_text, segment, quality, marketing, price_fit, competition)
        st.success("Scenario applied. Run ticks to watch how agents react.")

    strategy = engine.active_strategy
    st.write("**Active Scenario**")
    st.json(
        {
            "idea": strategy["idea"] or "(none)",
            "target_segment": strategy["segment"],
            "idea_score": round(strategy["idea_score"], 3),
            "quality": strategy["quality"],
            "marketing": strategy["marketing"],
            "price_fit": strategy["price_fit"],
            "competition": strategy["competition"],
        }
    )
    st.info(
        "How agents interact: strategy settings alter utility and demand pressure per tick, changing buy probability, "
        "transaction volume, and market sentiment."
    )

    st.subheader("Scenario Result Summary")
    summary = engine.scenario_summary()
    if summary.get("status") == "not_started":
        st.warning(summary["simple"])
    else:
        verdict = summary.get("verdict", "mixed")
        if verdict == "positive":
            st.success("Overall simulated outlook: Positive")
        elif verdict == "negative":
            st.error("Overall simulated outlook: Negative")
        else:
            st.info("Overall simulated outlook: Mixed / Neutral")

        st.markdown("**Simple explanation**")
        st.write(summary["simple"])
        st.markdown("**What this means for your business/decision**")
        st.write(summary["implication"])
        st.markdown("**Suggested next actions**")
        for idx, action in enumerate(summary["actions"], start=1):
            st.write(f"{idx}. {action}")
        st.markdown("**Technical explanation**")
        st.code(summary["technical"], language="text")

with tabs[4]:
    st.subheader("How to Use Shadow-Market (Beginner Friendly)")
    st.markdown(
        """
        ### 1) Controls (left sidebar)
        - **Initial Interest Rate (%)**: Higher values reward saving and usually reduce spending.
        - **Agent Population**: Number of simulated customers/participants (1,000 to 50,000).
        - **Market Stimulus**: Extra economic boost that increases spending capacity.
        - **Product Price**: Cost of the product agents may buy.
        - **Auto-Run + Speed**: Continuously advances the market each refresh cycle.
        - **Reset Simulation**: Rebuilds a fresh synthetic market.
        - **Black Swan Event**: Applies an immediate 50% wealth drop to test resilience.

        ### 2) Run the simulation
        - Use **Run 1 / 10 / 100 Ticks** to advance time.
        - Each tick is one decision cycle where agents decide whether to buy.

        ### 3) Business Scenario Lab
        - Enter your business idea and assumptions (quality, marketing, competition, price fit).
        - Click **Apply Scenario to Agents**, then run ticks.
        - Compare outcomes with and without your scenario.

        ### 4) Metric explanations
        - **Tick**: Current simulated time step.
        - **Transactions**: Number of successful purchases in the latest step.
        - **Sentiment**: Synthetic market confidence; higher can increase demand momentum.
        - **Frame Time (ms)**: Processing time for one tick; lower is better.
        - **Mean Wealth**: Average wealth across agents.
        - **P50 Wealth**: Median wealth (middle agent).
        - **P90 Wealth**: Wealth at the 90th percentile (top segment threshold).
        - **Gini Coefficient**: Inequality metric from 0 (equal) to 1 (highly unequal).
        - **Mean Happiness**: Simplified satisfaction proxy from agent purchase utility.
        - **Total Liquidity**: Sum of all agent wealth in the system.
        """
    )

# ============================================================
# Analytics & Export Layer
# ============================================================
st.caption("Security note: simulation state is maintained only in memory (session state + in-memory buffers).")