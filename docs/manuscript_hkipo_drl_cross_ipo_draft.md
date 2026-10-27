# Cross-IPO Generalization in Hong Kong IPO Aftermarket Trading: A Rolling Evaluation of Deep Q-Network Policies

**Manuscript draft — 27 October 2026**  
**Project:** *Deep Reinforcement Learning for Quantitative Trading*  
**Status:** Results-led research draft. The 2015–2020 IPO cohort is excluded throughout. The possible “first in Hong Kong IPO aftermarket DRL” novelty wording remains subject to a systematic literature review. The primary analysis is 30 sessions; quality-gated 20/40/60-session runs are included as descriptive horizon sensitivity analyses.

## Abstract

Newly listed stocks pose a difficult sequential trading problem: their histories are short, price movements can be large, and a policy trained on one IPO may not transfer to a later issue. This study evaluates whether a Deep Q-Network (DQN) can learn a long-only position-sizing policy that generalizes across Hong Kong initial public offerings (IPOs) after listing. We construct a chronological, expanding-window evaluation using listings from 2021–2026 and predeclare horizon-specific event-level data-quality rules for 20, 30, 40, and 60 sessions. The quality-gated panels contain 464, 425, 388, and 334 IPOs, respectively; their 2023–2026 test cohorts contain 318, 299, 278, and 240 IPOs. We compare a custom DQN using market and IPO-level features, a Stable-Baselines3 DQN in a matched IPO environment, and buy-and-hold. At a 25-basis-point one-way transaction-cost assumption, the 30-session custom DQN earns a mean net return of 2.89% and has mean maximum drawdown of 14.42%; buy-and-hold earns 10.39% with 24.30% mean maximum drawdown. The paired return difference versus buy-and-hold is −7.49 percentage points (95% event-bootstrap interval: −11.35 to −4.32), while mean drawdown improves by 9.87 points (9.20 to 10.58). The clean horizon-sensitivity runs show the same qualitative trade-off: the custom DQN has lower returns and lower drawdowns than buy-and-hold at 20, 40, and 60 sessions. At 40 and 60 sessions, it has higher mean returns than the matched SB3 DQN in the paired event bootstrap, whereas no return difference is distinguishable at 20 or 30 sessions. These retrospective results do not establish excess-return generation. They are limited by the historical data source, changing horizon-specific samples, execution assumptions, incomplete IPO attributes, and the absence of an untouched prospective cohort.

**Keywords:** deep reinforcement learning; DQN; IPO aftermarket; cross-sectional generalization; Hong Kong equities; rolling-origin evaluation; transaction costs

## 1. Introduction

The first weeks after an initial public offering (IPO) combine two features that are awkward for conventional single-stock forecasting. First, the newly listed security has little or no post-listing history from which to estimate its own dynamics. Second, price discovery continues after listing as investors trade in the secondary market. Hong Kong IPO research has documented the relevance of investor demand and sentiment for IPO pricing and aftermarket performance. For example, Jiang and Li (2013) examine 293 Hong Kong IPOs and report that aftermarket sentiment is related to subsequent price run-up. That evidence motivates studying post-listing decisions, but it does not directly establish whether a sequential trading policy can transfer across distinct IPO events.

Deep reinforcement learning (DRL) provides a framework for learning sequential actions from state transitions and rewards. DQN approximates action values with a neural network and has been applied broadly to decision problems (Mnih et al., 2015). In quantitative finance, libraries such as FinRL provide trading environments, DRL agents, and backtesting workflows (Liu et al., 2020). However, the short histories of IPOs create a particular validation problem: a model may appear effective on a small group of issues while failing to generalize to later offerings.

Recent work directly applies reinforcement learning to IPO stocks. Kim et al. (2026) introduce DQN-P, including a percentage-based action space and changes to the Q-network, and study four Korean IPO stocks listed on the same day. The authors describe their experiment as a case study rather than an attempt to generalize to the entire Korean IPO market. This paper therefore does not claim to be the first application of DRL to IPO trading. It instead asks a narrower question: **Can a DQN position-sizing policy trained on earlier Hong Kong IPOs maintain a measurable return-risk profile when evaluated on IPO cohorts listed in later years?**

We address this question using event-level episodes, IPO-level rolling-origin evaluation, and a rule-based data-quality gate fixed before the clean primary rerun. We train on IPOs from 2021 up to (but excluding) each test year, and then evaluate on the next year. The principal test cohorts are 2023–2026. We compare the custom DQN to a standard Stable-Baselines3 (SB3) DQN trained in the same IPO environment and to a passive buy-and-hold policy. Both learned policies choose among five long-only target weights, incur proportional trading costs, and use identical test events.

The study makes three empirical contributions. First, it evaluates IPO aftermarket DQN policies across later IPO vintages rather than restricting the test to a single same-day cohort. Second, it reports paired event-level return and drawdown comparisons with bootstrap intervals, and tests multiple one-way cost assumptions. Third, it applies and records a predeclared event-level OHLC data rule and reports missing and excluded IPOs against the full listing universe. The results show that the custom DQN reduces mean drawdown relative to buy-and-hold but earns lower net returns. The study therefore treats the learned policy as a risk-return case study, not as a profitable strategy claim.

## 2. Related literature

### 2.1 Hong Kong IPO aftermarket evidence

Prior empirical work treats the IPO aftermarket as a distinct stage of price discovery. Jiang and Li (2013) separate pre-market and aftermarket sentiment in a sample of 293 Hong Kong IPOs and report that aftermarket sentiment is associated with further secondary-market price run-up. Agarwal, Liu, and Rhee (2008), using Hong Kong IPOs from 1993–1997, study the relation between pre-offering demand and subsequent aftermarket performance, reporting different short- and long-run patterns for high- and low-demand offerings. These studies provide market-specific motivation for modelling the post-listing period. The present question differs: rather than explaining IPO pricing or average aftermarket outcomes, we evaluate a sequential policy that changes its exposure as each new IPO evolves.

### 2.2 DRL for trading and IPO stocks

DQN uses a neural network to estimate action values and combines value learning with replay and a target network (Mnih et al., 2015). FinRL organizes common DRL algorithms, simulated market environments, and evaluation baselines for quantitative-finance research (Liu et al., 2020). Such work provides a methodological starting point, but its typical stock or portfolio environments do not address the limited and heterogeneous histories of IPO episodes directly.

Kim et al. (2026) are the closest direct precedent identified in the current scoping search. Their DQN-P method introduces a percentage-based action space and Q-network modifications for IPO stocks. Its empirical case consists of four Korean IPOs listed on one date; the article expressly limits its objective to a case study and does not claim generalization to the whole Korean IPO market. This is relevant both methodologically and as a boundary on novelty claims. The contribution pursued here is the chronological, expanding-window evaluation across Hong Kong IPO cohorts and years. A claim that this is the first such Hong Kong study should remain qualified (“to our knowledge”) and only be retained after systematic database and repository searches.

## 3. Research design

### 3.1 IPO universe and quality-gated sample

The listing universe is drawn from HKEX New Listing Reports for main-board IPOs listed from 2021 through 2026. Daily OHLCV bars are retrieved from Yahoo Finance through `yfinance` and joined to the listing records. The raw panel contains 30,643 observations from 520 IPOs with at least one bar. The full listing master contains 533 events; 13 have no retrieved daily bars.

The raw panel contains 316 rows flagged by the OHLC consistency check across 169 events. The primary horizon is 30 sessions. Before rerunning the main comparison, we declared that an event is eligible only if all of its first 30 post-listing bars pass the check `high >= low`, `high >= max(open, close)`, and `low <= min(open, close)`. We neither repair OHLC values nor impute missing prices. An event with any invalid bar in the first 30 sessions is excluded as a whole, because removing a day would change the episode horizon and could affect both state features and execution prices.

Of the 316 flagged rows, 111 occur in sessions 1–30 and 205 occur in sessions 31–60. Eighty-six complete IPO episodes have one or more inconsistent rows in the primary 30-session window and are excluded. Nine additional events have fewer than 30 sessions at the data cutoff; these are also excluded. The resulting primary panel contains 425 events and 12,750 rows, with 30 rows per event and no OHLC flags. The test folds contain 53, 55, 97, and 94 events in 2023, 2024, 2025, and 2026, respectively, for 299 unique test IPOs.

For the 20-, 40-, and 60-session sensitivity analyses, the same event-level rule is applied separately at each horizon: require a complete consecutive history through session \(h\), and exclude the entire event if any of those \(h\) bars fails the OHLC check. The resulting eligible universe sizes are 464, 388, and 334 IPOs (9,280; 15,520; and 20,040 rows). Their test cohorts contain 318, 278, and 240 IPOs, respectively. No price values are repaired or imputed. Because the event samples differ by horizon, cross-horizon changes are descriptive rather than a paired test of changing only the holding period.

This rule addresses internal OHLC consistency, not the quality of the feed as a whole. Yahoo Finance is not an exchange-licensed data source; turnover is estimated for part of the panel, and the daily-bar simulation does not capture quote-level liquidity. Results should therefore be interpreted as exploratory historical estimates.

### 3.2 IPO-level chronological evaluation

The evaluation uses an expanding-window split. For test year \(y\), the training set contains eligible IPOs listed from 2021 through \(y-1\), while the test set contains eligible IPOs listed during year \(y\). Thus, the folds are:

| Test year | Training years | Training IPOs | Test IPOs |
| ---: | --- | ---: | ---: |
| 2023 | 2021–2022 | 126 | 53 |
| 2024 | 2021–2023 | 179 | 55 |
| 2025 | 2021–2024 | 234 | 97 |
| 2026 | 2021–2025 | 331 | 94 |

No IPO event appears in both training and test within a fold. Because later rolling folds add earlier test years to training, the pooled test performance is a retrospective rolling-origin summary, not a single untouched holdout. The data and test years were examined during earlier exploratory work; this study is not preregistered and does not provide prospective confirmation.

### 3.3 State, action, execution, and reward

Each IPO is simulated as an independent episode with normalized initial account value of one. The agent observes the close of session \(t\), selects a target stock weight, rebalances at the open of session \(t+1\), and marks the account at that session’s close. The first decision is therefore made after the first listing-day close; listing-day returns from the offer price or opening price are not captured by this trading experiment.

The state contains eight market/account variables and three static IPO characteristics:

- Market/account state: price premium to the offer price; one- and five-session close-to-close returns; intraday high-low range relative to close; log-volume change; five-session annualized realized volatility; IPO age in listed sessions; current portfolio weight.
- IPO context: log-scaled offer price; scaled sponsor count; scaled calendar days from prospectus date to listing date.

The features are computed using the event’s own available observations and are clipped/scaled as described in the run metadata. Fundraising amount is omitted because its point-in-time availability was not established; lot size, free float, cornerstone participation, and industry are unavailable or insufficiently complete in the current panel.

The discrete action space is a long-only target weight:

\[
 a_t \in \{0, 0.25, 0.50, 0.75, 1.00\}.
\]

For a trade with notional value \(|\Delta V_t|\), the simulator charges a proportional one-way fee \(c |\Delta V_t|\). The training reward is the change in log net asset value:

\[
 r_{t+1} = \log\left(\frac{NAV_{t+1}}{NAV_t}\right),
\]

where the next NAV reflects the rebalance at the next open, the assumed transaction cost, and mark-to-market at the next close. Maximum drawdown and cumulative turnover are evaluation metrics, not direct penalties in the reward. Buy-and-hold selects a 100% target weight at each decision; in practice this results in the initial purchase followed by little subsequent turnover.

### 3.4 DQN policies and training budget

The custom DQN uses two 64-unit hidden layers with ReLU activations and dropout, Adam learning rate 0.0005, discount factor 0.97, replay capacity 30,000, batch size 64, gradient updates every 32 environment transitions, and target-network synchronization every 1,000 transitions. The epsilon-greedy exploration rate decays from 0.8 toward a floor of 0.05 using a fold-specific reference-transition schedule. The model is trained with a 25 bps one-way cost.

The SB3 baseline uses Stable-Baselines3 DQN with a 64-by-64 MLP in the matched IPO environment. Its learning rate is 0.0001, discount factor 0.99, replay buffer 1,000,000, learning starts after 100 transitions, batch size 32, train frequency 4, and target network interval 10,000 steps. It is a standard-library DQN baseline in this study’s IPO environment, not an unmodified FinRL stock-trading environment and not an exact reproduction of the Korean DQN-P paper’s FinRL settings.

Each model is trained with seeds 0–9. To keep the interaction budget fixed according to the pre-existing 2021-start protocol, each fold’s per-seed budget is six times the number of eligible training IPOs times 29 transitions: 21,924 steps for the 2023 fold, 31,146 for 2024, 40,716 for 2025, and 57,594 for 2026. All evaluation-cost scenarios are applied to the trained policy without retraining. Prices, IPO events, and policy configuration are identical across strategies.

For the 20/40/60-session sensitivity runs, the same 30-session-equivalent budget formula is retained for protocol consistency. Episode lengths are \(h-1\) transitions, so the number of full episode passes and the epsilon-decay trajectory differ by horizon (approximately 9.2 passes for 20 sessions, 4.5 for 40, and 3.0 for 60 under the fixed budget). This is an additional reason to interpret the horizon comparison descriptively.

### 3.5 Costs, metrics, and uncertainty intervals

The prespecified primary evaluation cost is 25 basis points per side. Cost sensitivity is evaluated at 10, 25, and 50 bps per side. We report mean and median net return, win rate, mean maximum drawdown, and one-way turnover. The principal comparison is the custom DQN against buy-and-hold; the SB3 DQN provides a matched algorithm baseline.

For uncertainty intervals, seed-level returns are first averaged within each IPO event. We then resample IPO events 10,000 times. Pooled intervals are stratified within test year to preserve each test cohort’s observed size. Paired policy differences are computed on the same IPO events before resampling. These intervals quantify event-level dispersion under the observed historical cohorts; they do not eliminate shared market-regime dependence or include uncertainty over future market regimes.

## 4. Results

### 4.1 Pooled 30-session performance

Table 2 reports the pooled test results at 25 bps one-way cost. All three policies are evaluated on the same 299 IPOs.

| Policy | Mean net return | 95% event-bootstrap interval | Mean max drawdown | 95% event-bootstrap interval | One-way turnover |
| --- | ---: | ---: | ---: | ---: | ---: |
| Custom DQN + IPO features | 2.89% | [0.55%, 5.40%] | 14.42% | [13.59%, 15.27%] | 6.42× |
| SB3 DQN + IPO features | 3.31% | [0.69%, 6.19%] | 15.46% | [14.56%, 16.37%] | 6.82× |
| Buy-and-hold | 10.39% | [4.91%, 16.37%] | 24.30% | [22.85%, 25.80%] | 1.00× |

The custom DQN’s paired mean-return difference relative to buy-and-hold is −7.49 percentage points (95% CI: −11.35 to −4.32). Its paired drawdown improvement, defined as comparator drawdown minus DQN drawdown, is 9.87 points (95% CI: 9.20 to 10.58). Thus, the lower drawdown is accompanied by substantially lower return in this sample.

Relative to the SB3 DQN, the custom policy’s paired return difference is −0.42 points (95% CI: −1.18 to 0.28), which does not distinguish the two policies’ returns at this interval level. The custom DQN’s drawdown improvement is 1.04 points (95% CI: 0.76 to 1.32). Both learned policies trade more frequently than buy-and-hold, making their returns more sensitive to the cost assumptions.

### 4.2 Results by test cohort

At the primary 25 bps cost, custom-DQN mean returns are lower than buy-and-hold in each test year. The paired return intervals are below zero in all four folds. The DQN’s drawdown is also lower in each fold, although the magnitude varies across years.

| Test cohort | IPOs | Custom DQN return | SB3 DQN return | Buy-and-hold return | Custom DQN max drawdown | Buy-and-hold max drawdown |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2023 | 53 | −0.05% | 1.17% | 5.72% | 10.77% | 22.16% |
| 2024 | 55 | 1.71% | 1.85% | 8.44% | 14.46% | 26.11% |
| 2025 | 97 | 4.98% | 5.03% | 13.02% | 12.02% | 19.38% |
| 2026 | 94 | 3.10% | 3.59% | 11.44% | 18.94% | 29.51% |

The 2025 point estimates are close between the custom DQN and SB3 baseline. The return gap is larger in 2023, 2024, and 2026. The cohort pattern cautions against treating the pooled average as evidence of a stable return-generating mechanism.

### 4.3 Transaction-cost sensitivity

| One-way fee | Custom DQN mean return (95% CI) | Buy-and-hold mean return | Paired DQN minus buy-and-hold return |
| ---: | ---: | ---: | ---: |
| 10 bps | 3.89% [1.58%, 6.37%] | 10.55% | −6.66 pp [−10.50, −3.38] |
| 25 bps | 2.89% [0.55%, 5.40%] | 10.39% | −7.49 pp [−11.35, −4.32] |
| 50 bps | 1.27% [−1.09%, 3.71%] | 10.11% | −8.84 pp [−12.53, −5.53] |

As the assumed one-way cost rises, the custom DQN’s mean return declines and its interval crosses zero at 50 bps. The paired return gap against buy-and-hold remains negative at every cost level. Since the DQN’s average one-way turnover is approximately 6.4× versus 1.0× for buy-and-hold, execution cost is a material determinant of the learned policy’s net performance.

### 4.4 Quality-gated horizon sensitivity

The 20-, 40-, and 60-session models were re-trained on their own complete-case, OHLC-clean panels under the same chronological split, ten seeds, model configurations, and 30-session-equivalent interaction budget. Each horizon has its own eligible event set and test IPO count.

| Horizon | Test IPOs | Custom DQN mean return (95% CI) | Custom DQN mean max drawdown (95% CI) | SB3 DQN mean return | Buy-and-hold mean return | Buy-and-hold mean max drawdown |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 318 | 1.46% [-0.19%, 3.22%] | 11.01% [10.37%, 11.67%] | 1.72% | 5.38% | 21.16% |
| 40 | 278 | 5.59% [2.49%, 8.95%] | 18.16% [17.16%, 19.19%] | 4.30% | 14.03% | 26.86% |
| 60 | 240 | 6.55% [2.27%, 11.32%] | 22.27% [21.05%, 23.50%] | 4.75% | 19.62% | 32.01% |

At all three horizons, the custom DQN’s paired return difference versus buy-and-hold is negative and its drawdown improvement is positive, with 95% paired event-bootstrap intervals excluding zero. The return gaps are −3.92, −8.45, and −13.07 percentage points at 20, 40, and 60 sessions; drawdown improvements are 10.14, 8.71, and 9.74 points. Relative to the SB3 DQN, the custom policy’s paired return difference is not distinguishable from zero at 20 sessions (−0.26 pp, 95% CI [−0.98, 0.46]), but is positive at 40 (1.29 pp [0.40, 2.21]) and 60 sessions (1.80 pp [0.67, 2.98]). Its drawdown differences versus SB3 are not distinguishable from zero at 40 or 60 sessions.

Custom-DQN mean returns at 10/25/50 bps are 2.11% / 1.46% / 0.38% for 20 sessions, 6.86% / 5.59% / 3.48% for 40 sessions, and 8.64% / 6.55% / 3.17% for 60 sessions. The return advantage of the longer horizons must not be read as a pure horizon effect: samples change, more sessions increase exposure to later price movements, and the fixed transition budget produces different numbers of passes and exploration decay by horizon. The complete cost intervals, paired intervals, annual results, and coverage reports are provided in the [companion sensitivity report](hkipo_horizon_sensitivity_results_clean.md).

## 5. Discussion

The main finding is a return-risk trade-off. Under the current environment and data, DQN learned a policy that was less exposed than buy-and-hold and consequently experienced smaller average drawdowns. That protection did not translate into higher net returns. The result is consistent with the policy reducing exposure during some adverse price paths, but this study does not identify a causal mechanism or establish that the learned exposure rule is robust to new regimes.

The experiment’s main design distinction from the DQN-P Korean IPO case study is the unit and chronology of evaluation. Rather than evaluate four IPOs listed on one day, we train on earlier IPO events and test on later calendar-year cohorts. The 20/30/40/60-session sensitivity runs contain 318/299/278/240 test IPOs across four successive vintages and use paired event-level comparisons. This tests a form of across-IPO temporal transfer, although the repeated inspection of historical years means it remains retrospective. The appropriate research claim is therefore about the evaluation design and observed trade-off, not a claim of universal generalization or first application of DRL to IPOs.

The strategy is also not a deployable execution system. It trades at daily open prices with proportional costs and fractional positions. Actual Hong Kong trading involves board lots, bid-ask spreads, auction price formation, liquidity limits, market impact, and security-specific trading frictions. The current data do not model these in sufficient detail. In addition, individual IPOs can share the same market regime; event-level bootstrap intervals may understate uncertainty if outcomes are correlated within calendar periods.

Sample selection deserves attention. Thirteen IPOs have no daily bars, nine have incomplete 30-session histories at the data cutoff, and 86 otherwise complete events are removed because the first 30 OHLC bars contain a consistency violation. The exclusion rule is transparent and independent of model returns, but the remaining 425 events may not represent every listed IPO equally. The 2026 cohort is year-to-date and should not be interpreted as a complete annual regime.

The feature set is intentionally limited to variables available in the present merged data. IPO offer price, sponsor count, and prospectus-to-listing lead time are included; cornerstone allocation, free float, industry, and fundraising amount with verified point-in-time availability are not. This constrains the information available to the agent and means the study does not isolate which IPO attributes, if any, drive its policy.

## 6. Conclusion

This study evaluates DQN-based position sizing in the first 30 post-listing sessions of Hong Kong IPOs using a quality-gated panel and chronological expanding-window evaluation. Across 299 test IPOs, the custom DQN produces lower average drawdowns than buy-and-hold and the SB3 DQN baseline, but it does not outperform either comparator in mean return. Its return is particularly sensitive to transaction costs and its turnover is materially higher than passive holding.

The results support a limited conclusion: in this retrospective experiment, a DQN policy can learn a lower-exposure, lower-drawdown pattern across later IPO cohorts, but the pattern does not improve net return relative to buy-and-hold under the tested assumptions at any evaluated horizon. The custom DQN’s return is higher than the matched SB3 DQN at 40 and 60 sessions, but those relative differences do not overcome the gap to buy-and-hold and should be interpreted in light of horizon-dependent samples and training-pass differences. A prospective IPO cohort evaluated after the model and protocol are frozen, more reliable market data, execution-aware costs, and an expanded point-in-time IPO feature set are needed before making claims about live-market value.

## References

Jiang, L., & Li, G. (2013). Investor sentiment and IPO pricing during pre-market and aftermarket periods: Evidence from Hong Kong. *Pacific-Basin Finance Journal, 21*(1), 65–82. https://doi.org/10.1016/j.pacfin.2013.01.006

Kim, S., Song, Y., Ji, A., Lee, E., & Kim, S. (2026). DQN-P: A Percentage-Based Action Space for Trading Decisions in Korean IPO Stocks. *Journal of KIISE, 53*(10), 961–970. https://doi.org/10.5626/JOK.2026.53.10.961

Liu, X.-Y., Yang, H., Chen, Q., Zhang, R., Yang, L., Xiao, B., & Wang, C. (2020). *FinRL: A deep reinforcement learning library for automated stock trading in quantitative finance*. arXiv:2011.09607. https://arxiv.org/abs/2011.09607

Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. *Nature, 518*, 529–533. https://doi.org/10.1038/nature14236

Agarwal, S., Liu, C., & Rhee, S. G. (2008). Investor demand for IPOs and aftermarket performance: Evidence from the Hong Kong stock market. *Journal of International Financial Markets, Institutions and Money, 18*(2), 176–190. https://doi.org/10.1016/j.intfin.2006.09.001

## Editorial items before submission

1. Complete a documented systematic literature search (English and Chinese-language sources, academic indexes, conference proceedings, dissertations, and working-paper repositories) before retaining any “first in Hong Kong IPO” claim. Current wording is deliberately qualified and scoped.
2. Keep 30 sessions as the primary horizon and describe 20/40/60 sessions as quality-gated sensitivities. Report their distinct event samples and the fixed-budget/pass-count limitation; do not present the horizon differences as a causal effect.
3. Verify all bibliography entries and add missing citations for IPO aftermarket microstructure, chronological validation, and bootstrap inference.
4. Consider a prospective later cohort only after freezing the policy, costs, data gate, metrics, and analysis plan. Historical 2023–2026 results are retrospective.
5. For a journal submission, replace the exploratory Yahoo feed with a documented reliable source if available, or retain the current source with a transparent statement of its limitations.
