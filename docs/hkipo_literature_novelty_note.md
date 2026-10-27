# Related-work note: DRL for IPO aftermarket trading

**Search and framing note, 2026-10-27.** This is a focused scoping search, not a systematic review. It is intended to prevent an overbroad novelty claim while drafting the 2021–2026 Hong Kong IPO aftermarket study.

## Directly relevant IPO DRL study

Kim et al. (2026), “DQN-P: A Percentage-Based Action Space for Trading Decisions in Korean IPO Stocks,” proposes a percentage-based DQN action space and modifies Q-network normalization/initialization, with dropout. The KCI abstract describes a case study of four Korean IPO stocks listed on the same date and explicitly says that the goal is not generalization to the entire Korean IPO market. Its reported average return is about 5% higher than the FinRL comparison. This paper is direct precedent for applying RL/DQN to IPO stocks; therefore this project must not claim to be the first application of DRL/RL to IPO trading in general. [KCI record](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003394232).

## Distinction and claim boundary

The defensible potential contribution is narrower and empirical: **a chronological, expanding-window, cross-IPO evaluation of DQN-based aftermarket position management for Hong Kong IPOs**, with later IPO cohorts held out by listing year and the same event set used for the DQN variants and baselines. This differs in market, cohort structure, and evaluation target from the Korean four-IPO same-day case study. It does not by itself establish that no prior Hong Kong IPO DRL study exists.

Hong Kong-focused DRL quantitative-trading work exists outside the IPO setting. For example, a 2021 PolyU thesis studies trend-adaptive Dueling DQN in a generative trading environment, so the paper must not claim to be the first DRL quantitative-trading work in Hong Kong or the first DRL work on Hong Kong-market securities generally. [PolyU thesis record](https://theses.lib.polyu.edu.hk/handle/200/11376).

## Search scope and limitations

Focused searches on 2026-10-27 used combinations of “Hong Kong IPO” / “HK IPO” / “Hong Kong IPO stocks” with “reinforcement learning”, “deep reinforcement learning”, “DQN”, and Chinese/Korean equivalents, and checked the KCI record and Hong Kong university repository search results. The searches surfaced the Korean IPO paper and Hong Kong DRL trading research not specific to IPO aftermarket trading. This is not an exhaustive search of every academic index, dissertation, working-paper archive, conference database, Chinese-language database, or unpublished work. Before submission, run and document a reproducible systematic search across Scopus/Web of Science/Google Scholar, IEEE Xplore/ACM DL, SSRN/arXiv, CNKI, KCI, and university repositories, screening titles/abstracts/full text for Hong Kong IPO aftermarket trading with RL/DRL.

## Recommended manuscript wording

Until the systematic review is complete, use:

> “To our knowledge, this is the first study to evaluate deep reinforcement learning for cross-IPO generalization in Hong Kong IPO aftermarket trading using chronological rolling-origin cohorts.”

This statement should only remain if the documented review finds no earlier Hong Kong IPO-aftermarket DRL study. Avoid “the first to apply DRL to IPO stocks,” “the first DRL study in Hong Kong,” and unqualified “first ever.” If the review is incomplete at submission, use the more conservative wording: “We study DQN-based cross-IPO generalization in Hong Kong IPO aftermarket trading, an evaluation setting that differs from the four-stock same-day Korean IPO case study of Kim et al. (2026).”
