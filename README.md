MC Prosperity Trading Challenge (Solo)
This repository documents my participation in the IMC Prosperity Trading Competition, a five-round algorithmic trading simulation focused on market making, arbitrage, options pricing, and predictive modeling. Competing solo, I developed and refined nine distinct, fully-automated strategies, ultimately placing in the Top 10 in Canada out of over 19,000 teams globally.
This document serves as a retrospective on my strategic evolution, detailing how each algorithm was conceived, tested, and adapted across the rounds to achieve its final, profitable state.
🌀 Round 1: Core Market Making and Baseline Strategies
Round 1 served as a crucial proving ground for developing foundational, low-risk algorithms that would become the bedrock of my portfolio's PnL.
🌳 Rainforest Resin: The Anchor
Approach: Simple market making around a clear, fixed fair value of 10,000.
Implementation:
Passive Quoting: I placed limit buy orders at 9994 and sell orders at 10006, creating a consistent 12-point spread that balanced execution frequency with per-trade profit.
Inventory Skewing: To manage risk, quote sizes were dynamically adjusted based on current holdings—reducing buy sizes when long and sell sizes when short.
Aggressive Taking: The algorithm would instantly hit/lift any orders that appeared mispriced relative to 10,000.
Outcome: This strategy was profitable and exceptionally stable from day one. Its robust, non-predictive logic required no modifications and ran unchanged through all five rounds.
🌿 Kelp: Top-of-Book Liquidity
Approach: Market making with full order book visibility, exploiting the ability to see all bids and asks before placing an order.
Implementation:
Joining the Spread: Instead of calculating a mid-price, the strategy dynamically matched the current best bid and best ask, placing its own orders at the very top of the book.
Risk Management: It used a similar inventory skewing logic to the ResinStrategy, slightly shifting its quote prices away from the market when its position grew too large.
Outcome: A consistent, low-risk profit generator. This simple yet effective algorithm also ran untouched for the entire competition, providing a reliable stream of income.
🦑 Squid Ink: Evolving from Models to Signals
Approach: A reactive volatility strategy that adapted significantly over the competition.
Implementation & Evolution:
Early Rounds: I initially deployed a mean-reversion model using moving averages to capitalize on the asset's sharp price swings. This approach was only marginally profitable due to the tight spreads and high noise.
Round 5 (Final Version): With the introduction of trader-level data, I discovered that a trader named "Olivia" consistently traded just before significant price moves. I completely rewrote the strategy into a behavioral "follower" model, abandoning the moving averages entirely. The new logic was simple: buy immediately after Olivia buys, and sell immediately after she sells.
Outcome: The shift from a quantitative model to a behavioral signal in Round 5 transformed SQUID_INK from a negligible performer into a highly profitable asset.
🌀 Round 2: Synthetic Baskets and Statistical Arbitrage
Round 2 introduced a basket of correlated consumer goods, creating opportunities for more complex, multi-asset arbitrage strategies.
🧺 The Picnic Baskets (PB1, PB2)
Approach: A classic pairs trading strategy based on the stable, linear relationship between PICNIC_BASKET1 and PICNIC_BASKET2.
Implementation:
Spread Definition: I defined a synthetic spread Spread = PB1_price - (2.0 * PB2_price - 1869.0) which was historically mean-reverting.
Mean-Reversion Trading: The BasketStrategy would trade this spread when it deviated significantly from its long-term moving average, shorting the spread when high and longing it when low.
Enhanced Liquidity: A secondary MarketMakePB2Strategy ran in parallel, passively quoting PICNIC_BASKET2 to capture spread profit while the main pairs strategy awaited a signal.
Post-Competition Insight: My initial model for the baskets was based on their component prices. While stable, a more advanced regression-based model using components + ETF premium + random noise could have improved signal extraction and risk-adjusted returns by better isolating the true mispricing from the noise.
🍓 The Trios (JAMS, DJEMBES, CROISSANTS)
Approach: A statistical arbitrage strategy that evolved over the rounds.
Implementation & Evolution:
Initial Model: I started by modeling all three assets together. However, analysis showed that CROISSANTS behaved differently, adding noise to the arbitrage signal.
Final Model: I removed CROISSANTS from the basket and focused the TriosStrategy on a highly effective two-asset arbitrage between JAMS and DJEMBES. This refined model normalized their prices to identify statistical mispricings with greater accuracy.
Reassigning Croissants: CROISSANTS was later given its own "Olivia Follower" strategy in Round 5, similar to SQUID_INK, after her predictive trading patterns were identified in that asset as well.
🌀 Round 3: Options Trading and Delta Management
Round 3 introduced VOLCANIC_ROCK and its associated options, requiring a deep dive into derivatives pricing and risk management.
Approach: Trading mispricings identified by a fitted volatility curve, while actively managing directional risk.
Implementation & Evolution:
Volatility Modeling: I plotted the implied volatility smile across all option strikes and fitted a polynomial curve to it. This curve generated the model's "fair price" for each option using the Black-Scholes formula.
Strategy Pivot: My initial attempts at pure IV scalping proved less effective than a more direct approach. The final strategy pivoted to directly trading the difference between the model's fair price and the market price.
Delta Hedging: I implemented a delta hedging module to neutralize the portfolio's directional exposure by taking offsetting positions in the underlying VOLCANIC_ROCK.
Commentary: Due to position limits, the strategy was sometimes forced to carry unintended directional risk (delta). This exposure turned out to be profitable, though it was a result of luck rather than intention. I submitted this round's code at 6:58 AM after an all-nighter; I was thankful just to have a sound, positive-PnL strategy locked in despite the exhaustion.
🌀 Round 4: Macarons, Sunlight, and Cross-Market Arbitrage
Round 4 introduced MAGNIFICENT_MACARONS and a second exchange, complete with tariffs and transport fees, creating unique arbitrage and event-driven opportunities.
Approach: A hybrid strategy combining a primary, signal-based model with a secondary, risk-free arbitrage loop.
Implementation:
Signal Trading (Primary): The core of the strategy was an event-driven state machine tied to the sunlightIndex. This index was a powerful leading indicator: a sharp drop consistently preceded a price spike in macarons. The strategy would automatically go long on this signal and, under specific conditions, flip short to capture the subsequent crash.
Cross-Market Arbitrage (Secondary): I identified a persistent inefficiency where a market participant on the foreign exchange consistently sold Macarons at a price lower than what I could sell them for on the main exchange, even after accounting for all fees. My algorithm executed this near risk-free arbitrage loop continuously, providing a stable profit stream while the main logic awaited a sunlight signal.
🌀 Round 5: Trader Signal Exploitation
The final round provided the most valuable alpha source of the competition: trader-level public trade data.
Approach: A simple but powerful behavioral strategy based on signal extraction.
Implementation:
Identifying "Olivia": Analysis of the provided data quickly revealed that trader "Olivia" was an informed participant whose trades consistently preceded price movements in SQUID_INK and CROISSANTS.
Follower Logic: I built a simple but highly effective follower algorithm: buy immediately after Olivia buys, and sell immediately after she sells.
Commentary: Despite being too sleep-deprived from previous rounds to build a more complex statistical model around the trader data, this clear, high-conviction signal provided by Olivia's trades was more than enough to generate strong PnL and secure a top leaderboard position.