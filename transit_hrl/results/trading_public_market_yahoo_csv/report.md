# Public Market Data Evaluation

- source: `csv`
- symbols: ['SPY', 'QQQ', 'IWM']
- frequency encoder: `ema`
- predictor: previous-bar log return only, so current/future returns are not used as policy input
- bars: 1500
- total return: 0.3957
- annualized Sharpe: 0.406
- max drawdown: 0.2790
- turnover: 106.85
- promotions: 246

This is a public-data validation path for the Freq-HRL protocol. It is not investment advice and is not a production trading simulator.
