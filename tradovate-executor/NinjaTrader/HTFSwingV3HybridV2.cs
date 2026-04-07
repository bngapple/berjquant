/*
 * HTFSwingV3HybridV2.cs — standalone NinjaTrader 8 strategy.
 *
 * Native port of the Python HTF Swing v3 Hybrid v2 engine:
 * - RSI Extremes
 * - IB Breakout
 * - Momentum Bars
 *
 * LucidFlex presets are built in:
 * - 25k => 1 contract / $1,000 monthly halt
 * - 50k => 2 contracts / $2,000 monthly halt
 * - 100k => 2 contracts / $3,000 monthly halt
 * - 150k => 3 contracts / $4,500 monthly halt
 *
 * Default session behavior keeps the intended LucidFlex hours:
 * - no new entries after 16:30
 * - flatten all at 16:45
 *
 * Default concurrency matches the research backtests:
 * - up to one position per strategy
 * - up to 3 concurrent positions total
 *
 * Signals are generated from a completed 15-minute bar and executed on the
 * next bar's first tick using market orders.
 */

#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Globalization;
using System.IO;
using NinjaTrader.Cbi;
using NinjaTrader.NinjaScript;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class HTFSwingV3HybridV2 : Strategy
    {
        private const double DefaultPointValueUsd = 2.0;
        private const int MaxHistoryBars = 500;
        private const string RiskStateFileName = "HTFSwingV3HybridV2.risk";

        private class ClosedBar
        {
            public DateTime Time;
            public double Open;
            public double High;
            public double Low;
            public double Close;
            public long Volume;
        }

        private class IbRange
        {
            public DateTime SessionDate;
            public double High;
            public double Low;

            public IbRange(DateTime sessionDate)
            {
                SessionDate = sessionDate.Date;
                High = double.MinValue;
                Low = double.MaxValue;
            }

            public bool IsValid
            {
                get { return High != double.MinValue && Low != double.MaxValue && High > Low; }
            }

            public double Range
            {
                get { return IsValid ? High - Low : 0.0; }
            }
        }

        private class StrategyPosition
        {
            public string Name;
            public bool IsFlat;
            public MarketPosition Side;
            public double EntryPrice;
            public int Quantity;
            public int BarsHeld;

            public StrategyPosition(string name)
            {
                Name = name;
                Reset();
            }

            public void Enter(MarketPosition side, double entryPrice, int quantity)
            {
                if (quantity <= 0)
                    return;

                if (IsFlat || Side != side || Quantity <= 0)
                {
                    IsFlat = false;
                    Side = side;
                    EntryPrice = entryPrice;
                    Quantity = quantity;
                    BarsHeld = 0;
                    return;
                }

                double weightedCost = (EntryPrice * Quantity) + (entryPrice * quantity);
                Quantity += quantity;
                EntryPrice = weightedCost / Quantity;
            }

            public bool Reduce(int quantity)
            {
                if (IsFlat || quantity <= 0)
                    return true;

                Quantity = Math.Max(0, Quantity - quantity);
                if (Quantity == 0)
                {
                    Reset();
                    return true;
                }

                return false;
            }

            public void Reset()
            {
                IsFlat = true;
                Side = MarketPosition.Flat;
                EntryPrice = 0.0;
                Quantity = 0;
                BarsHeld = 0;
            }
        }

        private class PendingSignal
        {
            public string Strategy;
            public MarketPosition Side;
            public int Quantity;
            public double StopLossPts;
            public double TakeProfitPts;
            public int MaxHoldBars;
            public string Reason;
            public DateTime SignalBarTime;
            public double SignalPrice;
        }

        private readonly List<double> _closes = new List<double>();
        private readonly List<double> _highs = new List<double>();
        private readonly List<double> _lows = new List<double>();
        private readonly List<double> _volumes = new List<double>();
        private readonly List<double> _ibHistory = new List<double>();
        private readonly List<PendingSignal> _pendingSignals = new List<PendingSignal>();
        private readonly Dictionary<string, StrategyPosition> _positions =
            new Dictionary<string, StrategyPosition>(StringComparer.OrdinalIgnoreCase);

        private DateTime _lastSeenOpenBar = DateTime.MinValue;
        private DateTime _lastProcessedClosedBar = DateTime.MinValue;
        private DateTime _currentRiskDate = DateTime.MinValue;
        private int _currentRiskMonth = 0;
        private string _riskStatePath = string.Empty;

        private IbRange _todayIb;
        private bool _ibComplete;
        private bool _ibTradedToday;

        private double _ibPercentileLowValue;
        private double _ibPercentileHighValue;
        private double? _rsiValue;
        private double? _atrValue;
        private double? _emaValue;
        private double? _volumeSmaValue;

        private double _dailyPnl;
        private double _realizedPnl;
        private double _peakRealizedPnl;
        private bool _dailyLimitHit;
        private bool _maxDrawdownHit;
        private bool _tradingHalted;
        private bool _eodFlattened;

        // ----------------------------------------------------------------
        // Session / behavior
        // ----------------------------------------------------------------

        [NinjaScriptProperty]
        [Display(Name = "Session Start (HHMMSS)", GroupName = "Session", Order = 1)]
        public int SessionStart { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "No New Entries After (HHMMSS)", GroupName = "Session", Order = 2)]
        public int NoNewEntriesAfter { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "EOD Flatten Time (HHMMSS)", GroupName = "Session", Order = 3)]
        public int EodFlattenTime { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "One Position At A Time", GroupName = "Behavior", Order = 4)]
        public bool OnePositionAtATime { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Persist Risk State", GroupName = "Behavior", Order = 5)]
        public bool PersistRiskState { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Use LucidFlex Presets", GroupName = "Behavior", Order = 6)]
        public bool UseLucidFlexPresets { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "LucidFlex Account Size (25/50/100/150k)", GroupName = "Behavior", Order = 7)]
        public int LucidFlexAccountSizeK { get; set; }

        [NinjaScriptProperty]
        [Range(1, 3)]
        [Display(Name = "Contracts Per Strategy (manual)", GroupName = "Behavior", Order = 8)]
        public int ContractsPerStrategy { get; set; }

        // ----------------------------------------------------------------
        // Risk
        // ----------------------------------------------------------------

        [NinjaScriptProperty]
        [Display(Name = "Monthly Loss Limit USD (manual)", GroupName = "Risk", Order = 10)]
        public double MonthlyLossLimitUsd { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Daily Loss Limit USD (0 = off)", GroupName = "Risk", Order = 11)]
        public double DailyLossLimitUsd { get; set; }

        // ----------------------------------------------------------------
        // RSI
        // ----------------------------------------------------------------

        [NinjaScriptProperty]
        [Display(Name = "RSI Period", GroupName = "RSI", Order = 20)]
        public int RsiPeriod { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Oversold", GroupName = "RSI", Order = 21)]
        public double RsiOversold { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Overbought", GroupName = "RSI", Order = 22)]
        public double RsiOverbought { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Stop Loss Pts", GroupName = "RSI", Order = 23)]
        public double RsiStopLossPts { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Take Profit Pts", GroupName = "RSI", Order = 24)]
        public double RsiTakeProfitPts { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Max Hold Bars", GroupName = "RSI", Order = 25)]
        public int RsiMaxHoldBars { get; set; }

        // ----------------------------------------------------------------
        // IB Breakout
        // ----------------------------------------------------------------

        [NinjaScriptProperty]
        [Display(Name = "IB Start (HHMMSS)", GroupName = "IB", Order = 30)]
        public int IbStartTime { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "IB End (HHMMSS)", GroupName = "IB", Order = 31)]
        public int IbEndTime { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "IB Last Entry (HHMMSS)", GroupName = "IB", Order = 32)]
        public int IbLastEntryTime { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Stop Loss Pts", GroupName = "IB", Order = 33)]
        public double IbStopLossPts { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Take Profit Pts", GroupName = "IB", Order = 34)]
        public double IbTakeProfitPts { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Max Hold Bars", GroupName = "IB", Order = 35)]
        public int IbMaxHoldBars { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Min IB Range Pts", GroupName = "IB", Order = 36)]
        public double IbMinRangePts { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "IB Range Lookback Days", GroupName = "IB", Order = 37)]
        public int IbRangeLookbackDays { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Range Percentile Low", GroupName = "IB", Order = 38)]
        public double IbRangePercentileLow { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Range Percentile High", GroupName = "IB", Order = 39)]
        public double IbRangePercentileHigh { get; set; }

        // ----------------------------------------------------------------
        // Momentum
        // ----------------------------------------------------------------

        [NinjaScriptProperty]
        [Display(Name = "ATR Period", GroupName = "Momentum", Order = 40)]
        public int MomAtrPeriod { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "EMA Period", GroupName = "Momentum", Order = 41)]
        public int MomEmaPeriod { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Volume SMA Period", GroupName = "Momentum", Order = 42)]
        public int MomVolumeSmaPeriod { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Stop Loss Pts", GroupName = "Momentum", Order = 43)]
        public double MomStopLossPts { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Take Profit Pts", GroupName = "Momentum", Order = 44)]
        public double MomTakeProfitPts { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Max Hold Bars", GroupName = "Momentum", Order = 45)]
        public int MomMaxHoldBars { get; set; }

        // ----------------------------------------------------------------
        // Lifecycle
        // ----------------------------------------------------------------

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description                  = "Native NinjaTrader port of HTF Swing v3 Hybrid v2.";
                Name                         = "HTFSwingV3HybridV2";
                Calculate                    = Calculate.OnEachTick;
                EntriesPerDirection          = 3;
                EntryHandling                = EntryHandling.UniqueEntries;
                IsOverlay                    = true;
                BarsRequiredToTrade          = 0;
                IsExitOnSessionCloseStrategy = false;
                StartBehavior                = StartBehavior.WaitUntilFlat;
                RealtimeErrorHandling        = RealtimeErrorHandling.StopCancelClose;

                SessionStart                 = 93000;
                NoNewEntriesAfter            = 163000;
                EodFlattenTime               = 164500;
                OnePositionAtATime           = false;
                PersistRiskState             = true;
                UseLucidFlexPresets          = true;
                LucidFlexAccountSizeK        = 25;
                ContractsPerStrategy         = 1;

                MonthlyLossLimitUsd          = 1000.0;
                DailyLossLimitUsd            = 0.0;

                RsiPeriod                    = 5;
                RsiOversold                  = 35.0;
                RsiOverbought                = 65.0;
                RsiStopLossPts               = 10.0;
                RsiTakeProfitPts             = 100.0;
                RsiMaxHoldBars               = 5;

                IbStartTime                  = 93000;
                IbEndTime                    = 100000;
                IbLastEntryTime              = 153000;
                IbStopLossPts                = 10.0;
                IbTakeProfitPts              = 120.0;
                IbMaxHoldBars                = 15;
                IbMinRangePts                = 2.0;
                IbRangeLookbackDays          = 50;
                IbRangePercentileLow         = 25.0;
                IbRangePercentileHigh        = 75.0;

                MomAtrPeriod                 = 14;
                MomEmaPeriod                 = 21;
                MomVolumeSmaPeriod           = 20;
                MomStopLossPts               = 15.0;
                MomTakeProfitPts             = 100.0;
                MomMaxHoldBars               = 5;
            }
            else if (State == State.DataLoaded)
            {
                InitializeCollections();
                _riskStatePath = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
                    "NinjaTrader 8",
                    "bin",
                    "Custom",
                    RiskStateFileName
                );
                LoadRiskState();

                if (BarsPeriod != null && (BarsPeriod.BarsPeriodType.ToString() != "Minute" || BarsPeriod.Value != 15))
                    Print("[HTF] WARNING: Attach HTFSwingV3HybridV2 to a 15-minute chart for parity with the Python strategy.");
            }
            else if (State == State.Realtime)
            {
                int effectiveContracts = GetEffectiveContractsPerStrategy();
                double effectiveMonthlyLimit = GetEffectiveMonthlyLossLimitUsd();
                Print(string.Format(
                    "[HTF] Realtime ready on {0}. OnePositionAtATime={1}, contracts={2}, max drawdown=${3:F2}, tier={4}k",
                    Instrument != null ? Instrument.FullName : "chart",
                    OnePositionAtATime,
                    effectiveContracts,
                    effectiveMonthlyLimit,
                    NormalizeLucidFlexAccountSizeK(LucidFlexAccountSizeK)
                ));
                ValidateWarmup();
            }
            else if (State == State.Terminated)
            {
                SaveRiskState();
            }
        }

        protected override void OnBarUpdate()
        {
            if (BarsInProgress != 0 || CurrentBar < 1)
                return;

            if (Time[0] != _lastSeenOpenBar)
            {
                _lastSeenOpenBar = Time[0];
                HandleBarTransition(Time[0]);
            }

            if (State != State.Historical && !_eodFlattened && ToTime(Time[0]) >= EodFlattenTime && HasAnyOpenPosition())
            {
                _eodFlattened = true;
                Print(string.Format("[HTF] EOD flatten triggered at {0:HH:mm:ss}", Time[0]));
                FlattenAllStrategies("EOD");
            }
        }

        protected override void OnExecutionUpdate(
            Execution execution,
            string executionId,
            double price,
            int quantity,
            MarketPosition marketPosition,
            string orderId,
            DateTime time)
        {
            if (State == State.Historical || execution == null || execution.Order == null || quantity <= 0)
                return;

            Order order = execution.Order;
            string orderName = order.Name ?? string.Empty;
            string fromEntrySignal = order.FromEntrySignal ?? string.Empty;

            if (_positions.ContainsKey(orderName) &&
                (order.OrderAction == OrderAction.Buy || order.OrderAction == OrderAction.SellShort))
            {
                StrategyPosition pos = _positions[orderName];
                MarketPosition side = order.OrderAction == OrderAction.Buy
                    ? MarketPosition.Long
                    : MarketPosition.Short;

                pos.Enter(side, price, quantity);
                Print(string.Format(
                    "[HTF] Entry fill: {0} {1} {2} @ {3:F2} | total_qty={4} avg_entry={5:F2}",
                    orderName,
                    side == MarketPosition.Long ? "Long" : "Short",
                    quantity,
                    price,
                    pos.Quantity,
                    pos.EntryPrice
                ));
                return;
            }

            if (string.IsNullOrEmpty(fromEntrySignal) || !_positions.ContainsKey(fromEntrySignal))
                return;

            StrategyPosition strategyPos = _positions[fromEntrySignal];
            if (strategyPos.IsFlat)
                return;

            int exitQuantity = Math.Min(quantity, strategyPos.Quantity);
            if (exitQuantity <= 0)
                return;

            string exitType = ResolveExitType(order);
            double pnl = 0.0;

            if (strategyPos.Side == MarketPosition.Long)
                pnl = (price - strategyPos.EntryPrice) * GetPointValueUsd() * exitQuantity;
            else if (strategyPos.Side == MarketPosition.Short)
                pnl = (strategyPos.EntryPrice - price) * GetPointValueUsd() * exitQuantity;

            Print(string.Format(
                "[HTF] Exit: {0} {1} {2} @ {3:F2} | pnl={4:+0.00;-0.00;0.00}",
                fromEntrySignal,
                exitType,
                exitQuantity,
                price,
                pnl
            ));

            bool flatAfterExit = strategyPos.Reduce(exitQuantity);
            Print(string.Format(
                "[HTF] Position update: {0} remaining_qty={1} flat={2}",
                fromEntrySignal,
                strategyPos.Quantity,
                flatAfterExit
            ));
            RecordTradePnl(pnl, fromEntrySignal);
        }

        // ----------------------------------------------------------------
        // Bar lifecycle
        // ----------------------------------------------------------------

        private void HandleBarTransition(DateTime openBarTime)
        {
            CheckForCalendarResets(openBarTime);

            if (State != State.Historical)
                ExecutePendingSignals(openBarTime);

            ProcessCompletedBarIfNeeded(openBarTime);
        }

        private void ProcessCompletedBarIfNeeded(DateTime openBarTime)
        {
            DateTime closedBarTime = Time[1];
            if (closedBarTime == _lastProcessedClosedBar)
                return;

            _lastProcessedClosedBar = closedBarTime;

            ClosedBar bar = new ClosedBar
            {
                Time = closedBarTime,
                Open = Open[1],
                High = High[1],
                Low = Low[1],
                Close = Close[1],
                Volume = Convert.ToInt64(Volume[1], CultureInfo.InvariantCulture),
            };

            IngestClosedBar(bar);

            if (State == State.Historical)
                return;

            AdvanceHeldBars();
            CheckMaxHoldFlattens();

            if (!CanTradeAt(openBarTime))
                return;

            if (OnePositionAtATime && HasAnyOpenPosition())
                return;

            PendingSignal signal = EvaluateRsi(bar);
            if (signal != null)
            {
                QueueSignal(signal);
                if (OnePositionAtATime)
                    return;
            }

            signal = EvaluateIb(bar);
            if (signal != null)
            {
                QueueSignal(signal);
                if (OnePositionAtATime)
                    return;
            }

            signal = EvaluateMomentum(bar);
            if (signal != null)
                QueueSignal(signal);
        }

        private void IngestClosedBar(ClosedBar bar)
        {
            UpdateIbState(bar);

            _closes.Add(bar.Close);
            _highs.Add(bar.High);
            _lows.Add(bar.Low);
            _volumes.Add(bar.Volume);

            TrimSeries(_closes);
            TrimSeries(_highs);
            TrimSeries(_lows);
            TrimSeries(_volumes);

            UpdateIndicators();
        }

        private void UpdateIbState(ClosedBar bar)
        {
            if (_todayIb == null || _todayIb.SessionDate != bar.Time.Date)
            {
                FinalizeIbDay();
                _todayIb = new IbRange(bar.Time.Date);
                _ibComplete = false;
                _ibTradedToday = false;
            }

            int barTime = ToTime(bar.Time);
            if (barTime >= IbStartTime && barTime < IbEndTime)
            {
                _todayIb.High = Math.Max(_todayIb.High, bar.High);
                _todayIb.Low = Math.Min(_todayIb.Low, bar.Low);
            }

            if (barTime >= IbEndTime && !_ibComplete)
            {
                _ibComplete = true;
                if (_todayIb.IsValid)
                {
                    Print(string.Format(
                        "[HTF] IB complete — H={0:F2} L={1:F2} Range={2:F2}",
                        _todayIb.High,
                        _todayIb.Low,
                        _todayIb.Range
                    ));
                }
            }
        }

        private void FinalizeIbDay()
        {
            if (_todayIb == null || !_todayIb.IsValid)
                return;

            _ibHistory.Add(_todayIb.Range);
            while (_ibHistory.Count > Math.Max(1, IbRangeLookbackDays))
                _ibHistory.RemoveAt(0);

            if (TryGetIbRangePercentiles(out _ibPercentileLowValue, out _ibPercentileHighValue))
            {
                return;
            }

            _ibPercentileLowValue = 0.0;
            _ibPercentileHighValue = 0.0;
        }

        private void ExecutePendingSignals(DateTime openBarTime)
        {
            if (_pendingSignals.Count == 0)
                return;

            if (!CanTradeAt(openBarTime))
            {
                foreach (PendingSignal pending in _pendingSignals)
                {
                    Print(string.Format(
                        "[HTF] SKIPPED: {0} {1} @ {2:HH:mm:ss} because trading is not allowed",
                        pending.Strategy,
                        pending.Side == MarketPosition.Long ? "Buy" : "Sell",
                        openBarTime
                    ));
                }
                _pendingSignals.Clear();
                return;
            }

            if (OnePositionAtATime && HasAnyOpenPosition())
            {
                Print("[HTF] Pending signals cleared because a position is already open.");
                _pendingSignals.Clear();
                return;
            }

            foreach (PendingSignal pending in _pendingSignals)
            {
                int slTicks = PointsToTicks(pending.StopLossPts);
                int tpTicks = PointsToTicks(pending.TakeProfitPts);

                SetStopLoss(pending.Strategy, CalculationMode.Ticks, slTicks, false);
                SetProfitTarget(pending.Strategy, CalculationMode.Ticks, tpTicks);

                Print(string.Format(
                    "[HTF] EXECUTING: {0} {1} {2} @ market | reason={3}",
                    pending.Strategy,
                    pending.Side == MarketPosition.Long ? "Buy" : "Sell",
                    pending.Quantity,
                    pending.Reason
                ));

                if (pending.Side == MarketPosition.Long)
                    EnterLong(pending.Quantity, pending.Strategy);
                else
                    EnterShort(pending.Quantity, pending.Strategy);
            }

            _pendingSignals.Clear();
        }

        // ----------------------------------------------------------------
        // Strategy evaluations
        // ----------------------------------------------------------------

        private PendingSignal EvaluateRsi(ClosedBar bar)
        {
            StrategyPosition pos = _positions["RSI"];
            if (!pos.IsFlat || !_rsiValue.HasValue)
                return null;

            int contracts = GetEffectiveContractsPerStrategy();
            if (_rsiValue.Value < RsiOversold)
            {
                return BuildSignal(
                    "RSI",
                    MarketPosition.Long,
                    contracts,
                    RsiStopLossPts,
                    RsiTakeProfitPts,
                    RsiMaxHoldBars,
                    string.Format(CultureInfo.InvariantCulture, "RSI({0:F1}) < {1:F1}", _rsiValue.Value, RsiOversold),
                    bar
                );
            }

            if (_rsiValue.Value > RsiOverbought)
            {
                return BuildSignal(
                    "RSI",
                    MarketPosition.Short,
                    contracts,
                    RsiStopLossPts,
                    RsiTakeProfitPts,
                    RsiMaxHoldBars,
                    string.Format(CultureInfo.InvariantCulture, "RSI({0:F1}) > {1:F1}", _rsiValue.Value, RsiOverbought),
                    bar
                );
            }

            return null;
        }

        private PendingSignal EvaluateIb(ClosedBar bar)
        {
            StrategyPosition pos = _positions["IB"];
            if (!pos.IsFlat || _ibTradedToday || _todayIb == null || !_ibComplete || !_todayIb.IsValid)
                return null;

            int barTime = ToTime(bar.Time);
            if (barTime < IbEndTime || barTime >= IbLastEntryTime)
                return null;

            if (_todayIb.Range < IbMinRangePts)
                return null;

            if (TryGetIbRangePercentiles(out double percentileLow, out double percentileHigh))
            {
                if (!(_todayIb.Range >= percentileLow && _todayIb.Range <= percentileHigh))
                    return null;
            }

            int contracts = GetEffectiveContractsPerStrategy();
            PendingSignal signal = null;

            if (bar.Close > _todayIb.High)
            {
                signal = BuildSignal(
                    "IB",
                    MarketPosition.Long,
                    contracts,
                    IbStopLossPts,
                    IbTakeProfitPts,
                    IbMaxHoldBars,
                    string.Format(CultureInfo.InvariantCulture, "IB Breakout UP — close {0:F2} > IB high {1:F2}", bar.Close, _todayIb.High),
                    bar
                );
            }
            else if (bar.Close < _todayIb.Low)
            {
                signal = BuildSignal(
                    "IB",
                    MarketPosition.Short,
                    contracts,
                    IbStopLossPts,
                    IbTakeProfitPts,
                    IbMaxHoldBars,
                    string.Format(CultureInfo.InvariantCulture, "IB Breakout DOWN — close {0:F2} < IB low {1:F2}", bar.Close, _todayIb.Low),
                    bar
                );
            }

            if (signal != null)
                _ibTradedToday = true;

            return signal;
        }

        private PendingSignal EvaluateMomentum(ClosedBar bar)
        {
            StrategyPosition pos = _positions["MOM"];
            if (!pos.IsFlat || !_atrValue.HasValue || !_emaValue.HasValue || !_volumeSmaValue.HasValue)
                return null;

            double range = bar.High - bar.Low;
            if (range <= _atrValue.Value)
                return null;

            if (bar.Volume <= _volumeSmaValue.Value)
                return null;

            int contracts = GetEffectiveContractsPerStrategy();
            bool bullish = bar.Close > bar.Open;
            bool bearish = bar.Close < bar.Open;

            if (bullish && bar.Close > _emaValue.Value)
            {
                return BuildSignal(
                    "MOM",
                    MarketPosition.Long,
                    contracts,
                    MomStopLossPts,
                    MomTakeProfitPts,
                    MomMaxHoldBars,
                    string.Format(
                        CultureInfo.InvariantCulture,
                        "MOM bullish — range {0:F2} > ATR {1:F2}, vol {2} > SMA {3:F0}, close {4:F2} > EMA {5:F2}",
                        range,
                        _atrValue.Value,
                        bar.Volume,
                        _volumeSmaValue.Value,
                        bar.Close,
                        _emaValue.Value
                    ),
                    bar
                );
            }

            if (bearish && bar.Close < _emaValue.Value)
            {
                return BuildSignal(
                    "MOM",
                    MarketPosition.Short,
                    contracts,
                    MomStopLossPts,
                    MomTakeProfitPts,
                    MomMaxHoldBars,
                    string.Format(
                        CultureInfo.InvariantCulture,
                        "MOM bearish — range {0:F2} > ATR {1:F2}, vol {2} > SMA {3:F0}, close {4:F2} < EMA {5:F2}",
                        range,
                        _atrValue.Value,
                        bar.Volume,
                        _volumeSmaValue.Value,
                        bar.Close,
                        _emaValue.Value
                    ),
                    bar
                );
            }

            return null;
        }

        private PendingSignal BuildSignal(
            string strategy,
            MarketPosition side,
            int quantity,
            double stopLossPts,
            double takeProfitPts,
            int maxHoldBars,
            string reason,
            ClosedBar bar)
        {
            return new PendingSignal
            {
                Strategy = strategy,
                Side = side,
                Quantity = quantity,
                StopLossPts = stopLossPts,
                TakeProfitPts = takeProfitPts,
                MaxHoldBars = maxHoldBars,
                Reason = reason,
                SignalBarTime = bar.Time,
                SignalPrice = bar.Close,
            };
        }

        private void QueueSignal(PendingSignal signal)
        {
            _pendingSignals.Add(signal);
            Print(string.Format(
                "[HTF] QUEUED: {0} {1} {2} — execute next bar open",
                signal.Strategy,
                signal.Side == MarketPosition.Long ? "Buy" : "Sell",
                signal.Quantity
            ));
        }

        // ----------------------------------------------------------------
        // Position / exit management
        // ----------------------------------------------------------------

        private void AdvanceHeldBars()
        {
            foreach (StrategyPosition pos in _positions.Values)
            {
                if (!pos.IsFlat)
                    pos.BarsHeld += 1;
            }
        }

        private void CheckMaxHoldFlattens()
        {
            foreach (KeyValuePair<string, StrategyPosition> pair in _positions)
            {
                StrategyPosition pos = pair.Value;
                if (pos.IsFlat)
                    continue;

                int maxHoldBars = GetMaxHoldBars(pair.Key);
                if (pos.BarsHeld >= maxHoldBars)
                {
                    Print(string.Format(
                        "[HTF] Max hold reached for {0} ({1} bars) — flattening",
                        pair.Key,
                        pos.BarsHeld
                    ));
                    FlattenStrategy(pair.Key, "MaxHold");
                }
            }
        }

        private void FlattenAllStrategies(string reason)
        {
            foreach (KeyValuePair<string, StrategyPosition> pair in _positions)
            {
                if (!pair.Value.IsFlat)
                    FlattenStrategy(pair.Key, reason);
            }
        }

        private void FlattenStrategy(string strategy, string reason)
        {
            StrategyPosition pos = _positions[strategy];
            if (pos.IsFlat)
                return;

            string exitSignalName = strategy + "_" + reason;
            if (pos.Side == MarketPosition.Long)
                ExitLong(exitSignalName, strategy);
            else if (pos.Side == MarketPosition.Short)
                ExitShort(exitSignalName, strategy);
        }

        private bool HasAnyOpenPosition()
        {
            foreach (StrategyPosition pos in _positions.Values)
            {
                if (!pos.IsFlat)
                    return true;
            }
            return false;
        }

        private int GetMaxHoldBars(string strategy)
        {
            if (strategy.Equals("RSI", StringComparison.OrdinalIgnoreCase))
                return RsiMaxHoldBars;
            if (strategy.Equals("IB", StringComparison.OrdinalIgnoreCase))
                return IbMaxHoldBars;
            return MomMaxHoldBars;
        }

        private string ResolveExitType(Order order)
        {
            if (order == null)
                return "Exit";

            if (order.OrderType == OrderType.StopMarket)
                return "SL";
            if (order.OrderType == OrderType.Limit)
                return "TP";
            if (!string.IsNullOrEmpty(order.Name) && order.Name.EndsWith("_MaxHold", StringComparison.OrdinalIgnoreCase))
                return "MaxHold";
            if (!string.IsNullOrEmpty(order.Name) && order.Name.EndsWith("_EOD", StringComparison.OrdinalIgnoreCase))
                return "EOD";
            if (!string.IsNullOrEmpty(order.Name) && order.Name.EndsWith("_Risk", StringComparison.OrdinalIgnoreCase))
                return "Risk";
            return "Command";
        }

        // ----------------------------------------------------------------
        // Risk management
        // ----------------------------------------------------------------

        private void CheckForCalendarResets(DateTime openBarTime)
        {
            if (State == State.Historical || ToTime(openBarTime) < SessionStart)
                return;

            if (_currentRiskMonth == 0)
                _currentRiskMonth = openBarTime.Month;
            if (_currentRiskDate == DateTime.MinValue)
                _currentRiskDate = openBarTime.Date;

            if (_currentRiskDate.Date != openBarTime.Date)
                ResetDaily(openBarTime.Date);
        }

        private void ResetDaily(DateTime date)
        {
            _currentRiskDate = date.Date;
            _dailyPnl = 0.0;
            _dailyLimitHit = false;
            _eodFlattened = false;
            if (!_maxDrawdownHit)
                _tradingHalted = false;
            SaveRiskState();
            Print(string.Format("[HTF] Daily reset — {0:yyyy-MM-dd}", date));
        }

        private void RecordTradePnl(double pnl, string strategy)
        {
            _dailyPnl += pnl;
            _realizedPnl += pnl;
            if (_realizedPnl > _peakRealizedPnl)
                _peakRealizedPnl = _realizedPnl;

            Print(string.Format(
                "[HTF] Trade P&L: {0:+0.00;-0.00;0.00} ({1}) | daily={2:+0.00;-0.00;0.00} | realized={3:+0.00;-0.00;0.00} | dd={4:+0.00;-0.00;0.00}",
                pnl,
                strategy,
                _dailyPnl,
                _realizedPnl,
                _realizedPnl - _peakRealizedPnl
            ));

            if (DailyLossLimitUsd > 0.0 && !_dailyLimitHit && _dailyPnl <= -Math.Abs(DailyLossLimitUsd))
            {
                _dailyLimitHit = true;
                _tradingHalted = true;
                Print(string.Format("[HTF] DAILY LOSS LIMIT HIT: {0:F2}", _dailyPnl));
                FlattenAllStrategies("Risk");
            }

            double effectiveMonthlyLimit = GetEffectiveMonthlyLossLimitUsd();
            double currentDrawdown = _realizedPnl - _peakRealizedPnl;
            if (!_maxDrawdownHit && currentDrawdown <= -Math.Abs(effectiveMonthlyLimit))
            {
                _maxDrawdownHit = true;
                _tradingHalted = true;
                Print(string.Format(
                    "[HTF] MAX DRAWDOWN HIT: drawdown={0:F2} | realized={1:F2} | peak={2:F2}",
                    currentDrawdown,
                    _realizedPnl,
                    _peakRealizedPnl
                ));
                FlattenAllStrategies("Risk");
            }

            SaveRiskState();
        }

        private bool CanTradeAt(DateTime referenceTime)
        {
            if (_tradingHalted)
                return false;

            int timeValue = ToTime(referenceTime);
            if (timeValue < SessionStart)
                return false;
            if (timeValue >= NoNewEntriesAfter)
                return false;

            return true;
        }

        private void LoadRiskState()
        {
            if (!PersistRiskState)
                return;

            try
            {
                if (!File.Exists(_riskStatePath))
                    return;

                Dictionary<string, string> map = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
                string[] lines = File.ReadAllLines(_riskStatePath);
                foreach (string rawLine in lines)
                {
                    if (string.IsNullOrWhiteSpace(rawLine))
                        continue;
                    int separator = rawLine.IndexOf('=');
                    if (separator <= 0)
                        continue;
                    string key = rawLine.Substring(0, separator).Trim();
                    string value = rawLine.Substring(separator + 1).Trim();
                    map[key] = value;
                }

                DateTime now = DateTime.Now;
                DateTime savedDate;
                bool savedMaxDrawdownHit;
                bool savedDailyLimitHit;

                _currentRiskMonth = now.Month;
                _realizedPnl = GetDouble(map, "realized_pnl");
                _peakRealizedPnl = GetDouble(map, "peak_realized_pnl");
                if (_peakRealizedPnl < _realizedPnl)
                    _peakRealizedPnl = _realizedPnl;
                savedMaxDrawdownHit = GetBool(map, "max_drawdown_hit");
                _maxDrawdownHit = savedMaxDrawdownHit;
                if (_maxDrawdownHit)
                {
                    _tradingHalted = true;
                    Print(string.Format(
                        "[HTF] Max drawdown was hit in a previous session. realized={0:F2} peak={1:F2}",
                        _realizedPnl,
                        _peakRealizedPnl
                    ));
                }

                if (TryGetDate(map, "current_date", out savedDate) && savedDate.Date == now.Date)
                {
                    _currentRiskDate = savedDate.Date;
                    _dailyPnl = GetDouble(map, "daily_pnl");
                    savedDailyLimitHit = GetBool(map, "daily_limit_hit");
                    _dailyLimitHit = savedDailyLimitHit;
                }

                Print(string.Format(
                    "[HTF] Risk state loaded — daily={0:+0.00;-0.00;0.00}, realized={1:+0.00;-0.00;0.00}, peak={2:+0.00;-0.00;0.00}, halted={3}",
                    _dailyPnl,
                    _realizedPnl,
                    _peakRealizedPnl,
                    _tradingHalted
                ));
            }
            catch (Exception ex)
            {
                Print("[HTF] Could not load risk state: " + ex.Message);
            }
        }

        private void SaveRiskState()
        {
            if (!PersistRiskState || string.IsNullOrEmpty(_riskStatePath))
                return;

            try
            {
                string folder = Path.GetDirectoryName(_riskStatePath);
                if (!string.IsNullOrEmpty(folder))
                    Directory.CreateDirectory(folder);

                string[] lines =
                {
                    "realized_pnl=" + _realizedPnl.ToString(CultureInfo.InvariantCulture),
                    "peak_realized_pnl=" + _peakRealizedPnl.ToString(CultureInfo.InvariantCulture),
                    "max_drawdown_hit=" + _maxDrawdownHit.ToString(),
                    "current_month=" + (_currentRiskMonth == 0 ? DateTime.Now.Month : _currentRiskMonth).ToString(CultureInfo.InvariantCulture),
                    "current_date=" + (_currentRiskDate == DateTime.MinValue ? DateTime.Now.Date : _currentRiskDate.Date).ToString("yyyy-MM-dd", CultureInfo.InvariantCulture),
                    "daily_pnl=" + _dailyPnl.ToString(CultureInfo.InvariantCulture),
                    "daily_limit_hit=" + _dailyLimitHit.ToString(),
                };

                File.WriteAllLines(_riskStatePath, lines);
            }
            catch (Exception ex)
            {
                Print("[HTF] Could not save risk state: " + ex.Message);
            }
        }

        private void ValidateWarmup()
        {
            int minBars = Math.Max(RsiPeriod + 1, Math.Max(MomAtrPeriod + 1, Math.Max(MomEmaPeriod, MomVolumeSmaPeriod)));

            if (_closes.Count < minBars)
            {
                Print(string.Format(
                    "[HTF] WARNING: Only {0} completed bar(s) loaded. Indicators need at least {1} completed bars for full warmup.",
                    _closes.Count,
                    minBars
                ));
            }

            if (_ibHistory.Count < IbRangeLookbackDays)
            {
                Print(string.Format(
                    "[HTF] WARNING: Only {0} completed IB day(s) loaded. The IB percentile filter uses a trailing {1}-day window and only activates after more than 20 prior IB days.",
                    _ibHistory.Count,
                    IbRangeLookbackDays
                ));
            }
        }

        private bool TryGetIbRangePercentiles(out double percentileLow, out double percentileHigh)
        {
            percentileLow = 0.0;
            percentileHigh = 0.0;

            if (_ibHistory.Count <= 20)
                return false;

            int windowSize = Math.Min(Math.Max(1, IbRangeLookbackDays), _ibHistory.Count);
            List<double> recentHistory = _ibHistory.GetRange(_ibHistory.Count - windowSize, windowSize);
            percentileLow = Percentile(recentHistory, IbRangePercentileLow);
            percentileHigh = Percentile(recentHistory, IbRangePercentileHigh);
            return true;
        }

        private int GetEffectiveContractsPerStrategy()
        {
            if (!UseLucidFlexPresets)
                return ContractsPerStrategy;

            return GetPresetContractsPerStrategy(NormalizeLucidFlexAccountSizeK(LucidFlexAccountSizeK));
        }

        private double GetEffectiveMonthlyLossLimitUsd()
        {
            if (!UseLucidFlexPresets)
                return MonthlyLossLimitUsd;

            return GetPresetMonthlyLossLimitUsd(NormalizeLucidFlexAccountSizeK(LucidFlexAccountSizeK));
        }

        private static int NormalizeLucidFlexAccountSizeK(int configuredTierK)
        {
            if (configuredTierK <= 25)
                return 25;
            if (configuredTierK <= 50)
                return 50;
            if (configuredTierK <= 100)
                return 100;
            return 150;
        }

        private static int GetPresetContractsPerStrategy(int tierK)
        {
            if (tierK <= 25)
                return 1;
            if (tierK >= 150)
                return 3;
            return 2;
        }

        private static double GetPresetMonthlyLossLimitUsd(int tierK)
        {
            if (tierK <= 25)
                return 1000.0;
            if (tierK <= 50)
                return 2000.0;
            if (tierK <= 100)
                return 3000.0;
            return 4500.0;
        }

        // ----------------------------------------------------------------
        // Indicator math parity with Python implementation
        // ----------------------------------------------------------------

        private void UpdateIndicators()
        {
            _rsiValue = ComputeRsi(_closes, RsiPeriod);
            _atrValue = ComputeAtr(_highs, _lows, _closes, MomAtrPeriod);
            _emaValue = ComputeEma(_closes, MomEmaPeriod);
            _volumeSmaValue = ComputeSma(_volumes, MomVolumeSmaPeriod);
        }

        private static double? ComputeSma(List<double> data, int period)
        {
            if (data.Count < period || period <= 0)
                return null;

            double sum = 0.0;
            for (int i = data.Count - period; i < data.Count; i++)
                sum += data[i];
            return sum / period;
        }

        private static double? ComputeEma(List<double> data, int period)
        {
            if (data.Count < period || period <= 0)
                return null;

            double seed = 0.0;
            for (int i = 0; i < period; i++)
                seed += data[i];
            double ema = seed / period;
            double multiplier = 2.0 / (period + 1.0);

            for (int i = period; i < data.Count; i++)
                ema = (data[i] - ema) * multiplier + ema;

            return ema;
        }

        private static double? ComputeRsi(List<double> closes, int period)
        {
            if (period <= 0 || closes.Count < period + 1)
                return null;

            double avgGain = 0.0;
            double avgLoss = 0.0;

            for (int i = 1; i <= period; i++)
            {
                double change = closes[i] - closes[i - 1];
                if (change > 0)
                    avgGain += change;
                else
                    avgLoss += Math.Abs(change);
            }

            avgGain /= period;
            avgLoss /= period;

            for (int i = period + 1; i < closes.Count; i++)
            {
                double change = closes[i] - closes[i - 1];
                double gain = Math.Max(change, 0.0);
                double loss = Math.Abs(Math.Min(change, 0.0));
                avgGain = ((avgGain * (period - 1)) + gain) / period;
                avgLoss = ((avgLoss * (period - 1)) + loss) / period;
            }

            if (avgLoss == 0.0)
                return 100.0;

            double rs = avgGain / avgLoss;
            return 100.0 - (100.0 / (1.0 + rs));
        }

        private static double? ComputeAtr(List<double> highs, List<double> lows, List<double> closes, int period)
        {
            int count = closes.Count;
            if (period <= 0 || highs.Count != count || lows.Count != count || count < period + 1)
                return null;

            double atr = 0.0;
            for (int i = 1; i <= period; i++)
            {
                double tr = Math.Max(
                    highs[i] - lows[i],
                    Math.Max(
                        Math.Abs(highs[i] - closes[i - 1]),
                        Math.Abs(lows[i] - closes[i - 1])
                    )
                );
                atr += tr;
            }

            atr /= period;

            for (int i = period + 1; i < count; i++)
            {
                double tr = Math.Max(
                    highs[i] - lows[i],
                    Math.Max(
                        Math.Abs(highs[i] - closes[i - 1]),
                        Math.Abs(lows[i] - closes[i - 1])
                    )
                );
                atr = ((atr * (period - 1)) + tr) / period;
            }

            return atr;
        }

        private static double Percentile(List<double> data, double pct)
        {
            if (data == null || data.Count == 0)
                return 0.0;

            List<double> sorted = new List<double>(data);
            sorted.Sort();

            if (sorted.Count == 1)
                return sorted[0];

            double position = (pct / 100.0) * (sorted.Count - 1);
            int lower = (int)Math.Floor(position);
            int upper = (int)Math.Ceiling(position);

            if (lower == upper)
                return sorted[lower];

            double weight = position - lower;
            return sorted[lower] + ((sorted[upper] - sorted[lower]) * weight);
        }

        // ----------------------------------------------------------------
        // Helpers
        // ----------------------------------------------------------------

        private void InitializeCollections()
        {
            _closes.Clear();
            _highs.Clear();
            _lows.Clear();
            _volumes.Clear();
            _ibHistory.Clear();
            _pendingSignals.Clear();
            _positions.Clear();

            _positions["RSI"] = new StrategyPosition("RSI");
            _positions["IB"] = new StrategyPosition("IB");
            _positions["MOM"] = new StrategyPosition("MOM");

            _todayIb = null;
            _ibComplete = false;
            _ibTradedToday = false;
            _ibPercentileLowValue = 0.0;
            _ibPercentileHighValue = 0.0;
            _rsiValue = null;
            _atrValue = null;
            _emaValue = null;
            _volumeSmaValue = null;
            _lastSeenOpenBar = DateTime.MinValue;
            _lastProcessedClosedBar = DateTime.MinValue;
            _currentRiskDate = DateTime.MinValue;
            _currentRiskMonth = 0;
            _dailyPnl = 0.0;
            _realizedPnl = 0.0;
            _peakRealizedPnl = 0.0;
            _dailyLimitHit = false;
            _maxDrawdownHit = false;
            _tradingHalted = false;
            _eodFlattened = false;
        }

        private int PointsToTicks(double points)
        {
            return Math.Max(1, (int)Math.Round(points / TickSize, MidpointRounding.AwayFromZero));
        }

        private double GetPointValueUsd()
        {
            if (Instrument != null && Instrument.MasterInstrument != null && Instrument.MasterInstrument.PointValue > 0)
                return Instrument.MasterInstrument.PointValue;
            return DefaultPointValueUsd;
        }

        private static void TrimSeries(List<double> data)
        {
            while (data.Count > MaxHistoryBars)
                data.RemoveAt(0);
        }

        private static bool TryGetInt(Dictionary<string, string> data, string key, out int value)
        {
            value = 0;
            string raw;
            return data.TryGetValue(key, out raw) &&
                   int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out value);
        }

        private static bool TryGetDate(Dictionary<string, string> data, string key, out DateTime value)
        {
            value = DateTime.MinValue;
            string raw;
            return data.TryGetValue(key, out raw) &&
                   DateTime.TryParseExact(raw, "yyyy-MM-dd", CultureInfo.InvariantCulture, DateTimeStyles.None, out value);
        }

        private static double GetDouble(Dictionary<string, string> data, string key)
        {
            string raw;
            double value;
            if (data.TryGetValue(key, out raw) &&
                double.TryParse(raw, NumberStyles.Float | NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out value))
                return value;
            return 0.0;
        }

        private static bool GetBool(Dictionary<string, string> data, string key)
        {
            string raw;
            bool value;
            if (data.TryGetValue(key, out raw) && bool.TryParse(raw, out value))
                return value;
            return false;
        }
    }
}
