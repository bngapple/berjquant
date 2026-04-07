/*
 * PythonBridge.cs — NinjaTrader 8.1 NinjaScript Strategy
 * TCP server that accepts JSON commands from the Python engine and reports fills back.
 *
 * Protocol (newline-delimited JSON, port 6000 by default):
 *
 *   Python → NinjaTrader:
 *     {"cmd":"ENTRY",    "id":"req-001","strategy":"RSI","side":"Buy","qty":3,"sl_pts":10.0,"tp_pts":100.0}
 *     {"cmd":"FLATTEN",  "id":"req-002","strategy":"RSI"}
 *     {"cmd":"FLATTEN_ALL","id":"req-003"}
 *     {"cmd":"PING"}
 *
 *   NinjaTrader → Python:
 *     {"type":"fill",  "id":"req-001","strategy":"RSI","side":"Buy","qty":3,"fill_price":19500.25,"sl_price":19490.25,"tp_price":19600.25}
 *     {"type":"exit",  "strategy":"RSI","exit_type":"SL","fill_price":19490.25,"qty":3}
 *     {"type":"exit",  "strategy":"RSI","exit_type":"TP","fill_price":19600.25,"qty":3}
 *     {"type":"market","timestamp":"2026-04-02T14:45:00.0000000-04:00","price":19510.25,"volume":7}
 *     {"type":"bar","timestamp":"2026-04-02T14:30:00.0000000-04:00","open":19500.00,"high":19525.00,"low":19495.00,"close":19510.25,"volume":1234}
 *     {"type":"ack",   "id":"req-002"}
 *     {"type":"pong"}
 *     {"type":"error", "id":"req-001","message":"..."}
 *     {"type":"eod_flatten","strategy":"RSI","fill_price":19490.00}
 *
 * Setup:
 *   1. Copy this file to: Documents\NinjaTrader 8\bin\Custom\Strategies\PythonBridge.cs
 *   2. In NinjaTrader: Tools → Edit NinjaScript → Strategy → Compile
 *   3. Add to chart: New Strategy → PythonBridge (set instrument to MNQM6, 15-minute bars)
 *   4. Set TCP Port (default 6000) and EOD Flatten Time (default 164500 = 4:45 PM ET)
 *
 * Requirements: NinjaTrader 8.1 on Windows. No external dependencies — stdlib only.
 *
 * .NET Framework: 4.8
 */

#region Using declarations
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Globalization;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using NinjaTrader.Cbi;
using NinjaTrader.NinjaScript;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    /// <summary>
    /// NinjaScript strategy that acts as a TCP server for the Python HTF Swing engine.
    /// Receives entry/flatten commands and reports fills/exits back via the same connection.
    /// </summary>
    public class PythonBridge : Strategy
    {
        // ----------------------------------------------------------------
        // State
        // ----------------------------------------------------------------

        private TcpListener  _server;
        private TcpClient    _client;
        private StreamWriter _writer;

        private readonly object _writeLock  = new object();
        private readonly object _clientLock = new object();

        // Commands queued by the TCP reader thread, drained in OnBarUpdate
        private readonly ConcurrentQueue<string> _commandQueue = new ConcurrentQueue<string>();

        // Per-strategy params stored when ENTRY command received
        // Key: strategy name ("RSI" / "IB" / "MOM")
        private readonly ConcurrentDictionary<string, EntryParams> _entryParams
            = new ConcurrentDictionary<string, EntryParams>(StringComparer.OrdinalIgnoreCase);

        // Maps strategy name → pending request ID (for fill correlation)
        private readonly ConcurrentDictionary<string, string> _pendingReqIds
            = new ConcurrentDictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        private Thread _listenThread;
        private volatile bool _serverRunning    = false;
        private volatile bool _clientConnected  = false;
        private volatile bool _historyPending   = false;
        private DateTime _lastMarketBarTime     = DateTime.MinValue;
        private long     _lastPublishedVolume   = 0;

        // ----------------------------------------------------------------
        // NinjaScript Parameters (shown in Strategy UI)
        // ----------------------------------------------------------------

        [NinjaScriptProperty]
        [Description("TCP port to listen on (Python connects here)")]
        [Display(Name = "TCP Port", GroupName = "Connection", Order = 1)]
        public int TcpPort { get; set; }

        [NinjaScriptProperty]
        [Description("EOD auto-flatten time in HHMMSS format (e.g. 164500 = 4:45 PM)")]
        [Display(Name = "EOD Flatten Time (HHMMSS)", GroupName = "Session", Order = 2)]
        public int EodFlattenTime { get; set; }

        // ----------------------------------------------------------------
        // Lifecycle
        // ----------------------------------------------------------------

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description                  = "Python TCP Bridge for HTF Swing v3";
                Name                         = "PythonBridge";
                Calculate                    = Calculate.OnEachTick;
                EntriesPerDirection          = 3;
                EntryHandling                = EntryHandling.UniqueEntries;
                IsOverlay                    = true;
                BarsRequiredToTrade          = 0;
                IsExitOnSessionCloseStrategy = false;
                TcpPort                      = 6000;
                EodFlattenTime               = 164500;
            }
            else if (State == State.DataLoaded)
            {
                StartServer();
                if (BarsPeriod != null && (BarsPeriod.BarsPeriodType.ToString() != "Minute" || BarsPeriod.Value != 15))
                    Print("[PB] WARNING: Attach PythonBridge to a 15-minute chart so historical warmup matches the Python strategy timeframe.");
            }
            else if (State == State.Terminated)
            {
                StopServer();
            }
        }

        // ----------------------------------------------------------------
        // Main Update — runs on every tick on the NT strategy thread
        // ----------------------------------------------------------------

        protected override void OnBarUpdate()
        {
            if (BarsInProgress != 0)
                return;

            string cmdJson;
            while (_commandQueue.TryDequeue(out cmdJson))
            {
                try   { ExecuteCommand(cmdJson); }
                catch (Exception ex) { Print("[PB] Command error: " + ex.Message); }
            }

            if (_clientConnected)
            {
                if (_historyPending)
                    SendHistoricalBars();
                PublishMarketData();
            }

            // EOD auto-flatten
            if (ToTime(Time[0]) >= EodFlattenTime && Position.MarketPosition != MarketPosition.Flat)
            {
                Print("[PB] EOD flatten at " + Time[0].ToString("HH:mm:ss"));
                FlattenAll("EOD");
            }
        }

        // ----------------------------------------------------------------
        // Fill Notifications — runs on NT strategy thread
        // ----------------------------------------------------------------

        protected override void OnExecutionUpdate(
            Execution      execution,
            string         executionId,
            double         price,
            int            quantity,
            MarketPosition marketPosition,
            string         orderId,
            DateTime       time)
        {
            if (execution == null || execution.Order == null)
                return;

            var order = execution.Order;

            // ---- Entry fill ------------------------------------------------
            if (order.OrderType == OrderType.Market && _entryParams.ContainsKey(order.Name))
            {
                string strategy = order.Name;
                EntryParams ep;
                if (!_entryParams.TryGetValue(strategy, out ep))
                    return;

                string side    = order.OrderAction == OrderAction.Buy ? "Buy" : "Sell";
                double slPrice = side == "Buy" ? price - ep.SlPts : price + ep.SlPts;
                double tpPrice = side == "Buy" ? price + ep.TpPts : price - ep.TpPts;

                slPrice = RoundToTick(slPrice);
                tpPrice = RoundToTick(tpPrice);

                string reqId;
                _pendingReqIds.TryGetValue(strategy, out reqId);
                string unused;
                _pendingReqIds.TryRemove(strategy, out unused);

                Send(BuildJson(
                    "type",       Jq("fill"),
                    "id",         Jq(reqId ?? ""),
                    "strategy",   Jq(strategy),
                    "side",       Jq(side),
                    "qty",        quantity.ToString(),
                    "fill_price", Jd(price),
                    "sl_price",   Jd(slPrice),
                    "tp_price",   Jd(tpPrice)
                ));

                Print(string.Format("[PB] Fill: {0} {1} {2} @ {3:F2} | SL={4:F2} TP={5:F2}",
                    strategy, side, quantity, price, slPrice, tpPrice));
                return;
            }

            // ---- Bracket exit (SL or TP) -----------------------------------
            string fromSignal = order.FromEntrySignal;
            if (!string.IsNullOrEmpty(fromSignal) && _entryParams.ContainsKey(fromSignal))
            {
                string exitType = null;
                if (order.OrderType == OrderType.StopMarket) exitType = "SL";
                else if (order.OrderType == OrderType.Limit) exitType = "TP";

                if (exitType == null)
                    return;

                Send(BuildJson(
                    "type",       Jq("exit"),
                    "strategy",   Jq(fromSignal),
                    "exit_type",  Jq(exitType),
                    "fill_price", Jd(price),
                    "qty",        quantity.ToString()
                ));

                EntryParams removed;
                _entryParams.TryRemove(fromSignal, out removed);

                Print(string.Format("[PB] Exit ({0}): {1} @ {2:F2}", exitType, fromSignal, price));
            }
        }

        // ----------------------------------------------------------------
        // Command Dispatcher (always called on NT strategy thread)
        // ----------------------------------------------------------------

        private void ExecuteCommand(string json)
        {
            string type  = JsonStr(json, "cmd");
            string reqId = JsonStr(json, "id") ?? "";

            if (type == null)
            {
                Print("[PB] Invalid JSON or missing cmd: " + json);
                return;
            }

            switch (type)
            {
                case "ENTRY":
                    HandleEntry(json, reqId);
                    break;

                case "FLATTEN":
                    HandleFlatten(json, reqId);
                    break;

                case "FLATTEN_ALL":
                    FlattenAll("Command");
                    Send(BuildJson("type", Jq("ack"), "id", Jq(reqId)));
                    break;

                case "PING":
                    Send(BuildJson("type", Jq("pong")));
                    break;

                default:
                    Print("[PB] Unknown command: " + type);
                    Send(BuildJson(
                        "type",    Jq("error"),
                        "id",      Jq(reqId),
                        "message", Jq("Unknown command: " + type)
                    ));
                    break;
            }
        }

        private void HandleEntry(string json, string reqId)
        {
            string strategy = JsonStr(json, "strategy");
            string side     = JsonStr(json, "side");
            int    qty      = JsonInt(json, "qty",    1);
            double slPts    = JsonDbl(json, "sl_pts", 10.0);
            double tpPts    = JsonDbl(json, "tp_pts", 100.0);

            if (string.IsNullOrEmpty(strategy) || string.IsNullOrEmpty(side))
            {
                Send(BuildJson("type", Jq("error"), "id", Jq(reqId), "message", Jq("Missing strategy or side")));
                return;
            }

            _entryParams[strategy]   = new EntryParams { SlPts = slPts, TpPts = tpPts };
            _pendingReqIds[strategy] = reqId;

            int slTicks = PointsToTicks(slPts);
            int tpTicks = PointsToTicks(tpPts);

            SetStopLoss(strategy,     CalculationMode.Ticks, slTicks, false);
            SetProfitTarget(strategy, CalculationMode.Ticks, tpTicks);

            if (side == "Buy")
                EnterLong(qty, strategy);
            else
                EnterShort(qty, strategy);

            Print(string.Format("[PB] Entry queued: {0} {1} {2} | SL={3}pts TP={4}pts (req={5})",
                strategy, side, qty, slPts, tpPts, reqId));
        }

        private void HandleFlatten(string json, string reqId)
        {
            string strategy = JsonStr(json, "strategy");
            if (string.IsNullOrEmpty(strategy))
            {
                Send(BuildJson("type", Jq("error"), "id", Jq(reqId), "message", Jq("Missing strategy")));
                return;
            }

            ExitLong(strategy);
            ExitShort(strategy);

            EntryParams removed;
            _entryParams.TryRemove(strategy, out removed);
            string removedReq;
            _pendingReqIds.TryRemove(strategy, out removedReq);

            Send(BuildJson("type", Jq("ack"), "id", Jq(reqId)));
            Print("[PB] Flatten: " + strategy);
        }

        /// <summary>Flatten all open positions and notify Python for each open strategy.</summary>
        private void FlattenAll(string reason)
        {
            ExitLong();
            ExitShort();

            var strategies = new List<string>(_entryParams.Keys);
            foreach (string strat in strategies)
            {
                Send(BuildJson(
                    "type",       Jq("exit"),
                    "strategy",   Jq(strat),
                    "exit_type",  Jq(reason),
                    "fill_price", Jd(Close[0]),
                    "qty",        "0"
                ));
            }

            _entryParams.Clear();
            _pendingReqIds.Clear();
            Print("[PB] Flatten all (" + reason + ")");
        }

        // ----------------------------------------------------------------
        // TCP Infrastructure
        // ----------------------------------------------------------------

        private void StartServer()
        {
            _serverRunning = true;
            try
            {
                _server = new TcpListener(IPAddress.Any, TcpPort);
                _server.Start();
                Print("[PB] Listening on port " + TcpPort);
            }
            catch (Exception ex)
            {
                Print("[PB] Failed to start server: " + ex.Message);
                return;
            }

            _listenThread = new Thread(AcceptLoop) { IsBackground = true, Name = "PB-Accept" };
            _listenThread.Start();
        }

        private void StopServer()
        {
            _serverRunning   = false;
            _clientConnected = false;

            try { if (_server != null) _server.Stop(); }  catch { }
            try { if (_client != null) _client.Close(); } catch { }

            Print("[PB] Server stopped");
        }

        /// <summary>Background thread — accepts one Python client at a time.</summary>
        private void AcceptLoop()
        {
            while (_serverRunning)
            {
                TcpClient newClient;
                try
                {
                    newClient = _server.AcceptTcpClient();
                }
                catch (SocketException ex)
                {
                    if (!_serverRunning) break;
                    Print("[PB] Accept error: " + ex.Message);
                    Thread.Sleep(1000);
                    continue;
                }
                catch (Exception ex)
                {
                    Print("[PB] Accept error: " + ex.Message);
                    Thread.Sleep(1000);
                    continue;
                }

                lock (_clientLock)
                {
                    try { if (_client != null) _client.Close(); } catch { }
                    _client = newClient;
                    var stream = newClient.GetStream();
                    _writer = new StreamWriter(stream, new UTF8Encoding(false)) { AutoFlush = false };
                    _clientConnected = true;
                    _historyPending = true;
                    _lastMarketBarTime = DateTime.MinValue;
                    _lastPublishedVolume = 0;
                }

                Print("[PB] Python connected from " + ((IPEndPoint)newClient.Client.RemoteEndPoint).Address);

                ReadLoop(newClient);

                lock (_clientLock)
                {
                    _clientConnected = false;
                    _writer = null;
                }

                Print("[PB] Python disconnected");
            }
        }

        /// <summary>Reads newline-delimited JSON from the connected client.</summary>
        private void ReadLoop(TcpClient client)
        {
            var sr = new StreamReader(client.GetStream(), Encoding.UTF8);
            while (_serverRunning)
            {
                string line;
                try
                {
                    line = sr.ReadLine();
                    if (line == null) break;
                }
                catch (IOException)  { break; }
                catch (Exception ex) { Print("[PB] Read error: " + ex.Message); break; }

                line = line.Trim();
                if (line.Length > 0)
                    _commandQueue.Enqueue(line);
            }
        }

        /// <summary>Thread-safe send of a JSON string to the connected Python client.</summary>
        private void Send(string json)
        {
            lock (_writeLock)
            {
                if (_writer == null || !_clientConnected) return;
                try
                {
                    _writer.WriteLine(json);
                    _writer.Flush();
                }
                catch (Exception ex)
                {
                    Print("[PB] Send error: " + ex.Message);
                    _clientConnected = false;
                }
            }
        }

        /// <summary>Send recent completed chart bars once after Python connects.</summary>
        private void SendHistoricalBars()
        {
            if (CurrentBar < 1)
                return;

            int barsToSend = Math.Min(CurrentBar, 100);
            for (int barsAgo = barsToSend; barsAgo >= 1; barsAgo--)
            {
                Send(BuildJson(
                    "type",      Jq("bar"),
                    "timestamp", Jq(Time[barsAgo].ToString("o")),
                    "open",      Jd(Open[barsAgo]),
                    "high",      Jd(High[barsAgo]),
                    "low",       Jd(Low[barsAgo]),
                    "close",     Jd(Close[barsAgo]),
                    "volume",    Convert.ToInt64(Volume[barsAgo]).ToString()
                ));
            }

            _historyPending = false;
            Print("[PB] Sent " + barsToSend + " historical bars to Python");
        }

        /// <summary>Send live market updates using the current chart's price and incremental volume.</summary>
        private void PublishMarketData()
        {
            if (CurrentBar < 0)
                return;

            DateTime barTime = Time[0];
            long currentVolume = Convert.ToInt64(Volume[0]);

            if (_lastMarketBarTime != barTime)
            {
                _lastMarketBarTime = barTime;
                _lastPublishedVolume = 0;
            }

            long volumeDelta = currentVolume - _lastPublishedVolume;
            if (volumeDelta < 0)
                volumeDelta = 0;

            _lastPublishedVolume = currentVolume;

            Send(BuildJson(
                "type",      Jq("market"),
                "timestamp", Jq(barTime.ToString("o")),
                "price",     Jd(Close[0]),
                "volume",    volumeDelta.ToString()
            ));
        }

        // ----------------------------------------------------------------
        // Minimal JSON helpers — no external dependencies
        // ----------------------------------------------------------------

        /// <summary>Extract a string value from a flat JSON object.</summary>
        private static string JsonStr(string json, string key)
        {
            var m = Regex.Match(json, "\"" + Regex.Escape(key) + "\"\\s*:\\s*\"([^\"]*)\"");
            return m.Success ? m.Groups[1].Value : null;
        }

        /// <summary>Extract an integer value from a flat JSON object.</summary>
        private static int JsonInt(string json, string key, int defaultVal)
        {
            var m = Regex.Match(json, "\"" + Regex.Escape(key) + "\"\\s*:\\s*(-?[0-9]+)");
            if (!m.Success) return defaultVal;
            int v;
            return int.TryParse(m.Groups[1].Value, out v) ? v : defaultVal;
        }

        /// <summary>Extract a double value from a flat JSON object.</summary>
        private static double JsonDbl(string json, string key, double defaultVal)
        {
            var m = Regex.Match(json, "\"" + Regex.Escape(key) + "\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]*)?)");
            if (!m.Success) return defaultVal;
            double v;
            return double.TryParse(m.Groups[1].Value, NumberStyles.Float,
                CultureInfo.InvariantCulture, out v) ? v : defaultVal;
        }

        /// <summary>
        /// Build a flat JSON object from alternating key/value pairs.
        /// Values must already be JSON-encoded (use Jq for strings, Jd for doubles).
        /// </summary>
        private static string BuildJson(params string[] kv)
        {
            var sb = new StringBuilder("{");
            for (int i = 0; i + 1 < kv.Length; i += 2)
            {
                if (i > 0) sb.Append(',');
                sb.Append('"').Append(kv[i]).Append("\":").Append(kv[i + 1]);
            }
            return sb.Append('}').ToString();
        }

        /// <summary>Encode a string as a JSON string value (quoted, special chars escaped).</summary>
        private static string Jq(string s)
        {
            if (s == null) return "null";
            return "\"" + s.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";
        }

        /// <summary>Encode a double as a JSON number (2 decimal places, invariant culture).</summary>
        private static string Jd(double v)
        {
            return v.ToString("F2", CultureInfo.InvariantCulture);
        }

        // ----------------------------------------------------------------
        // Other helpers
        // ----------------------------------------------------------------

        private double RoundToTick(double price)
        {
            return Math.Round(price / TickSize) * TickSize;
        }

        private int PointsToTicks(double points)
        {
            return Math.Max(1, (int)Math.Round(points / TickSize));
        }

        private struct EntryParams
        {
            public double SlPts;
            public double TpPts;
        }
    }
}
