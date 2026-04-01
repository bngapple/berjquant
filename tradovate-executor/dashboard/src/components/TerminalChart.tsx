import { useEffect, useRef } from "react";
import {
  createChart,
  ColorType,
  CrosshairMode,
  LineStyle,
  type IChartApi,
  type ISeriesApi,
  type UTCTimestamp,
  type IPriceLine,
} from "lightweight-charts";
import type { Bar, Trade, Position } from "../types";

interface Props {
  bars: Bar[];
  trades: Trade[];
  positions: Record<string, Position | null>;
}

export function TerminalChart({ bars, trades, positions }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const emaRef = useRef<ISeriesApi<"Line"> | null>(null);
  const priceLinesRef = useRef<Record<string, IPriceLine[]>>({});

  // Create chart once on mount
  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#0d0d0d" },
        textColor: "#555",
        fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "rgba(255,255,255,0.03)" },
        horzLines: { color: "rgba(255,255,255,0.03)" },
      },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: {
        borderVisible: false,
        textColor: "#555",
      },
      timeScale: {
        borderVisible: false,
        timeVisible: true,
        secondsVisible: false,
        tickMarkFormatter: (time: number) => {
          const d = new Date(time * 1000);
          const h = d.getUTCHours().toString().padStart(2, "0");
          const m = d.getUTCMinutes().toString().padStart(2, "0");
          return `${h}:${m}`;
        },
      },
      width: containerRef.current.clientWidth,
      height: containerRef.current.clientHeight,
    });

    chartRef.current = chart;

    candleRef.current = chart.addCandlestickSeries({
      upColor: "#00d4aa",
      downColor: "#ef4444",
      borderUpColor: "#00d4aa",
      borderDownColor: "#ef4444",
      wickUpColor: "rgba(0,212,170,0.6)",
      wickDownColor: "rgba(239,68,68,0.6)",
    });

    emaRef.current = chart.addLineSeries({
      color: "rgba(245,158,11,0.7)",
      lineWidth: 1,
      lineStyle: LineStyle.Solid,
      priceLineVisible: false,
      lastValueVisible: true,
      crosshairMarkerVisible: false,
    });

    const obs = new ResizeObserver(() => {
      if (containerRef.current) {
        chart.resize(containerRef.current.clientWidth, containerRef.current.clientHeight);
      }
    });
    obs.observe(containerRef.current);

    return () => {
      obs.disconnect();
      chart.remove();
      chartRef.current = null;
      candleRef.current = null;
      emaRef.current = null;
      priceLinesRef.current = {};
    };
  }, []);

  // Update OHLCV + EMA data
  useEffect(() => {
    if (!candleRef.current || bars.length === 0) return;

    const seen = new Set<number>();
    const candles = bars
      .map((b) => ({
        time: (Math.floor(new Date(b.timestamp).getTime() / 1000)) as UTCTimestamp,
        open: b.open,
        high: b.high,
        low: b.low,
        close: b.close,
      }))
      .filter((b) => { if (seen.has(b.time)) return false; seen.add(b.time); return true; })
      .sort((a, b) => a.time - b.time);

    try { candleRef.current.setData(candles); } catch { /* stale ref */ }

    if (emaRef.current) {
      const emaSeen = new Set<number>();
      const emaData = bars
        .filter((b) => b.ema != null)
        .map((b) => ({
          time: (Math.floor(new Date(b.timestamp).getTime() / 1000)) as UTCTimestamp,
          value: b.ema as number,
        }))
        .filter((b) => { if (emaSeen.has(b.time)) return false; emaSeen.add(b.time); return true; })
        .sort((a, b) => a.time - b.time);
      try { emaRef.current.setData(emaData); } catch { /* stale ref */ }
    }
  }, [bars]);

  // Trade markers — snap to nearest bar timestamp
  useEffect(() => {
    if (!candleRef.current || bars.length === 0) return;

    const barTimes = bars.map((b) => Math.floor(new Date(b.timestamp).getTime() / 1000));

    const snapToBar = (ts: string): UTCTimestamp | null => {
      const sec = Math.floor(new Date(ts).getTime() / 1000);
      let best = -1, bestDiff = Infinity;
      for (const t of barTimes) {
        const d = Math.abs(t - sec);
        if (d < bestDiff) { bestDiff = d; best = t; }
      }
      return bestDiff <= 15 * 60 ? (best as UTCTimestamp) : null;
    };

    const seen = new Set<string>();
    const markers = trades
      .filter((t) => t.timestamp)
      .flatMap((t) => {
        const time = snapToBar(t.timestamp);
        if (!time) return [];
        const key = `${t.strategy}-${t.action}-${t.timestamp}`;
        if (seen.has(key)) return [];
        seen.add(key);
        return [{
          time,
          position: (t.action === "entry"
            ? (t.side === "Buy" ? "belowBar" : "aboveBar")
            : "aboveBar") as "aboveBar" | "belowBar" | "inBar",
          color: t.action === "entry"
            ? (t.side === "Buy" ? "#00d4aa" : "#ef4444")
            : "#888",
          shape: (t.action === "entry"
            ? (t.side === "Buy" ? "arrowUp" : "arrowDown")
            : "circle") as "arrowUp" | "arrowDown" | "circle" | "square",
          text: t.action === "entry"
            ? `${t.strategy} ${t.side === "Buy" ? "▲" : "▼"}`
            : `${t.strategy} ${t.exit_reason ?? "exit"}`,
          size: 1,
        }];
      })
      .sort((a, b) => (a.time as number) - (b.time as number));

    try { candleRef.current.setMarkers(markers); } catch { /* stale ref */ }
  }, [bars, trades]);

  // SL / TP price lines
  useEffect(() => {
    if (!candleRef.current) return;

    // Remove stale price lines
    for (const lines of Object.values(priceLinesRef.current)) {
      for (const pl of lines) {
        try { candleRef.current.removePriceLine(pl); } catch { /* stale */ }
      }
    }
    priceLinesRef.current = {};

    // Add active position lines
    for (const [strategy, pos] of Object.entries(positions)) {
      if (!pos) continue;
      const lines: IPriceLine[] = [];
      if (pos.sl) {
        lines.push(candleRef.current.createPriceLine({
          price: pos.sl,
          color: "rgba(239,68,68,0.7)",
          lineWidth: 1,
          lineStyle: LineStyle.Dashed,
          axisLabelVisible: true,
          title: `${strategy} SL`,
        }));
      }
      if (pos.tp) {
        lines.push(candleRef.current.createPriceLine({
          price: pos.tp,
          color: "rgba(0,212,170,0.7)",
          lineWidth: 1,
          lineStyle: LineStyle.Dashed,
          axisLabelVisible: true,
          title: `${strategy} TP`,
        }));
      }
      if (pos.entry_price) {
        lines.push(candleRef.current.createPriceLine({
          price: pos.entry_price,
          color: "rgba(255,255,255,0.25)",
          lineWidth: 1,
          lineStyle: LineStyle.Dotted,
          axisLabelVisible: true,
          title: `${strategy} entry`,
        }));
      }
      priceLinesRef.current[strategy] = lines;
    }
  }, [positions]);

  return <div ref={containerRef} className="w-full h-full" />;
}
