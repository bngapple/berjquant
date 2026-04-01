import { useEffect, useRef, useState, useCallback } from "react";
import type { Position, PnL, Signal, Trade, EngineStatus, EquityPoint, WSData, Bar } from "../types";

export function useWebSocket(): WSData {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout>>(undefined);
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState<EngineStatus | null>(null);
  const [positions, setPositions] = useState<Record<string, Position | null>>({});
  const [pnl, setPnl] = useState<PnL | null>(null);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [equityHistory, setEquityHistory] = useState<EquityPoint[]>([]);
  const [bars, setBars] = useState<Bar[]>([]);

  const connect = useCallback(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/live`);
    wsRef.current = ws;
    ws.onopen = () => setConnected(true);
    ws.onclose = () => { setConnected(false); reconnectRef.current = setTimeout(connect, 3000); };
    ws.onerror = () => { ws.close(); };
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      switch (msg.type) {
        case "status": setStatus(msg.data as EngineStatus); break;
        case "position": setPositions(p => ({ ...p, [(msg.data as Position).strategy]: msg.data as Position })); break;
        case "pnl": {
          const d = msg.data as PnL;
          setPnl(d);
          setEquityHistory(prev => [...prev.slice(-200), { time: new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false }), value: d.daily }]);
          break;
        }
        case "signal": setSignals(p => [...p.slice(-49), { ...(msg.data as Signal), timestamp: msg.timestamp }]); break;
        case "fill": setTrades(p => [...p.slice(-49), { ...(msg.data as Trade), timestamp: msg.timestamp, action: "entry" as const }]); break;
        case "exit":
          setTrades(p => [...p.slice(-49), { ...(msg.data as Trade), timestamp: msg.timestamp, action: "exit" as const }]);
          setPositions(p => ({ ...p, [(msg.data as { strategy: string }).strategy]: null }));
          break;
        case "bar":
          setBars(p => {
            const b = msg.data as Bar;
            // Replace existing bar with same timestamp or append
            const idx = p.findIndex(x => x.timestamp === b.timestamp);
            if (idx >= 0) { const next = [...p]; next[idx] = b; return next; }
            return [...p.slice(-199), b];
          });
          break;
      }
    };
  }, []);

  useEffect(() => { connect(); return () => { clearTimeout(reconnectRef.current); wsRef.current?.close(); }; }, [connect]);

  return { connected, status, positions, pnl, signals, trades, equityHistory, bars };
}
