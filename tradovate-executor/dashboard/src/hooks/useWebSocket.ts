import { useEffect, useRef, useState, useCallback } from "react";
import type { Position, PnL, Signal, Trade, EngineStatus } from "../types";

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout>>();
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState<EngineStatus | null>(null);
  const [positions, setPositions] = useState<Record<string, Position | null>>(
    {}
  );
  const [pnl, setPnl] = useState<PnL | null>(null);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);

  const connect = useCallback(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/live`);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);

    ws.onclose = () => {
      setConnected(false);
      reconnectRef.current = setTimeout(connect, 3000);
    };

    ws.onerror = () => {
      ws.close();
    };

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);

      switch (msg.type) {
        case "status":
          setStatus(msg.data as EngineStatus);
          break;
        case "position":
          setPositions((prev) => ({
            ...prev,
            [(msg.data as Position).strategy]: msg.data as Position,
          }));
          break;
        case "pnl":
          setPnl(msg.data as PnL);
          break;
        case "signal":
          setSignals((prev) => [
            ...prev.slice(-49),
            { ...(msg.data as Signal), timestamp: msg.timestamp },
          ]);
          break;
        case "fill":
          setTrades((prev) => [
            ...prev.slice(-49),
            { ...(msg.data as Trade), timestamp: msg.timestamp, action: "entry" as const },
          ]);
          break;
        case "exit":
          setTrades((prev) => [
            ...prev.slice(-49),
            { ...(msg.data as Trade), timestamp: msg.timestamp, action: "exit" as const },
          ]);
          setPositions((prev) => ({
            ...prev,
            [(msg.data as { strategy: string }).strategy]: null,
          }));
          break;
      }
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return { connected, status, positions, pnl, signals, trades };
}
