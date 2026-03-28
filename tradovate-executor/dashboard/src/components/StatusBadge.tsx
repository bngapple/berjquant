export function StatusBadge({
  running,
  connected,
}: {
  running: boolean;
  connected: boolean;
}) {
  if (!connected) {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-gray-500">
        <span className="w-2 h-2 rounded-full bg-gray-600" />
        Disconnected
      </span>
    );
  }
  return (
    <span
      className={`inline-flex items-center gap-1.5 text-xs ${running ? "text-green-400" : "text-gray-400"}`}
    >
      <span
        className={`w-2 h-2 rounded-full ${running ? "bg-green-500 animate-pulse" : "bg-gray-600"}`}
      />
      {running ? "Running" : "Stopped"}
    </span>
  );
}
