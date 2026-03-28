interface Props { onConfirm: () => void; onCancel: () => void; }

export function FlattenModal({ onConfirm, onCancel }: Props) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" onClick={onCancel} style={{ background: "rgba(0,0,0,0.7)" }}>
      <div className="panel rounded-lg p-6 w-96" onClick={e => e.stopPropagation()}>
        <p className="text-sm font-medium mb-1" style={{ color: "var(--text)" }}>Flatten All Positions</p>
        <p className="text-xs mb-5" style={{ color: "var(--text-muted)" }}>This will flatten ALL positions on ALL accounts. Are you sure?</p>
        <div className="flex gap-2 justify-end">
          <button onClick={onCancel} className="px-4 py-1.5 text-xs rounded" style={{ color: "var(--text-muted)", border: "1px solid var(--border)" }}>Cancel</button>
          <button onClick={onConfirm} className="px-4 py-1.5 text-xs rounded bg-red-600 hover:bg-red-500 text-white">Flatten</button>
        </div>
      </div>
    </div>
  );
}
