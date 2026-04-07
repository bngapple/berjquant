import { Routes, Route, Navigate } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Dashboard } from "./pages/Dashboard";
import { Calendar } from "./pages/Calendar";
import { Cockpit } from "./pages/Cockpit";
import { Setup } from "./pages/Setup";
import { Settings } from "./pages/Settings";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/calendar" element={<Calendar />} />
        <Route path="/cockpit" element={<Cockpit />} />
        <Route path="/setup" element={<Setup />} />
        <Route path="/settings" element={<Settings />} />
      </Route>
    </Routes>
  );
}
