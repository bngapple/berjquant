import { Routes, Route } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Dashboard } from "./pages/Dashboard";
import { Calendar } from "./pages/Calendar";
import { Cockpit } from "./pages/Cockpit";
import { Setup } from "./pages/Setup";
import { Settings } from "./pages/Settings";
import { Terminal } from "./pages/Terminal";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/terminal" element={<Terminal />} />
        <Route path="/calendar" element={<Calendar />} />
        <Route path="/cockpit" element={<Cockpit />} />
        <Route path="/setup" element={<Setup />} />
        <Route path="/settings" element={<Settings />} />
      </Route>
    </Routes>
  );
}
