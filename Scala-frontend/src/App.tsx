import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import MainLayout from './MainLayout/MainLayout';
import Home from './components/Home';
import Scanner from './components/Scanner';
import BatchAudit from './components/BatchAudit';
import Dashboard from './components/Dashboard';
import History from './components/History';
import Prediction from './components/Prediction';
import './App.css';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<MainLayout />}>
          <Route path="/" element={<Home />} />
          <Route path="/scanner" element={<Scanner />} />
          <Route path="/prediction" element={<Prediction />} />
          <Route path="/batch" element={<BatchAudit />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/history" element={<History />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
