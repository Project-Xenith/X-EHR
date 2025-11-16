// frontend/src/App.jsx
import { BrowserRouter, Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { useState } from 'react'
import Navbar from './components/Navbar'
import PatientDashboard from './pages/PatientDashboard'
import OperationsDashboard from './pages/OperationsDashboard'
import BillingDashboard from './pages/BillingDashboard'
import InsuranceDashboard from './pages/InsuranceDashboard'
import { Moon, Sun } from 'lucide-react'

function Layout() {
  const [darkMode, setDarkMode] = useState(false)
  const location = useLocation()

  // Sync dark mode with <html>
  if (darkMode) {
    document.documentElement.classList.add('dark')
  } else {
    document.documentElement.classList.remove('dark')
  }

  const tabs = [
    { path: '/patient', label: 'Patient', icon: 'User' },
    { path: '/operations', label: 'Operations', icon: 'Briefcase' },
    { path: '/billing', label: 'Billing', icon: 'DollarSign' },
    { path: '/insurance', label: 'Insurance', icon: 'FileText' },
  ]

  return (
    <div className={darkMode ? 'dark' : ''}>
      <Navbar tabs={tabs} currentPath={location.pathname} />
      <div className="container mx-auto p-6">
        <Routes>
          <Route path="/patient" element={<PatientDashboard />} />
          <Route path="/operations" element={<OperationsDashboard />} />
          <Route path="/billing" element={<BillingDashboard />} />
          <Route path="/insurance" element={<InsuranceDashboard />} />
          <Route path="*" element={<Navigate to="/patient" replace />} />
        </Routes>
      </div>

      {/* Dark Mode Toggle – floating bottom right */}
      <button
        onClick={() => setDarkMode(!darkMode)}
        className="fixed bottom-6 right-6 p-3 bg-gray-200 dark:bg-gray-700 rounded-full shadow-lg hover:scale-110 transition"
      >
        {darkMode ? <Sun size={24} /> : <Moon size={24} />}
      </button>
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Layout />
    </BrowserRouter>
  )
}