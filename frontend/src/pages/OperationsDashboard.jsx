import { useState, useEffect } from 'react'
import api from '../api/client'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

export default function OperationsDashboard() {
  const [insights, setInsights] = useState(null)

  useEffect(() => {
    api.get('/operations/insights?start_date=2023-01-01&end_date=2023-12-31').then(res => {
      setInsights(res.data)
    })
  }, [])

  if (!insights) return <p>Loading...</p>

  const diagData = Object.entries(insights.top_diagnoses).map(([code, count]) => ({ code, count }))
  const procData = Object.entries(insights.top_procedures).map(([code, count]) => ({ code, count }))

  return (
    <div className="space-y-6">
      <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow">
        <h2 className="text-xl font-bold mb-2">Operational Insights</h2>
        <p className="text-lg">Active Patients: <strong>{insights.total_active_patients}</strong></p>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow">
          <h3 className="font-semibold mb-4">Top Diagnoses</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={diagData}>
              <XAxis dataKey="code" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#2563eb" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow">
          <h3 className="font-semibold mb-4">Top Procedures</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={procData}>
              <XAxis dataKey="code" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#16a34a" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}