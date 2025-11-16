import { useState } from 'react'
import api from '../api/client'
import { format } from 'date-fns'

export default function PatientDashboard() {
  const [patientId, setPatientId] = useState('')
  const [summary, setSummary] = useState(null)
  const [loading, setLoading] = useState(false)

  const fetchSummary = async () => {
    setLoading(true)
    try {
      const res = await api.get(`/patient/${patientId}/summary`)
      setSummary(res.data)
    } catch (err) {
      alert('Patient not found or error')
    }
    setLoading(false)
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow">
        <h2 className="text-xl font-bold mb-4">Patient Summary</h2>
        <div className="flex gap-2 mb-4">
          <input
            type="text"
            placeholder="Enter Patient ID"
            value={patientId}
            onChange={e => setPatientId(e.target.value)}
            className="flex-1 p-2 border rounded dark:bg-gray-700"
          />
          <button
            onClick={fetchSummary}
            disabled={loading}
            className="px-6 py-2 bg-primary text-white rounded"
          >
            {loading ? 'Loading...' : 'Search'}
          </button>
        </div>

        {summary && (
          <div className="space-y-4">
            <div className="bg-blue-50 dark:bg-blue-900 p-4 rounded">
              <h3 className="font-semibold">Medical Summary</h3>
              <p className="whitespace-pre-wrap">{summary.medical_summary}</p>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-green-50 dark:bg-green-900 p-4 rounded">
                <p className="text-sm">Total Submitted</p>
                <p className="text-2xl font-bold">${summary.total_submitted.toFixed(2)}</p>
              </div>
              <div className="bg-red-50 dark:bg-red-900 p-4 rounded">
                <p className="text-sm">Total Paid</p>
                <p className="text-2xl font-bold">${summary.total_paid.toFixed(2)}</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}