import { useEffect, useState } from 'react'
import api from '../api/client'

export default function BillingDashboard() {
  const [data, setData] = useState(null)

  useEffect(() => {
    api.get('/billing/validate').then(res => setData(res.data))
  }, [])

  if (!data) return <p>Loading...</p>

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-red-100 dark:bg-red-900 p-6 rounded-xl text-center">
          <p className="text-3xl font-bold">{data.total_invalid}</p>
          <p>Invalid Claims</p>
        </div>
        <div className="bg-orange-100 dark:bg-orange-900 p-6 rounded-xl text-center">
          <p className="text-3xl font-bold">{data.total_duplicates}</p>
          <p>Duplicates</p>
        </div>
        <div className="bg-yellow-100 dark:bg-yellow-900 p-6 rounded-xl text-center">
          <p className="text-3xl font-bold">{data.total_long_turnaround}</p>
          <p>Long Turnaround</p>
        </div>
      </div>
    </div>
  )
}