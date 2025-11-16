import { useEffect, useState } from 'react'
import api from '../api/client'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

export default function InsuranceDashboard() {
  const [model, setModel] = useState(null)
  const [plan, setPlan] = useState({ deductible: 500, coinsurance: 0.2, oop_max: 4000 })

  useEffect(() => {
    api.get('/insurance/model').then(res => setModel(res.data))
  }, [])

  const simulate = () => {
    api.post('/insurance/simulate', plan).then(res => {
      alert(`Plan Cost PMPM: $${res.data.avg_plan_cost_pmpm.toFixed(2)}`)
    })
  }

  if (!model) return <p>Loading model...</p>

  return (
    <div className="space-y-6">
      <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow">
        <h2 className="text-xl font-bold">Age Rating Curve</h2>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={model.age_rating_table}>
            <XAxis dataKey="age" />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="predicted_pmpm" stroke="#2563eb" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow">
        <h3 className="font-semibold mb-4">Simulate Plan</h3>
        <div className="grid md:grid-cols-3 gap-4">
          <div>
            <label>Deductible</label>
            <input type="range" min="0" max="3000" step="100"
              value={plan.deductible} onChange={e => setPlan({...plan, deductible: +e.target.value})}
              className="w-full" />
            <p>${plan.deductible}</p>
          </div>
          <div>
            <label>Coinsurance</label>
            <input type="range" min="0" max="0.5" step="0.05"
              value={plan.coinsurance} onChange={e => setPlan({...plan, coinsurance: +e.target.value})}
              className="w-full" />
            <p>{(plan.coinsurance * 100).toFixed(0)}%</p>
          </div>
          <div>
            <label>OOP Max</label>
            <input type="range" min="1000" max="10000" step="500"
              value={plan.oop_max} onChange={e => setPlan({...plan, oop_max: +e.target.value})}
              className="w-full" />
            <p>${plan.oop_max}</p>
          </div>
        </div>
        <button onClick={simulate} className="mt-4 px-6 py-2 bg-primary text-white rounded">
          Simulate
        </button>
      </div>
    </div>
  )
}