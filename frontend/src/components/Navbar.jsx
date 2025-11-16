// frontend/src/components/Navbar.jsx
import { NavLink } from 'react-router-dom'
import { User, Briefcase, DollarSign, FileText } from 'lucide-react'

const iconMap = {
  User,
  Briefcase,
  DollarSign,
  FileText,
}

export default function Navbar({ tabs, currentPath }) {
  return (
    <nav className="bg-white dark:bg-gray-800 shadow-lg sticky top-0 z-10">
      <div className="container mx-auto px-4 py-4 flex justify-between items-center">
        <h1 className="text-2xl font-bold text-primary">EHR Intelligence</h1>

        <div className="flex gap-3">
          {tabs.map((tab) => {
            const Icon = iconMap[tab.icon]
            return (
              <NavLink
                key={tab.path}
                to={tab.path}
                className={({ isActive }) =>
                  `flex items-center gap-2 px-5 py-2.5 rounded-lg transition ${
                    isActive
                      ? 'bg-primary text-white'
                      : 'bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600'
                  }`
                }
              >
                <Icon size={20} />
                {tab.label}
              </NavLink>
            )
          })}
        </div>
      </div>
    </nav>
  )
}