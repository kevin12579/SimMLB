'use client'
import { useEffect, useState } from 'react'

interface ShapFeature {
  feature: string
  value: number
  shap_value: number
}

interface Prediction {
  game_pk: number
  home_team: string
  away_team: string
  home_win_prob: number
  away_win_prob: number
  confidence: 'HIGH' | 'MED' | 'LOW'
  reasoning: string
  top5_features: ShapFeature[]
}

interface ApiResponse {
  date: string
  games: Prediction[]
  message?: string
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const CONFIDENCE_STYLE: Record<string, string> = {
  HIGH: 'bg-green-100 text-green-800 border border-green-300',
  MED:  'bg-yellow-100 text-yellow-800 border border-yellow-300',
  LOW:  'bg-gray-100 text-gray-600 border border-gray-300',
}

const CONFIDENCE_LABEL: Record<string, string> = {
  HIGH: '높음',
  MED:  '보통',
  LOW:  '낮음',
}

function ProbBar({ prob, label }: { prob: number; label: string }) {
  const pct = Math.round(prob * 100)
  const color = pct >= 60 ? 'bg-blue-500' : pct >= 50 ? 'bg-blue-400' : 'bg-gray-300'
  return (
    <div className="flex items-center gap-2 text-sm">
      <span className="w-12 text-right font-mono font-semibold">{pct}%</span>
      <div className="flex-1 bg-gray-200 rounded-full h-2">
        <div className={`${color} h-2 rounded-full transition-all`} style={{ width: `${pct}%` }} />
      </div>
      <span className="w-20 text-gray-600 truncate">{label}</span>
    </div>
  )
}

function PredictionCard({ p }: { p: Prediction }) {
  const [expanded, setExpanded] = useState(false)
  return (
    <div className="border rounded-xl p-5 mb-4 shadow-sm bg-white hover:shadow-md transition-shadow">
      <div className="flex justify-between items-start mb-3">
        <div>
          <span className="text-lg font-bold">{p.away_team}</span>
          <span className="mx-2 text-gray-400">@</span>
          <span className="text-lg font-bold">{p.home_team}</span>
          <span className="ml-2 text-xs text-gray-400">(홈)</span>
        </div>
        <span className={`text-xs font-semibold px-2 py-1 rounded-full ${CONFIDENCE_STYLE[p.confidence]}`}>
          신뢰도 {CONFIDENCE_LABEL[p.confidence]}
        </span>
      </div>

      <div className="space-y-1 mb-3">
        <ProbBar prob={p.home_win_prob} label={`${p.home_team} 승`} />
        <ProbBar prob={p.away_win_prob} label={`${p.away_team} 승`} />
      </div>

      {p.reasoning && (
        <p className="text-sm text-gray-600 bg-gray-50 rounded-lg p-3 mb-2 leading-relaxed">
          {p.reasoning}
        </p>
      )}

      {p.top5_features?.length > 0 && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-xs text-blue-500 hover:underline mt-1"
        >
          {expanded ? '피처 숨기기' : 'SHAP 피처 보기'}
        </button>
      )}

      {expanded && (
        <div className="mt-2 space-y-1">
          {p.top5_features.map((f, i) => (
            <div key={i} className="flex justify-between text-xs text-gray-500">
              <span className="font-mono">{f.feature}</span>
              <span className={f.shap_value > 0 ? 'text-blue-500' : 'text-red-400'}>
                {f.shap_value > 0 ? '+' : ''}{f.shap_value.toFixed(3)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default function Home() {
  const [data, setData] = useState<ApiResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch(`${API_URL}/predictions/today`)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then((d: ApiResponse) => { setData(d); setLoading(false) })
      .catch(e => { setError(String(e.message)); setLoading(false) })
  }, [])

  return (
    <main className="min-h-screen bg-gray-50">
      <header className="bg-white border-b shadow-sm">
        <div className="max-w-3xl mx-auto px-4 py-4 flex items-center gap-3">
          <span className="text-2xl">⚾</span>
          <div>
            <h1 className="text-xl font-bold text-gray-900">MLB 승부 예측</h1>
            <p className="text-xs text-gray-500">
              {data?.date ?? new Date().toISOString().slice(0, 10)} 경기 AI 분석
            </p>
          </div>
        </div>
      </header>

      <div className="max-w-3xl mx-auto px-4 py-6">
        {loading && (
          <div className="text-center py-16 text-gray-400">
            <div className="text-4xl mb-3 animate-bounce">⚾</div>
            <p>예측 데이터 로딩 중...</p>
          </div>
        )}

        {error && (
          <div className="text-center py-16 text-red-400">
            <p className="font-semibold">API 연결 오류</p>
            <p className="text-sm mt-1">{error}</p>
            <p className="text-xs mt-2 text-gray-400">백엔드 서버가 실행 중인지 확인하세요.</p>
          </div>
        )}

        {!loading && !error && data && data.games.length === 0 && (
          <div className="text-center py-16 text-gray-400">
            <div className="text-4xl mb-3">📋</div>
            <p>{data.message || '오늘 예측 데이터가 아직 없습니다.'}</p>
            <p className="text-xs mt-1">매일 19:30 KST에 자동 생성됩니다.</p>
          </div>
        )}

        {!loading && !error && data?.games?.map(p => (
          <PredictionCard key={p.game_pk} p={p} />
        ))}
      </div>
    </main>
  )
}
