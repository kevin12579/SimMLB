'use client'
import { useEffect, useState } from 'react'

interface LiveState {
  polled_at: string | null
  status: string
  inning: number | null
  half: string | null
  outs: number | null
  balls: number | null
  strikes: number | null
  home_score: number
  away_score: number
  bases: { first: boolean; second: boolean; third: boolean }
  mlb_win_prob: number | null
  live_home_prob: number
}

interface LiveStatesResponse {
  game_pk: number
  count: number
  states: LiveState[]
}

interface GamePrediction {
  game_pk: number
  home_team: string
  away_team: string
  home_win_prob: number
  confidence: string
  reasoning: string
  weather_temp_f: number | null
  weather_condition: string | null
  weather_wind: string | null
  home_lineup_preview?: string[]
  away_lineup_preview?: string[]
  live?: {
    status: string | null
    home_win_prob: number | null
    current_inning: number | null
    score_home: number | null
    score_away: number | null
    updated_at: string | null
  } | null
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function GameLivePage({ params }: { params: { pk: string } }) {
  const pk = parseInt(params.pk, 10)
  const [pred, setPred] = useState<GamePrediction | null>(null)
  const [series, setSeries] = useState<LiveState[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false

    async function load() {
      try {
        const [pRes, sRes] = await Promise.all([
          fetch(`${API_URL}/predictions/${pk}`),
          fetch(`${API_URL}/predictions/${pk}/live`),
        ])
        const pJson: GamePrediction = await pRes.json()
        const sJson: LiveStatesResponse = await sRes.json()
        if (!cancelled) {
          setPred(pJson)
          setSeries(sJson.states ?? [])
          setLoading(false)
        }
      } catch (e) {
        console.error(e)
        if (!cancelled) setLoading(false)
      }
    }

    load()
    // 진행 중인 경기는 1분마다 polling
    const id = setInterval(load, 60_000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [pk])

  if (loading) {
    return <div style={{ padding: 40, color: '#888' }}>로딩 중...</div>
  }
  if (!pred) {
    return <div style={{ padding: 40, color: '#c33' }}>예측 정보를 찾을 수 없습니다.</div>
  }

  const latest = series[series.length - 1]
  const isLive = pred.live?.status === 'In Progress'
  const isFinal = pred.live?.status === 'Final'

  return (
    <main style={{ padding: '32px', maxWidth: 980, margin: '0 auto', fontFamily: 'system-ui' }}>
      <div style={{ marginBottom: 8 }}>
        <a href="/" style={{ color: '#666', fontSize: 12 }}>← 오늘 예측</a>
      </div>
      <h1 style={{ fontSize: 26, fontWeight: 700, marginBottom: 4 }}>
        {pred.away_team} @ {pred.home_team}
      </h1>
      <div style={{ fontSize: 12, color: '#888', marginBottom: 16 }}>
        {isLive ? '🔴 LIVE' : isFinal ? 'FINAL' : 'PRE-GAME'} ·
        Pre-game 홈 승률 {(pred.home_win_prob * 100).toFixed(1)}% ·
        신뢰도 {pred.confidence}
      </div>

      {/* Live 스코어 카드 */}
      <div style={{
        border: '1px solid #ddd', borderRadius: 6, padding: 20, marginBottom: 20,
        background: isLive ? '#fff8f0' : '#f8f8f8',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-around', alignItems: 'center' }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 12, color: '#888' }}>AWAY</div>
            <div style={{ fontSize: 36, fontWeight: 700 }}>{pred.live?.score_away ?? 0}</div>
            <div style={{ fontSize: 14 }}>{pred.away_team}</div>
          </div>
          <div style={{ textAlign: 'center', color: '#666' }}>
            {latest && (
              <>
                <div style={{ fontSize: 11, letterSpacing: 1 }}>
                  {latest.inning}회 {latest.half} · {latest.outs}OUT
                </div>
                <div style={{ fontSize: 10, color: '#999', marginTop: 4 }}>
                  {latest.balls}-{latest.strikes}
                </div>
              </>
            )}
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 12, color: '#888' }}>HOME</div>
            <div style={{ fontSize: 36, fontWeight: 700 }}>{pred.live?.score_home ?? 0}</div>
            <div style={{ fontSize: 14 }}>{pred.home_team}</div>
          </div>
        </div>
        {pred.live?.home_win_prob != null && (
          <div style={{ marginTop: 14, textAlign: 'center', fontSize: 13 }}>
            라이브 홈 승률: <strong>{(pred.live.home_win_prob * 100).toFixed(1)}%</strong>
          </div>
        )}
      </div>

      {/* Lineup + Weather */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <div style={{ border: '1px solid #eee', borderRadius: 6, padding: 14 }}>
          <div style={{ fontSize: 11, color: '#888', letterSpacing: 1 }}>날씨</div>
          <div style={{ fontSize: 14, marginTop: 4 }}>
            {pred.weather_temp_f != null ? `${pred.weather_temp_f}°F` : '—'}
            {pred.weather_condition && ` · ${pred.weather_condition}`}
            {pred.weather_wind && ` · ${pred.weather_wind}`}
          </div>
        </div>
        <div style={{ border: '1px solid #eee', borderRadius: 6, padding: 14 }}>
          <div style={{ fontSize: 11, color: '#888', letterSpacing: 1 }}>라인업 상위</div>
          <div style={{ fontSize: 12, marginTop: 4 }}>
            <div><strong>{pred.away_team}</strong>: {(pred.away_lineup_preview ?? []).join(', ') || '—'}</div>
            <div style={{ marginTop: 4 }}><strong>{pred.home_team}</strong>: {(pred.home_lineup_preview ?? []).join(', ') || '—'}</div>
          </div>
        </div>
      </div>

      {/* 라이브 승률 시계열 (간단 ASCII 그래프) */}
      {series.length > 1 && (
        <div style={{ border: '1px solid #eee', borderRadius: 6, padding: 14, marginBottom: 20 }}>
          <div style={{ fontSize: 11, color: '#888', letterSpacing: 1, marginBottom: 8 }}>
            라이브 홈 승률 추이 (총 {series.length} 시점)
          </div>
          <svg width="100%" height="120" viewBox="0 0 600 120" preserveAspectRatio="none">
            <line x1="0" y1="60" x2="600" y2="60" stroke="#ddd" strokeDasharray="3,3" />
            <polyline
              fill="none"
              stroke="#1E4D8C"
              strokeWidth="2"
              points={series.map((s, i) => {
                const x = (i / Math.max(series.length - 1, 1)) * 600
                const y = 120 - s.live_home_prob * 120
                return `${x},${y}`
              }).join(' ')}
            />
          </svg>
          <div style={{ fontSize: 10, color: '#999', display: 'flex', justifyContent: 'space-between' }}>
            <span>1회</span><span>경기 진행 →</span><span>현재</span>
          </div>
        </div>
      )}

      {/* AI 근거 */}
      <div style={{ border: '1px solid #eee', borderRadius: 6, padding: 14 }}>
        <div style={{ fontSize: 11, color: '#888', letterSpacing: 1, marginBottom: 6 }}>AI 근거</div>
        <p style={{ fontSize: 13, lineHeight: 1.6 }}>{pred.reasoning || '근거 미생성'}</p>
      </div>
    </main>
  )
}
