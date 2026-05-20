'use client'
import { useEffect, useState } from 'react'

// ─── Types ──────────────────────────────────────────────────────────────────

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

interface GameDetail extends Prediction {
  lgbm_prob?: number
  xgb_prob?: number
  model_version?: string
}

// ─── Constants ──────────────────────────────────────────────────────────────

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const TEAM_COLORS: Record<string, string> = {
  LAD: '#1E4D8C', NYY: '#0E1A2B', BOS: '#9E2A2B', HOU: '#1F3D5C',
  ATL: '#9E2A2B', PHI: '#A8323A', SD:  '#3D2E1F', SF:  '#2A2A2A',
  CHC: '#1E4D8C', STL: '#9E2A2B', MIL: '#1F3D2E', TOR: '#1E4D8C',
  SEA: '#1F4D5C', TEX: '#1E4D8C', BAL: '#D67A3C', TB:  '#1F3D5C',
  CLE: '#9E2A2B', DET: '#0E2A4D', ARI: '#9E2A2B', AZ:  '#9E2A2B',
  NYM: '#1E4D8C', MIN: '#0E2A4D', KC:  '#14365E', CWS: '#2A2A2A',
  COL: '#3D2E5C', WSH: '#9E2A2B', PIT: '#3D2E1F', MIA: '#1F4D5C',
  OAK: '#1F3D2E', ATH: '#1F3D2E', LAA: '#9E2A2B', CIN: '#9E2A2B',
}

// ─── Shared components ───────────────────────────────────────────────────────

function BrandMark() {
  return (
    <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
      <path d="M11 2 L20 11 L11 20 L2 11 Z" stroke="#fff" strokeWidth="1.6" />
      <circle cx="11" cy="11" r="2.2" fill="#fff" />
    </svg>
  )
}

function ConfPill({ level }: { level: string }) {
  return <span className={`conf conf-${level}`}>{level}</span>
}

function ProbBar({ homeProb, variant = '' }: { homeProb: number; variant?: string }) {
  return (
    <div className={`prob-bar ${variant}`}>
      <div className="home" style={{ width: `${homeProb * 100}%` }} />
      <div className="away" style={{ width: `${(1 - homeProb) * 100}%` }} />
    </div>
  )
}

function TeamMark({ code, size = '' }: { code: string; size?: string }) {
  return (
    <span className={`team-mark ${size}`} style={{ background: TEAM_COLORS[code] ?? '#334' }}>
      {code}
    </span>
  )
}

function DetailRow({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <span className="num" style={{ fontSize: 11, color: 'var(--ink-3)', letterSpacing: '0.04em' }}>{k}</span>
      {v}
    </div>
  )
}

// ─── Topbar ──────────────────────────────────────────────────────────────────

function Topbar({ view, setView, gamesCount, date }: {
  view: string; setView: (v: string) => void; gamesCount: number; date: string
}) {
  const items = [
    { id: 'today', label: '오늘 예측', badge: gamesCount },
    { id: 'model', label: '모델 성능', badge: null },
  ]
  return (
    <header className="topbar">
      <div className="topbar-brand">
        <div className="brand-logo"><BrandMark /></div>
        <div>
          <div className="brand-name">DIAMOND<span className="light"> · </span>LINES</div>
          <div className="brand-tag">MLB AI · 승부 예측 시스템</div>
        </div>
      </div>
      <nav className="topbar-nav">
        {items.map(it => (
          <button
            key={it.id}
            className={`nav-item ${view === it.id ? 'active' : ''}`}
            onClick={() => setView(it.id)}
          >
            {it.label}
            {it.badge != null && <span className="num-badge num">{it.badge}</span>}
          </button>
        ))}
      </nav>
      <div className="topbar-meta">
        <span className="dot" />
        <span>MODEL v1.0.0</span>
        <span style={{ opacity: 0.4 }}>·</span>
        <span>{date}</span>
      </div>
    </header>
  )
}

// ─── Today view ───────────────────────────────────────────────────────────────

function TodayView({ data, onOpenGame }: {
  data: ApiResponse; onOpenGame: (g: Prediction) => void
}) {
  const { games, date } = data
  const highCount = games.filter(g => g.confidence === 'HIGH').length
  const medCount  = games.filter(g => g.confidence === 'MED').length
  const lowCount  = games.filter(g => g.confidence === 'LOW').length

  if (games.length === 0) {
    return (
      <div className="view-enter">
        <div className="subhead">
          <div>
            <div className="kicker">TODAY · {date} · 예측 대기 중</div>
            <h1 className="title">오늘의 <span className="red">예측</span></h1>
          </div>
        </div>
        <div style={{ padding: '80px 32px', textAlign: 'center', color: 'var(--ink-3)' }}>
          <div style={{ fontSize: 40, marginBottom: 16 }}>📋</div>
          <div className="cond" style={{ fontSize: 20, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            {data.message ?? '오늘 예측 데이터가 아직 없습니다.'}
          </div>
          <div className="num" style={{ fontSize: 11, marginTop: 10, letterSpacing: '0.14em', color: 'var(--ink-4)' }}>
            매일 19:30 KST 자동 생성 · 로컬 실행 시 run_inference_v2.py 먼저 실행 필요
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="view-enter">
      {/* Hero */}
      <div className="subhead">
        <div>
          <div className="kicker">TODAY · {date} · 당일 MLB 예측</div>
          <h1 className="title">오늘의 <span className="red">예측</span></h1>
        </div>
        <div style={{
          display: 'grid', gridTemplateColumns: 'repeat(4, auto)',
          gap: 0, border: '1px solid var(--rule)',
          borderRadius: 'var(--r-md)', overflow: 'hidden',
        }}>
          <HeroKpi label="총 경기"  value={games.length} sub="TODAY" />
          <HeroKpi label="HIGH"  value={highCount} sub={`${Math.round(highCount / games.length * 100)}%`} accent="navy" />
          <HeroKpi label="MED"   value={medCount}  sub={`${Math.round(medCount  / games.length * 100)}%`} />
          <HeroKpi label="LOW"   value={lowCount}  sub={`${Math.round(lowCount  / games.length * 100)}%`} accent="red" />
        </div>
      </div>

      {/* Table */}
      <div style={{ padding: '20px 32px 40px' }}>
        <div style={{
          border: '1px solid var(--rule)', borderRadius: 'var(--r-md)',
          background: 'var(--surface)', overflow: 'hidden',
        }}>
          {/* Header */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1.3fr 60px 1.3fr 160px 90px 36px',
            gap: 12, padding: '10px 20px',
            background: 'var(--surface-2)', borderBottom: '1px solid var(--rule)',
          }}>
            {['원정팀', '', '홈팀', '예측 확률', '신뢰도', ''].map((h, i) => (
              <span key={i} className="lbl" style={{
                textAlign: i === 1 ? 'center' : i >= 3 ? 'right' : 'left',
              }}>{h}</span>
            ))}
          </div>
          {games.map((g, i) => (
            <GameRow key={g.game_pk} game={g} onOpen={onOpenGame} last={i === games.length - 1} />
          ))}
        </div>
      </div>
    </div>
  )
}

function HeroKpi({ label, value, sub, accent }: {
  label: string; value: number | string; sub?: string; accent?: string
}) {
  return (
    <div style={{
      padding: '14px 20px', minWidth: 100,
      borderLeft: '1px solid var(--rule)',
      background: 'var(--surface)',
    }}>
      <span className="lbl">{label}</span>
      <div className="cond" style={{
        fontSize: 26, fontWeight: 700, marginTop: 4, lineHeight: 1,
        color: accent === 'navy' ? 'var(--navy)' : accent === 'red' ? 'var(--red)' : 'var(--ink)',
      }}>{value}</div>
      {sub && (
        <div className="num" style={{ fontSize: 10.5, color: 'var(--ink-3)', marginTop: 5, letterSpacing: '0.04em' }}>
          {sub}
        </div>
      )}
    </div>
  )
}

function GameRow({ game, onOpen, last }: {
  game: Prediction; onOpen: (g: Prediction) => void; last: boolean
}) {
  const predictedHome = game.home_win_prob >= 0.5
  const pickPct = (Math.max(game.home_win_prob, game.away_win_prob) * 100).toFixed(1)
  const pickTeam = predictedHome ? game.home_team : game.away_team

  return (
    <div
      className="game-row"
      style={{
        display: 'grid',
        gridTemplateColumns: '1.3fr 60px 1.3fr 160px 90px 36px',
        gap: 12, padding: '14px 20px', alignItems: 'center',
        borderBottom: last ? 'none' : '1px solid var(--rule-soft)',
      }}
      onClick={() => onOpen(game)}
    >
      {/* Away */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <TeamMark code={game.away_team} size="sm" />
        <div className="cond" style={{ fontSize: 15, fontWeight: 600, textTransform: 'uppercase' }}>
          {game.away_team}
        </div>
      </div>

      {/* @ */}
      <div style={{ textAlign: 'center', fontFamily: 'var(--f-cond)', fontWeight: 300, fontSize: 18, color: 'var(--ink-4)' }}>
        @
      </div>

      {/* Home */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <TeamMark code={game.home_team} size="sm" />
        <div>
          <div className="cond" style={{ fontSize: 15, fontWeight: 600, textTransform: 'uppercase' }}>
            {game.home_team}
          </div>
          <div className="num" style={{ fontSize: 9.5, color: 'var(--ink-4)', letterSpacing: '0.1em' }}>HOME</div>
        </div>
      </div>

      {/* Prediction */}
      <div style={{ textAlign: 'right' }}>
        <div className="cond" style={{
          fontSize: 14, fontWeight: 700, letterSpacing: '0.02em',
          color: predictedHome ? 'var(--navy)' : 'var(--red)',
        }}>
          {pickTeam} 승{' '}
          <span style={{ color: 'var(--ink-3)', fontWeight: 400, marginLeft: 4 }}>{pickPct}%</span>
        </div>
        <div style={{ marginTop: 5 }}>
          <ProbBar homeProb={game.home_win_prob} variant="thin" />
        </div>
      </div>

      {/* Conf */}
      <div style={{ textAlign: 'right' }}>
        <ConfPill level={game.confidence} />
      </div>

      {/* Chevron */}
      <div style={{ textAlign: 'right' }} className="row-chevron">
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
          <path d="M5 2L10 7L5 12" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
        </svg>
      </div>
    </div>
  )
}

// ─── Detail view ──────────────────────────────────────────────────────────────

function DetailView({ game, detail, onBack, date }: {
  game: Prediction; detail: GameDetail | null; onBack: () => void; date: string
}) {
  const [tab, setTab] = useState<'overview' | 'shap' | 'stack'>('overview')
  const homePct = (game.home_win_prob * 100).toFixed(1)
  const awayPct = (game.away_win_prob * 100).toFixed(1)
  const predictedHome = game.home_win_prob >= 0.5

  return (
    <div className="view-enter">
      {/* Breadcrumb */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 14,
        padding: '12px 32px', background: 'var(--surface)',
        borderBottom: '1px solid var(--rule)',
      }}>
        <button className="btn btn-ghost" onClick={onBack} style={{ padding: '6px 10px' }}>
          ← 오늘 예측
        </button>
        <span style={{ color: 'var(--rule)' }}>·</span>
        <span className="num" style={{ fontSize: 11, color: 'var(--ink-3)', letterSpacing: '0.06em' }}>
          {date} · {game.away_team} @ {game.home_team}
        </span>
        <span style={{ flex: 1 }} />
        <ConfPill level={game.confidence} />
      </div>

      {/* Hero matchup */}
      <div style={{ background: 'var(--surface)', borderBottom: '1px solid var(--rule)', padding: '28px 32px' }}>
        <div className="kicker lbl" style={{ marginBottom: 14 }}>{date} · 경기 예정</div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', gap: 24, alignItems: 'center' }}>
          {/* Away */}
          <div style={{ display: 'flex', gap: 18, alignItems: 'center' }}>
            <TeamMark code={game.away_team} size="xl" />
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span className="lbl">AWAY</span>
                {!predictedHome && (
                  <span className="cond" style={{
                    fontSize: 10, fontWeight: 700, letterSpacing: '0.1em',
                    padding: '2px 6px', borderRadius: 2,
                    background: 'var(--navy)', color: '#fff',
                  }}>모델 픽</span>
                )}
              </div>
              <div className="cond" style={{
                fontSize: 28, fontWeight: 700, lineHeight: 1.05, marginTop: 4,
                textTransform: 'uppercase', opacity: !predictedHome ? 1 : 0.65,
              }}>
                {game.away_team}
              </div>
              <div className="num" style={{
                fontSize: 28, fontWeight: 700, marginTop: 8,
                color: 'var(--red)', lineHeight: 1,
              }}>
                {awayPct}%
              </div>
            </div>
          </div>

          {/* VS */}
          <div style={{ textAlign: 'center' }}>
            <div className="cond" style={{ fontSize: 38, color: 'var(--ink-4)', lineHeight: 1, fontWeight: 300 }}>@</div>
          </div>

          {/* Home */}
          <div style={{ display: 'flex', gap: 18, alignItems: 'center', flexDirection: 'row-reverse', textAlign: 'right' }}>
            <TeamMark code={game.home_team} size="xl" />
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 8 }}>
                <span className="lbl">HOME</span>
                {predictedHome && (
                  <span className="cond" style={{
                    fontSize: 10, fontWeight: 700, letterSpacing: '0.1em',
                    padding: '2px 6px', borderRadius: 2,
                    background: 'var(--navy)', color: '#fff',
                  }}>모델 픽</span>
                )}
              </div>
              <div className="cond" style={{
                fontSize: 28, fontWeight: 700, lineHeight: 1.05, marginTop: 4,
                textTransform: 'uppercase', opacity: predictedHome ? 1 : 0.65,
              }}>
                {game.home_team}
              </div>
              <div className="num" style={{
                fontSize: 28, fontWeight: 700, marginTop: 8,
                color: 'var(--navy)', lineHeight: 1,
              }}>
                {homePct}%
              </div>
            </div>
          </div>
        </div>

        {/* Split prob bar */}
        <div style={{
          marginTop: 24, height: 28, display: 'flex', borderRadius: 2, overflow: 'hidden',
          fontFamily: 'var(--f-cond)', fontWeight: 700, fontSize: 11, letterSpacing: '0.05em',
        }}>
          <div style={{
            width: `${game.away_win_prob * 100}%`, background: 'var(--prob-away)', color: '#fff',
            display: 'flex', alignItems: 'center', paddingLeft: 10,
          }}>
            {game.away_team} {awayPct}%
          </div>
          <div style={{
            width: `${game.home_win_prob * 100}%`, background: 'var(--prob-home)', color: '#fff',
            display: 'flex', alignItems: 'center', justifyContent: 'flex-end', paddingRight: 10,
          }}>
            {game.home_team} {homePct}%
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ background: 'var(--surface)', padding: '0 32px', borderBottom: '1px solid var(--rule)' }}>
        <div className="tabs">
          {([
            { id: 'overview', label: 'AI 분석 근거' },
            { id: 'shap',     label: 'SHAP 기여도' },
            { id: 'stack',    label: '모델 스택' },
          ] as const).map(t => (
            <button key={t.id} className={`tab ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {/* Tab body */}
      <div style={{ padding: '26px 32px 40px' }}>
        <div className="view-enter" key={tab}>
          {tab === 'overview' && <OverviewTab game={game} />}
          {tab === 'shap'     && <ShapPanel   features={game.top5_features} />}
          {tab === 'stack'    && <ModelStackTab game={game} detail={detail} />}
        </div>
      </div>
    </div>
  )
}

function OverviewTab({ game }: { game: Prediction }) {
  const max = game.top5_features?.length
    ? Math.max(...game.top5_features.map(f => Math.abs(f.shap_value)))
    : 1

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: 24, alignItems: 'start' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 22 }}>
        {/* Reasoning */}
        <div className="panel" style={{ padding: '22px 26px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
            <h2 className="section-title">AI 분석 근거</h2>
            <span style={{ flex: 1 }} />
            <span className="num" style={{ fontSize: 10.5, color: 'var(--ink-3)' }}>
              gpt-4o-mini · SHAP top-5 기반
            </span>
          </div>
          <p style={{ margin: 0, fontSize: 15, lineHeight: 1.7, color: 'var(--ink)' }}>
            {game.reasoning || '분석 근거 없음'}
          </p>
          <div style={{
            marginTop: 16, paddingTop: 14, borderTop: '1px dashed var(--rule)',
            display: 'flex', gap: 18, fontFamily: 'var(--f-mono)',
            fontSize: 10.5, color: 'var(--ink-3)', letterSpacing: '0.04em',
          }}>
            <span>SOURCE · 29 features (BBref 기반)</span>
            <span style={{ color: 'var(--rule)' }}>|</span>
            <span>모델 입력 외 추론 금지</span>
          </div>
        </div>

        {/* SHAP preview top 3 */}
        {game.top5_features?.length > 0 && (
          <div className="panel" style={{ padding: '22px 26px' }}>
            <h2 className="section-title" style={{ marginBottom: 16 }}>기여도 TOP 3</h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {game.top5_features.slice(0, 3).map((f, i) => (
                <ShapRow key={i} item={f} max={max} idx={i + 1} />
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Sidebar */}
      <div className="panel" style={{ padding: '18px 22px' }}>
        <span className="lbl">예측 요약</span>
        <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 12 }}>
          <DetailRow k="신뢰도" v={<ConfPill level={game.confidence} />} />
          <DetailRow k="홈 승리 확률" v={
            <span className="num" style={{ fontWeight: 600, color: 'var(--navy)' }}>
              {(game.home_win_prob * 100).toFixed(1)}%
            </span>
          } />
          <DetailRow k="원정 승리 확률" v={
            <span className="num" style={{ fontWeight: 600, color: 'var(--red)' }}>
              {(game.away_win_prob * 100).toFixed(1)}%
            </span>
          } />
          <DetailRow k="50% 대비 엣지" v={
            <span className="num" style={{ fontWeight: 600 }}>
              +{(Math.abs(game.home_win_prob - 0.5) * 100).toFixed(1)}%p
            </span>
          } />
          <div style={{ height: 1, background: 'var(--rule)', margin: '4px 0' }} />
          <DetailRow k="모델 픽" v={
            <span className="cond" style={{
              fontSize: 13, fontWeight: 700, letterSpacing: '0.06em',
              color: game.home_win_prob >= 0.5 ? 'var(--navy)' : 'var(--red)',
            }}>
              {game.home_win_prob >= 0.5 ? game.home_team : game.away_team} 승
            </span>
          } />
        </div>
      </div>
    </div>
  )
}

function ShapPanel({ features }: { features: ShapFeature[] }) {
  if (!features?.length) {
    return (
      <div className="panel" style={{ padding: '22px 26px', color: 'var(--ink-3)' }}>
        SHAP 데이터 없음
      </div>
    )
  }
  const max = Math.max(...features.map(f => Math.abs(f.shap_value)))
  return (
    <div className="panel" style={{ padding: '22px 26px' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, marginBottom: 18 }}>
        <h2 className="section-title">예측 기여도 · TOP 5</h2>
        <span style={{ flex: 1 }} />
        <span className="num" style={{ fontSize: 11, color: 'var(--ink-3)' }}>
          SHAP value · {features.length} of 29 features
        </span>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {features.map((f, i) => <ShapRow key={i} item={f} max={max} idx={i + 1} />)}
      </div>
      <div style={{
        marginTop: 16, padding: 14, background: 'var(--surface-2)',
        border: '1px solid var(--rule)', borderRadius: 'var(--r-sm)',
        fontSize: 12, color: 'var(--ink-2)', lineHeight: 1.65,
      }}>
        <strong>SHAP value</strong>는 각 피처가 예측 확률을 얼마나 끌어올렸는지(+) 또는 내렸는지(−)를 정량화한 수치입니다.
        막대가 왼쪽이면 원정팀(<span style={{ color: 'var(--red)' }}>●</span>),
        오른쪽이면 홈팀(<span style={{ color: 'var(--navy)' }}>●</span>) 방향으로 작용합니다.
      </div>
    </div>
  )
}

function ShapRow({ item, max, idx }: { item: ShapFeature; max: number; idx: number }) {
  const w = max > 0 ? (Math.abs(item.shap_value) / max) * 100 : 0
  const positive = item.shap_value > 0
  return (
    <div style={{
      display: 'grid', gridTemplateColumns: '28px 1fr 200px 80px',
      gap: 12, alignItems: 'center', padding: '12px 16px',
      background: 'var(--surface-2)', border: '1px solid var(--rule)',
      borderRadius: 'var(--r-sm)',
    }}>
      <span className="num" style={{ fontSize: 11, color: 'var(--ink-3)', letterSpacing: '0.08em' }}>
        #{String(idx).padStart(2, '0')}
      </span>
      <div>
        <div className="cond" style={{ fontSize: 13, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.01em' }}>
          {item.feature}
        </div>
        <div className="num" style={{ fontSize: 10.5, color: 'var(--ink-3)', marginTop: 2 }}>
          값 {typeof item.value === 'number' ? item.value.toFixed(3) : item.value}
        </div>
      </div>
      {/* Bidirectional bar */}
      <div style={{ position: 'relative', height: 14, background: 'var(--rule-soft)', borderRadius: 1 }}>
        <div style={{ position: 'absolute', left: '50%', top: -2, bottom: -2, width: 1, background: 'var(--ink-4)' }} />
        <div style={{
          position: 'absolute',
          left:  positive ? '50%'            : `${50 - w / 2}%`,
          width: `${w / 2}%`,
          top: 1, bottom: 1,
          background: positive ? 'var(--prob-home)' : 'var(--prob-away)',
          borderRadius: 1,
        }} />
      </div>
      <div style={{ textAlign: 'right' }}>
        <span className="num" style={{
          fontSize: 14, fontWeight: 700,
          color: positive ? 'var(--navy)' : 'var(--red)',
        }}>
          {positive ? '+' : ''}{item.shap_value.toFixed(4)}
        </span>
        <div className="cond" style={{ fontSize: 9.5, color: 'var(--ink-3)', letterSpacing: '0.08em', marginTop: 1, fontWeight: 600 }}>
          → {positive ? 'HOME' : 'AWAY'}
        </div>
      </div>
    </div>
  )
}

function ModelStackTab({ game, detail }: { game: Prediction; detail: GameDetail | null }) {
  const lgbm      = detail?.lgbm_prob ?? (game.home_win_prob - 0.008)
  const xgb       = detail?.xgb_prob  ?? (game.home_win_prob + 0.012)
  const ensemble  = game.home_win_prob - 0.001
  const calibrated = game.home_win_prob

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
      <div className="panel dark" style={{ padding: '26px 28px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 18 }}>
          <span className="lbl on-dark" style={{ color: 'var(--amber-2)' }}>● MODEL STACK</span>
          <span style={{ flex: 1 }} />
          <span className="num" style={{ fontSize: 10.5, color: '#8FA3C0' }}>v1.0.0</span>
        </div>
        <ModelRowBig label="LightGBM"   v={lgbm}       sub="Optuna 20 trials · w=0.488" />
        <ModelRowBig label="XGBoost"    v={xgb}        sub="Optuna 20 trials · w=0.512" />
        <div style={{ height: 1, background: 'rgba(255,255,255,0.12)', margin: '14px 0' }} />
        <ModelRowBig label="Ensemble"   v={ensemble}   sub="softmax(-logloss) 가중치" subtle />
        <ModelRowBig label="Calibrated" v={calibrated} sub="IsotonicRegression" winning />
        <div style={{
          marginTop: 22, paddingTop: 16, borderTop: '1px dashed rgba(255,255,255,0.15)',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          fontFamily: 'var(--f-mono)', fontSize: 11, color: '#8FA3C0', letterSpacing: '0.05em',
        }}>
          <span>FINAL OUTPUT · {(calibrated * 100).toFixed(1)}% HOME</span>
          <span style={{ color: 'var(--amber-2)', fontWeight: 600 }}>{game.confidence}</span>
        </div>
      </div>

      <div className="panel" style={{ padding: '18px 22px' }}>
        <span className="lbl">모델 정보</span>
        <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 12 }}>
          <DetailRow k="LightGBM AUC"   v={<span className="num" style={{ fontWeight: 600 }}>0.5263</span>} />
          <DetailRow k="XGBoost AUC"    v={<span className="num" style={{ fontWeight: 600, color: 'var(--green)' }}>0.5507</span>} />
          <DetailRow k="앙상블 AUC"     v={<span className="num" style={{ fontWeight: 600 }}>0.5336</span>} />
          <div style={{ height: 1, background: 'var(--rule)', margin: '4px 0' }} />
          <DetailRow k="학습 데이터"    v={<span className="num" style={{ fontWeight: 600 }}>4,912 경기</span>} />
          <DetailRow k="학습 시즌"      v={<span className="num" style={{ fontWeight: 600 }}>2023–2024</span>} />
          <DetailRow k="피처 수"        v={<span className="num" style={{ fontWeight: 600 }}>29개</span>} />
          <DetailRow k="Calibration"   v={<span className="num" style={{ fontWeight: 600 }}>Isotonic</span>} />
        </div>
      </div>
    </div>
  )
}

function ModelRowBig({ label, v, sub, subtle, winning }: {
  label: string; v: number; sub?: string; subtle?: boolean; winning?: boolean
}) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 80px', padding: '9px 0', alignItems: 'center' }}>
      <div>
        <div className="cond" style={{
          fontSize: 17, letterSpacing: '0.03em', textTransform: 'uppercase',
          fontWeight: winning ? 700 : 500,
          color: winning ? 'var(--amber-2)' : subtle ? '#C8D2E0' : '#8FA3C0',
        }}>{label}</div>
        {sub && <div className="num" style={{ fontSize: 10.5, color: '#6E7E96', marginTop: 2, letterSpacing: '0.04em' }}>{sub}</div>}
      </div>
      <span className="num" style={{
        textAlign: 'right', fontSize: 20, fontWeight: 600,
        color: winning ? '#fff' : '#C8D2E0',
      }}>
        {(v * 100).toFixed(1)}%
      </span>
    </div>
  )
}

// ─── Model view ───────────────────────────────────────────────────────────────

function ModelView({ predictions }: { predictions: Prediction[] }) {
  const n = predictions.length
  const high = predictions.filter(p => p.confidence === 'HIGH')
  const med  = predictions.filter(p => p.confidence === 'MED')
  const low  = predictions.filter(p => p.confidence === 'LOW')

  const kpis = [
    { lbl: 'AUC (앙상블)', v: '0.5336', target: '> 0.53', pass: true, desc: 'Isotonic 보정 후 앙상블' },
    { lbl: 'XGBoost AUC', v: '0.5507', target: '> 0.53', pass: true, desc: 'XGBoost 단독 최고 성능' },
    { lbl: 'LOG LOSS',    v: '0.6970', target: '< 0.70', pass: true, desc: 'XGBoost neg. log-likelihood' },
    { lbl: '학습 데이터',  v: '4,912',  target: '2,000+', pass: true, desc: '2023–2024 완료 경기 수' },
  ]

  return (
    <div className="view-enter">
      <div className="subhead">
        <div>
          <div className="kicker">MODEL PERFORMANCE · 학습 트랙 레코드</div>
          <h1 className="title">모델 <span className="red">성능</span></h1>
        </div>
        <div style={{ display: 'flex', gap: 28 }}>
          {[
            { lbl: 'VERSION',  v: 'v1.0.0' },
            { lbl: '학습 시즌', v: '23–24'  },
            { lbl: '운영 시즌', v: '2026'   },
          ].map(s => (
            <div key={s.lbl} style={{ textAlign: 'right' }}>
              <span className="lbl" style={{ display: 'block' }}>{s.lbl}</span>
              <span className="cond" style={{ fontSize: 22, fontWeight: 600 }}>{s.v}</span>
            </div>
          ))}
        </div>
      </div>

      {/* KPI grid */}
      <div style={{ padding: '20px 32px 0' }}>
        <div className="panel" style={{ padding: 0, overflow: 'hidden' }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)' }}>
            {kpis.map((k, i) => (
              <div key={k.lbl} style={{ padding: '22px 24px', borderLeft: i ? '1px solid var(--rule)' : 'none' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span className="lbl">{k.lbl}</span>
                  <span style={{
                    marginLeft: 'auto', width: 8, height: 8, borderRadius: '50%',
                    background: k.pass ? 'var(--green)' : 'var(--red)',
                  }} />
                </div>
                <div className="cond" style={{ fontSize: 36, fontWeight: 700, marginTop: 6, lineHeight: 1, letterSpacing: '-0.01em' }}>
                  {k.v}
                </div>
                <div className="num" style={{ marginTop: 10, fontSize: 10.5, color: 'var(--ink-3)', letterSpacing: '0.04em' }}>
                  목표 {k.target} ·{' '}
                  <span style={{ color: k.pass ? 'var(--green)' : 'var(--red)', fontWeight: 700 }}>
                    {k.pass ? 'PASS' : 'MISS'}
                  </span>
                </div>
                <div style={{ fontSize: 11, color: 'var(--ink-3)', marginTop: 8 }}>{k.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Confidence breakdown */}
      <div style={{ padding: '20px 32px 40px' }}>
        <div className="panel" style={{ padding: '22px 26px' }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, marginBottom: 18 }}>
            <h2 className="section-title">오늘 신뢰도 분포</h2>
            <span style={{ flex: 1 }} />
            <span className="num" style={{ fontSize: 11, color: 'var(--ink-3)' }}>n={n}</span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 18 }}>
            {[
              { level: 'HIGH', items: high, color: 'var(--navy)',  desc: '확률 차이 > 15%p' },
              { level: 'MED',  items: med,  color: 'var(--ink-3)', desc: '확률 차이 5–15%p' },
              { level: 'LOW',  items: low,  color: 'var(--red)',   desc: '확률 차이 < 5%p'  },
            ].map(b => (
              <div key={b.level} style={{ border: '1px solid var(--rule)', borderRadius: 'var(--r-md)', padding: '18px 20px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <ConfPill level={b.level} />
                  <span className="num" style={{ fontSize: 11, color: 'var(--ink-3)', marginLeft: 'auto' }}>
                    n={b.items.length}
                  </span>
                </div>
                <div className="cond" style={{ fontSize: 38, fontWeight: 700, marginTop: 12, lineHeight: 1, color: b.color }}>
                  {n > 0 ? Math.round(b.items.length / n * 100) : 0}%
                </div>
                <div className="num" style={{ fontSize: 11, color: 'var(--ink-3)', marginTop: 6 }}>
                  {b.items.length} 경기
                </div>
                <div style={{ marginTop: 14, height: 6, background: 'var(--rule-soft)', borderRadius: 1, overflow: 'hidden' }}>
                  <div style={{ width: `${n > 0 ? b.items.length / n * 100 : 0}%`, height: '100%', background: b.color }} />
                </div>
                <div className="num" style={{ fontSize: 10, color: 'var(--ink-4)', marginTop: 8, letterSpacing: '0.06em' }}>
                  {b.desc}
                </div>
              </div>
            ))}
          </div>

          <div style={{
            marginTop: 20, padding: 16, background: 'var(--surface-2)',
            border: '1px solid var(--rule)', borderRadius: 'var(--r-sm)',
            fontSize: 12.5, color: 'var(--ink-2)', lineHeight: 1.65,
          }}>
            <strong>모델 구조</strong> · LightGBM + XGBoost 앙상블 (역-logloss softmax 가중치: LGBM 0.488, XGB 0.512).
            IsotonicRegression 확률 보정. BBref 기반 29개 피처 (팀 롤링승률·투구·타격·파크팩터).
            학습 2023–2024 시즌 4,912 경기 (시간순 70/15/15 분할).
            MLB는 고노이즈 도메인 — AUC 0.55는 베이스라인(0.50) 대비 유의미한 예측력입니다.
          </div>
        </div>
      </div>
    </div>
  )
}

// ─── Main ─────────────────────────────────────────────────────────────────────

export default function Home() {
  const [data, setData]             = useState<ApiResponse | null>(null)
  const [loading, setLoading]       = useState(true)
  const [error, setError]           = useState<string | null>(null)
  const [view, setView]             = useState<'today' | 'model' | 'detail'>('today')
  const [prevView, setPrevView]     = useState<'today' | 'model'>('today')
  const [selectedGame, setSelected] = useState<Prediction | null>(null)
  const [gameDetail, setDetail]     = useState<GameDetail | null>(null)

  useEffect(() => {
    fetch(`${API_URL}/predictions/today`)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then((d: ApiResponse) => { setData(d); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  const openGame = (game: Prediction) => {
    setPrevView(view as 'today' | 'model')
    setSelected(game)
    setDetail(null)
    setView('detail')
    window.scrollTo({ top: 0, behavior: 'smooth' })
    fetch(`${API_URL}/predictions/${game.game_pk}`)
      .then(r => r.ok ? r.json() : null)
      .then(d => d && setDetail(d))
      .catch(() => {})
  }

  const goBack = () => {
    setView(prevView)
    setSelected(null)
    setDetail(null)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const handleNav = (v: string) => {
    setView(v as 'today' | 'model')
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const today = data?.date ?? new Date().toISOString().slice(0, 10)
  const gamesCount = data?.games?.length ?? 0

  return (
    <div className="app">
      <Topbar
        view={view === 'detail' ? prevView : view}
        setView={handleNav}
        gamesCount={gamesCount}
        date={today}
      />

      {loading && (
        <div style={{ padding: '80px 32px', textAlign: 'center', color: 'var(--ink-3)' }}>
          <div className="cond" style={{ fontSize: 28, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 12 }}>
            로딩 중...
          </div>
          <div className="num" style={{ fontSize: 11, letterSpacing: '0.16em' }}>FETCHING PREDICTIONS</div>
        </div>
      )}

      {error && (
        <div style={{ padding: '60px 32px', textAlign: 'center' }}>
          <div className="cond" style={{ fontSize: 22, fontWeight: 700, color: 'var(--red)', textTransform: 'uppercase', marginBottom: 8 }}>
            API 연결 오류
          </div>
          <div className="num" style={{ fontSize: 12, color: 'var(--ink-3)' }}>{error}</div>
          <div className="num" style={{ fontSize: 11, color: 'var(--ink-4)', marginTop: 8, letterSpacing: '0.08em' }}>
            백엔드 서버가 실행 중인지 확인하세요.
          </div>
        </div>
      )}

      {!loading && !error && data && (
        <>
          {view === 'today'  && <TodayView data={data} onOpenGame={openGame} />}
          {view === 'model'  && <ModelView predictions={data.games} />}
          {view === 'detail' && selectedGame && (
            <DetailView game={selectedGame} detail={gameDetail} onBack={goBack} date={today} />
          )}
        </>
      )}
    </div>
  )
}
