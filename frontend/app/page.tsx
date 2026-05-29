'use client'
import { useEffect, useState, useCallback, useRef } from 'react'

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

/* ── Team data ── */
const TEAMS: Record<string, { city: string; name: string; primary: string }> = {
  LAD:{city:"Los Angeles",name:"Dodgers",primary:"#1E4D8C"},NYY:{city:"New York",name:"Yankees",primary:"#0E1A2B"},
  BOS:{city:"Boston",name:"Red Sox",primary:"#9E2A2B"},HOU:{city:"Houston",name:"Astros",primary:"#1F3D5C"},
  ATL:{city:"Atlanta",name:"Braves",primary:"#9E2A2B"},PHI:{city:"Philadelphia",name:"Phillies",primary:"#A8323A"},
  SD:{city:"San Diego",name:"Padres",primary:"#3D2E1F"},SF:{city:"San Francisco",name:"Giants",primary:"#3B3B3B"},
  CHC:{city:"Chicago",name:"Cubs",primary:"#1E4D8C"},STL:{city:"St. Louis",name:"Cardinals",primary:"#9E2A2B"},
  MIL:{city:"Milwaukee",name:"Brewers",primary:"#1F3D2E"},TOR:{city:"Toronto",name:"Blue Jays",primary:"#1E4D8C"},
  SEA:{city:"Seattle",name:"Mariners",primary:"#1F4D5C"},TEX:{city:"Texas",name:"Rangers",primary:"#1E4D8C"},
  BAL:{city:"Baltimore",name:"Orioles",primary:"#C85A00"},TB:{city:"Tampa Bay",name:"Rays",primary:"#1F3D5C"},
  CLE:{city:"Cleveland",name:"Guardians",primary:"#9E2A2B"},DET:{city:"Detroit",name:"Tigers",primary:"#0E2A4D"},
  AZ:{city:"Arizona",name:"D-backs",primary:"#6B2043"},NYM:{city:"New York",name:"Mets",primary:"#1E4D8C"},
  MIN:{city:"Minnesota",name:"Twins",primary:"#0E2A4D"},KC:{city:"Kansas City",name:"Royals",primary:"#14365E"},
  CWS:{city:"Chicago",name:"White Sox",primary:"#1F1F1F"},COL:{city:"Colorado",name:"Rockies",primary:"#3D2E5C"},
  WSH:{city:"Washington",name:"Nationals",primary:"#9E2A2B"},PIT:{city:"Pittsburgh",name:"Pirates",primary:"#2B1A00"},
  MIA:{city:"Miami",name:"Marlins",primary:"#1F4D5C"},OAK:{city:"Oakland",name:"Athletics",primary:"#1F3D2E"},
  ATH:{city:"Athletics",name:"Athletics",primary:"#1F3D2E"},LAA:{city:"Los Angeles",name:"Angels",primary:"#9E2A2B"},
  CIN:{city:"Cincinnati",name:"Reds",primary:"#9E2A2B"},
}

/* ── Types ── */
interface ShapFeature { feature: string; value: number; shap_value: number }
interface TodayGame {
  game_pk:number; home_team:string; away_team:string
  home_win_prob:number; away_win_prob:number
  confidence:'HIGH'|'MED'|'LOW'; reasoning:string; top5_features:ShapFeature[]
  model_version?:string; game_datetime?:string|null
}
interface TodayData { date:string; games:TodayGame[] }
interface HistRow {
  game_pk:number; date:string; game_datetime:string|null
  home_team:string; away_team:string; home_score:number|null; away_score:number|null
  status:string; home_win_prob:number; away_win_prob:number
  confidence:string; is_correct:number|null; pick_team:string; pick_prob:number
}
interface HistData { days:number; total:number; graded:number; correct:number; accuracy:number|null; brier:number|null; rows:HistRow[] }
interface LiveData {
  game_pk:number; status:string; detailed_state:string
  current_inning:number|null; inning_state:string
  balls:number; strikes:number; outs:number
  runs:{home:number;away:number}; hits:{home:number;away:number}; errors:{home:number;away:number}
  runners:{first:boolean;second:boolean;third:boolean}
  home_team:string; away_team:string; home_name:string; away_name:string; venue:string
  pitchers?:{home_probable:string;away_probable:string;current:string;winner:string;loser:string}
  live_home_prob?:number; play_event?:string; is_new_play?:boolean
}
interface UserInfo { username:string; user_id:number; total:number; graded:number; correct:number; accuracy:number|null; streak:number; by_conf:Record<string,{n:number;correct:number;acc:number|null}> }
interface UserPickItem { id:number; game_pk:number; game_date:string; home_team:string; away_team:string; pick_team:string; pick_prob:number|null; confidence:string; is_correct:number|null }
interface Standing { team_id:number; team_name:string; abbr:string; division:string; wins:number; losses:number; win_pct:number; gb:string; streak:string; home_w:number|string; home_l:number|string; away_w:number|string; away_l:number|string; l10_w:number|string; l10_l:number|string }
interface StandingsData { AL:Standing[]; NL:Standing[] }
interface CalDay { date:string; total:number; correct:number; graded:number; accuracy:number|null }
interface ArchiveSummary { date:string; total:number; graded:number; correct:number; accuracy:number|null; high_med_accuracy:number|null; games:any[] }
interface MlbScheduleGame {
  game_pk:number
  game_datetime:string|null
  status:string
  detailed_state?:string
  home_team:string
  away_team:string
  home_name:string
  away_name:string
  venue?:string
  home_score:number|null
  away_score:number|null
  home_probable_pitcher?:string|null
  away_probable_pitcher?:string|null
  win_pitcher?:string|null
  loss_pitcher?:string|null
}

/* ── Auth context ── */
interface AuthState { token:string|null; username:string|null; userId:number|null }

/* ── Shared components ── */
function TM({ code, size='md' }:{code:string;size?:'xs'|'sm'|'md'|'lg'|'xl'}) {
  const bg = TEAMS[code]?.primary ?? '#334155'
  const [logoOk, setLogoOk] = useState(true)
  const logoSrc = `/logos/${code}.png`

  return (
    <span className={`tm tm-${size} ${logoOk ? 'tm-logo-ok' : ''}`} style={{backgroundColor:bg, borderColor:bg}}>
      {logoOk ? (
        <img
          className="tm-img"
          src={logoSrc}
          alt={code}
          onError={() => setLogoOk(false)}
        />
      ) : (
        code
      )}
    </span>
  )
}
function Conf({level}:{level:string}) { return <span className={`conf conf-${level}`}>{level}</span> }
function RC({v}:{v:number|null}) {
  if(v===null) return <span className="rc rc-pend">—</span>
  return v===1 ? <span className="rc rc-hit">✓</span> : <span className="rc rc-miss">✗</span>
}
function PBar({home,h=7}:{home:number;h?:number}) {
  return <div className="prob-bar" style={{height:h,borderRadius:1}}><div className="a" style={{width:`${(1-home)*100}%`}}/><div className="h" style={{width:`${home*100}%`}}/></div>
}
function Spinner() { return <div style={{display:'flex',justifyContent:'center',padding:60}}><div className="spin"/></div> }
function Loading() { return <Spinner/> }
function ProbBar({home,h=7}:{home:number;h?:number}) {
  return <div className="prob-bar" style={{height:h,borderRadius:1}}><div className="a" style={{width:`${(1-home)*100}%`}}/><div className="h" style={{width:`${home*100}%`}}/></div>
}
function ResultChip({v}:{v:number|null}) {
  if(v===null) return <span className="rc rc-pend">—</span>
  return v===1?<span className="rc rc-hit">○ HIT</span>:<span className="rc rc-miss">✕ MISS</span>
}
function EmptyBox({msg,sub}:{msg:string;sub?:string}) {
  return <div style={{padding:'60px 40px',textAlign:'center',background:'var(--surface)',border:'1px solid var(--rule)',borderRadius:'var(--r-lg)'}}><div style={{fontSize:36,marginBottom:12}}>⚾</div><div className="cond" style={{fontSize:20,fontWeight:700,color:'var(--ink-2)',marginBottom:6}}>{msg}</div>{sub&&<div className="num" style={{fontSize:11,color:'var(--ink-4)'}}>{sub}</div>}</div>
}

const TEAM_CODE_ALIASES: Record<string,string> = {
  ARI:'AZ', AZ:'AZ', ATH:'ATH', OAK:'ATH', CWS:'CWS', CHW:'CWS', WSH:'WSH', WAS:'WSH', SF:'SF', SFG:'SF', SD:'SD', SDP:'SD', TB:'TB', TBR:'TB', KC:'KC', KCR:'KC', NYY:'NYY', NYM:'NYM', LAA:'LAA', LAD:'LAD', TOR:'TOR', TEX:'TEX', SEA:'SEA', STL:'STL', CHC:'CHC', CIN:'CIN', CLE:'CLE', COL:'COL', DET:'DET', HOU:'HOU', MIA:'MIA', MIL:'MIL', MIN:'MIN', ATL:'ATL', BAL:'BAL', BOS:'BOS', PHI:'PHI', PIT:'PIT'
}
function normalizeTeamCode(code?: string|null) {
  const raw = String(code || '').trim().toUpperCase()
  return TEAM_CODE_ALIASES[raw] || raw
}
function getKstDateStr(): string {
  return new Date().toLocaleDateString('en-CA', { timeZone: 'Asia/Seoul' })
}
function getLastName(fullName?: string|null): string {
  if (!fullName) return ''
  const parts = fullName.trim().split(' ')
  return parts[parts.length - 1]
}
function inningLabel(ld?: LiveData|null) {
  if(!ld || !ld.current_inning) return ''
  const state = String(ld.inning_state || ld.detailed_state || '').toLowerCase()
  if(state.includes('top') || state.includes('초')) return `${ld.current_inning}회초`
  if(state.includes('bottom') || state.includes('말')) return `${ld.current_inning}회말`
  if(state.includes('middle')) return `${ld.current_inning}회초 종료`
  if(state.includes('end')) return `${ld.current_inning}회말 종료`
  return `${ld.current_inning}회`
}
function statusKo(status?: string|null, detailed?: string|null) {
  const s = String(status || detailed || '')
  if(/final/i.test(s)) return '종료'
  if(/live|in progress|warmup/i.test(s)) return '진행 중'
  if(/preview|scheduled|pre-game/i.test(s)) return '예정'
  return s || '예정'
}
function formatKstTime(value?: string | null) {
  if (!value) return '—'

  const raw = String(value).trim()
  if (!raw) return '—'

  const normalized = raw.includes('T') ? raw : raw.replace(' ', 'T')

  // MLB API 시간은 UTC 기준인데, DB나 프론트로 오면서 Z가 빠지면
  // 브라우저가 한국 로컬시간으로 착각해서 01:15처럼 그대로 표시할 수 있음.
  const withTimezone = /(?:Z|[+-]\d{2}:?\d{2})$/i.test(normalized)
    ? normalized
    : `${normalized}Z`

  const dt = new Date(withTimezone)
  if (Number.isNaN(dt.getTime())) return '—'

  return dt.toLocaleTimeString('ko-KR', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
    timeZone: 'Asia/Seoul',
  })
}
async function fetchMlbScheduleByDate(kstDate:string): Promise<MlbScheduleGame[]> {
  // MLB game dates are US Eastern time. KST = UTC+9, ET ≈ UTC-4,
  // so even a 1pm ET game is 2am KST next day → always fetch US date = KST date - 1 day.
  const [ky,km,kd] = kstDate.split('-').map(Number)
  const usDate = new Date(ky, km-1, kd-1)
  const date = `${usDate.getFullYear()}-${String(usDate.getMonth()+1).padStart(2,'0')}-${String(usDate.getDate()).padStart(2,'0')}`
  const url = `https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=${date}&hydrate=team,linescore,decisions,probablePitcher`
  const r = await fetch(url)
  if(!r.ok) throw new Error('MLB schedule fetch failed')
  const d = await r.json()
  const games = d?.dates?.[0]?.games || []
  return games.map((g:any) => {
    const away = g.teams?.away || {}
    const home = g.teams?.home || {}
    const awayTeam = away.team || {}
    const homeTeam = home.team || {}
    return {
      game_pk: g.gamePk,
      game_datetime: g.gameDate || null,
      status: g.status?.abstractGameState || g.status?.codedGameState || 'Preview',
      detailed_state: g.status?.detailedState || '',
      away_team: normalizeTeamCode(awayTeam.abbreviation || awayTeam.teamCode || awayTeam.fileCode),
      home_team: normalizeTeamCode(homeTeam.abbreviation || homeTeam.teamCode || homeTeam.fileCode),
      away_name: awayTeam.name || '',
      home_name: homeTeam.name || '',
      venue: g.venue?.name || '',
      away_score: away.score ?? null,
      home_score: home.score ?? null,
      home_probable_pitcher: home.probablePitcher?.fullName || null,
      away_probable_pitcher: away.probablePitcher?.fullName || null,
      win_pitcher: g.decisions?.winner?.fullName || null,
      loss_pitcher: g.decisions?.loser?.fullName || null,
    }
  })
}

/* ── Diamond (baseball bases) ── */
function Diamond({runners}:{runners:{first:boolean;second:boolean;third:boolean}}) {
  return (
    <div className="diamond-wrap">
      <div className={`diamond-base db-2nd ${runners.second?'on':'off'}`}/>
      <div className={`diamond-base db-1st ${runners.first?'on':'off'}`}/>
      <div className={`diamond-base db-3rd ${runners.third?'on':'off'}`}/>
      <div className="diamond-base db-home off"/>
    </div>
  )
}

/* ── Auth Modal ── */
function AuthModal({mode,onClose,onSuccess}:{mode:'login'|'register';onClose:()=>void;onSuccess:(token:string,username:string,userId:number)=>void}) {
  const [m,setM] = useState(mode)
  const [username,setUsername] = useState('')
  const [email,setEmail] = useState('')
  const [password,setPassword] = useState('')
  const [err,setErr] = useState('')
  const [loading,setLoading] = useState(false)

  const submit = async () => {
    setErr(''); setLoading(true)
    try {
      const body = m==='login' ? {username,password} : {username,email,password}
      const r = await fetch(`${API}/users/${m==='login'?'login':'register'}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
      const d = await r.json()
      if(!r.ok) { setErr(d.detail||'오류가 발생했습니다'); setLoading(false); return }
      onSuccess(d.token, d.username, d.user_id)
    } catch { setErr('서버 연결 오류'); setLoading(false) }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={e=>e.stopPropagation()}>
        <div className="modal-top">
          <div className="modal-title">{m==='login'?'로그인':'회원가입'}</div>
          <div className="modal-sub">SIMMLB · MLB AI 승부 예측</div>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>
        <div className="modal-body">
          <div className="field"><label>닉네임</label><input value={username} onChange={e=>setUsername(e.target.value)} placeholder="닉네임 입력" onKeyDown={e=>e.key==='Enter'&&submit()}/></div>
          {m==='register'&&<div className="field"><label>이메일</label><input type="email" value={email} onChange={e=>setEmail(e.target.value)} placeholder="이메일 입력"/></div>}
          <div className="field"><label>비밀번호</label><input type="password" value={password} onChange={e=>setPassword(e.target.value)} placeholder="비밀번호 입력" onKeyDown={e=>e.key==='Enter'&&submit()}/></div>
          {err&&<div className="err-msg">{err}</div>}
          <button className="btn-full red" style={{marginTop:16}} disabled={loading} onClick={submit}>{loading?'처리 중...':(m==='login'?'로그인':'가입하기')}</button>
          <div className="modal-switch">
            {m==='login'?<>계정이 없으신가요? <button onClick={()=>setM('register')}>회원가입</button></>:<>이미 계정이 있으신가요? <button onClick={()=>setM('login')}>로그인</button></>}
          </div>
        </div>
      </div>
    </div>
  )
}

/* ── Topbar ── */
function Topbar({view,setView,gameCount,auth,onAuthClick,onLogout}:{view:string;setView:(v:string)=>void;gameCount:number;auth:AuthState;onAuthClick:(m:'login'|'register')=>void;onLogout:()=>void}) {
  const today=getKstDateStr()
  const wd=['SUN','MON','TUE','WED','THU','FRI','SAT'][new Date(today+'T12:00:00').getDay()]
  return (
    <header className="topbar">
      <div className="topbar-brand" style={{cursor:'pointer'}} onClick={()=>setView('today')}>
        <div className="brand-logo">
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
            <circle cx="10" cy="10" r="8.5" stroke="#fff" strokeWidth="1.4"/>
            <path d="M3.5 10 Q10 4.5 16.5 10 Q10 15.5 3.5 10Z" fill="#fff" opacity="0.9"/>
            <circle cx="10" cy="10" r="2" fill="var(--red)"/>
          </svg>
        </div>
        <div><div className="brand-name">SIMMLB</div><div className="brand-tag">MLB AI · 모델 성능 공개</div></div>
      </div>
      <nav className="topbar-nav">
        {[
          {id:'today',label:'오늘 예측',badge:gameCount},
          {id:'live',label:'라이브',badge:null},
          {id:'schedule',label:'경기 일정',badge:null},
          {id:'standings',label:'순위표',badge:null},
          {id:'model',label:'모델 성능',badge:null},
          ...(auth.token?[{id:'my',label:'나의 픽',badge:null}]:[]),
        ].map(it=>(
          <button key={it.id} className={`nav-item ${view===it.id?'active':''}`} onClick={()=>setView(it.id)}>
            {it.label}{it.badge!=null&&<span className="num-badge">{it.badge}</span>}
          </button>
        ))}
      </nav>
      <div className="topbar-right">
        <span className="live-dot"/><span className="topbar-date">{today} · {wd}</span>
        {auth.token ? (
          <div style={{display:'flex',alignItems:'center',gap:8}}>
            <button className="user-chip" onClick={()=>setView('my')}>
              <span className="user-avatar">{auth.username?.[0]?.toUpperCase()}</span>
              {auth.username}
            </button>
            <button className="btn-auth" onClick={onLogout}>로그아웃</button>
          </div>
        ) : (
          <div style={{display:'flex',gap:6}}>
            <button className="btn-auth" onClick={()=>onAuthClick('login')}>로그인</button>
            <button className="btn-auth primary" onClick={()=>onAuthClick('register')}>가입</button>
          </div>
        )}
      </div>
    </header>
  )
}

/* ════════════════════════════════════
   SCREEN: TODAY
════════════════════════════════════ */
function ScreenToday({auth}:{auth:AuthState}) {
  const [data,setData]=useState<TodayData|null>(null)
  const [loading,setLoading]=useState(true)
  const [err,setErr]=useState<string|null>(null)
  const [filter,setFilter]=useState<'ALL'|'HIGH'|'MED'|'LOW'>('ALL')
  const [sel,setSel]=useState<TodayGame|null>(null)
  const [myPicks,setMyPicks]=useState<Record<number,string>>({})
  const [pickLoading,setPickLoading]=useState<Record<number,boolean>>({})

  useEffect(()=>{
    fetch(`${API}/predictions/today`).then(r=>r.json()).then((d:TodayData)=>{setData(d);setLoading(false)}).catch(e=>{setErr(e.message);setLoading(false)})
  },[])

  useEffect(()=>{
    if(!auth.token) return
    fetch(`${API}/users/me/picks`,{headers:{'Authorization':`Bearer ${auth.token}`}}).then(r=>r.json()).then(d=>{
      const map:Record<number,string>={}
      d.picks?.forEach((p:any)=>{ map[p.game_pk]=p.pick_team })
      setMyPicks(map)
    }).catch(()=>{})
  },[auth.token])

  const addPick = async (game_pk:number, pick_team:string) => {
    if(!auth.token) return
    setPickLoading(p=>({...p,[game_pk]:true}))
    try {
      const r = await fetch(`${API}/users/me/picks`,{method:'POST',headers:{'Content-Type':'application/json','Authorization':`Bearer ${auth.token}`},body:JSON.stringify({game_pk,pick_team})})
      if(r.ok) setMyPicks(p=>({...p,[game_pk]:pick_team}))
    } finally { setPickLoading(p=>({...p,[game_pk]:false})) }
  }
  const removePick = async (game_pk:number) => {
    if(!auth.token) return
    setPickLoading(p=>({...p,[game_pk]:true}))
    try {
      const r = await fetch(`${API}/users/me/picks/${game_pk}`,{method:'DELETE',headers:{'Authorization':`Bearer ${auth.token}`}})
      if(r.ok) setMyPicks(p=>{const n={...p};delete n[game_pk];return n})
    } finally { setPickLoading(p=>({...p,[game_pk]:false})) }
  }

  if(sel) return <DetailToday game={sel} onBack={()=>setSel(null)} auth={auth} myPick={myPicks[sel.game_pk]||null} onPick={addPick} onUnpick={removePick} pickLoading={!!pickLoading[sel.game_pk]}/>

  const games=data?.games??[]
  const list=filter==='ALL'?games:games.filter(g=>g.confidence===filter)
  const sorted=[...list].sort((a,b)=>Math.abs(b.home_win_prob-.5)-Math.abs(a.home_win_prob-.5))

  return (
    <div className="view-enter">
      <div className="subhead">
        <div><div className="kicker">TODAY PREDICTIONS · {data?.date??'—'}</div><h1 className="page-title">오늘의 <span className="red">예측</span></h1></div>
        {!loading&&!err&&(
          <div className="hero-kpis">
            <div className="hero-kpi"><div className="hk-lbl">총 경기</div><div className="hk-val cond">{games.length}</div><div className="hk-sub">TODAY</div></div>
            <div className="hero-kpi"><div className="hk-lbl">HIGH</div><div className="hk-val cond green">{games.filter(g=>g.confidence==='HIGH').length}</div><div className="hk-sub">GAMES</div></div>
            <div className="hero-kpi"><div className="hk-lbl">최고 엣지</div><div className="hk-val cond red">{games.length>0?`${(Math.max(...games.map(g=>Math.abs(g.home_win_prob-.5)))*100).toFixed(1)}%`:'—'}</div><div className="hk-sub">EDGE</div></div>
            <div className="hero-kpi"><div className="hk-lbl">내 픽</div><div className="hk-val cond amber">{auth.token?Object.keys(myPicks).length:'—'}</div><div className="hk-sub">{auth.token?'PICKED':'LOGIN'}</div></div>
          </div>
        )}
      </div>
      {!loading&&!err&&games.length>0&&(
        <div className="filterbar">
          <span className="lbl" style={{marginRight:4}}>신뢰도</span>
          {(['ALL','HIGH','MED','LOW'] as const).map(f=>(
            <button key={f} className={`chip ${filter===f?'active':''}`} onClick={()=>setFilter(f)}>
              {f==='ALL'?'전체':f}<span className="ct">{f==='ALL'?games.length:games.filter(g=>g.confidence===f).length}</span>
            </button>
          ))}
          <span style={{marginLeft:'auto',fontFamily:'var(--f-mono)',fontSize:11,color:'var(--ink-3)'}}>엣지 높은 순 · {sorted.length}경기</span>
        </div>
      )}
      {loading&&<Spinner/>}
      {err&&<div style={{margin:'32px',padding:28,background:'var(--surface)',border:'1px solid var(--rule)',borderRadius:'var(--r-lg)',textAlign:'center'}}><div className="cond" style={{fontSize:18,fontWeight:700,color:'var(--red)',marginBottom:6}}>API 연결 오류</div><div className="num" style={{fontSize:11,color:'var(--ink-3)'}}>{err}</div></div>}
      {!loading&&!err&&games.length===0&&<div style={{padding:'24px 32px'}}><EmptyBox msg="오늘 예측 데이터가 없습니다" sub="python scripts/run_inference_v2.py 실행 후 새로고침"/></div>}
      {!loading&&!err&&sorted.length>0&&(
        <div className="today-grid">
          {sorted.map(g=>(
            <TodayCard key={g.game_pk} g={g} onClick={()=>setSel(g)} auth={auth}
              myPick={myPicks[g.game_pk]||null} onPick={addPick} onUnpick={removePick} pickLoading={!!pickLoading[g.game_pk]}/>
          ))}
        </div>
      )}
    </div>
  )
}

function TodayCard({g,onClick,auth,myPick,onPick,onUnpick,pickLoading}:{g:TodayGame;onClick:()=>void;auth:AuthState;myPick:string|null;onPick:(pk:number,team:string)=>void;onUnpick:(pk:number)=>void;pickLoading:boolean}) {
  const pickHome=g.home_win_prob>=.5
  const pick=pickHome?g.home_team:g.away_team
  const pct=(Math.max(g.home_win_prob,g.away_win_prob)*100).toFixed(1)
  const edge=(Math.abs(g.home_win_prob-.5)*100).toFixed(1)
  const pickedHome=myPick===g.home_team
  const pickedAway=myPick===g.away_team

  const handlePick=(team:string,e:React.MouseEvent)=>{
    e.stopPropagation()
    if(myPick===team) onUnpick(g.game_pk)
    else onPick(g.game_pk,team)
  }

  return (
    <div className="gcard">
      <div style={{display:'grid',gridTemplateColumns:'1fr auto 1fr',alignItems:'center',gap:10,padding:'16px 18px 12px'}} onClick={onClick}>
        <div style={{display:'flex',alignItems:'center',gap:9}}>
          <TM code={g.away_team} size="md"/>
          <div><div className="lbl">AWAY</div><div className="cond" style={{fontSize:19,fontWeight:700,textTransform:'uppercase',lineHeight:1.1}}>{g.away_team}</div><div className="cond" style={{fontSize:11,color:'var(--ink-3)'}}>{TEAMS[g.away_team]?.name}</div></div>
        </div>
        <div style={{textAlign:'center'}}><div className="cond" style={{fontSize:24,color:'var(--ink-4)',fontWeight:300,lineHeight:1}}>vs</div></div>
        <div style={{display:'flex',alignItems:'center',gap:9,flexDirection:'row-reverse'}}>
          <TM code={g.home_team} size="md"/>
          <div style={{textAlign:'right'}}><div className="lbl">HOME</div><div className="cond" style={{fontSize:19,fontWeight:700,textTransform:'uppercase',lineHeight:1.1}}>{g.home_team}</div><div className="cond" style={{fontSize:11,color:'var(--ink-3)'}}>{TEAMS[g.home_team]?.name}</div></div>
        </div>
      </div>
      <div style={{padding:'0 18px 10px'}} onClick={onClick}>
        <div style={{display:'flex',justifyContent:'space-between',marginBottom:5}}>
          <span className="num" style={{fontSize:12,fontWeight:700,color:'var(--red)'}}>{(g.away_win_prob*100).toFixed(1)}%</span>
          <span className="num" style={{fontSize:10,color:'var(--ink-4)'}}>EDGE +{edge}%p</span>
          <span className="num" style={{fontSize:12,fontWeight:700,color:'var(--navy)'}}>{(g.home_win_prob*100).toFixed(1)}%</span>
        </div>
        <PBar home={g.home_win_prob}/>
      </div>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',padding:'10px 18px 14px',background:'var(--surface-2)',borderTop:'1px solid var(--rule-soft)'}}>
        <div onClick={onClick} style={{cursor:'pointer'}}>
          <div className="lbl" style={{marginBottom:3}}>모델 픽</div>
          <div className="cond" style={{fontSize:19,fontWeight:700,textTransform:'uppercase',color:pickHome?'var(--navy)':'var(--red)'}}>
            {pick} <span style={{fontWeight:400,color:'var(--ink-3)',fontSize:14}}>{pct}%</span>
          </div>
        </div>
        <div style={{display:'flex',alignItems:'center',gap:6}}>
          {auth.token&&(
            <>
              <button className={`pick-btn ${pickedAway?'picked-away':''}`} disabled={pickLoading} onClick={e=>handlePick(g.away_team,e)}>
                {pickedAway?'✓ ':''}{g.away_team}
              </button>
              <button className={`pick-btn ${pickedHome?'picked-home':''}`} disabled={pickLoading} onClick={e=>handlePick(g.home_team,e)}>
                {pickedHome?'✓ ':''}{g.home_team}
              </button>
            </>
          )}
          <Conf level={g.confidence}/>
          <svg className="row-chevron" width="13" height="13" viewBox="0 0 14 14" fill="none" onClick={onClick}><path d="M5 2L10 7L5 12" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/></svg>
        </div>
      </div>
    </div>
  )
}

/* ════════════════════════════════════
   SCREEN: TODAY DETAIL
════════════════════════════════════ */
function DetailToday({game:g,onBack,auth,myPick,onPick,onUnpick,pickLoading}:{game:TodayGame;onBack:()=>void;auth:AuthState;myPick:string|null;onPick:(pk:number,team:string)=>void;onUnpick:(pk:number)=>void;pickLoading:boolean}) {
  const [tab,setTab]=useState<'ai'|'shap'|'live'>('ai')
  const pickHome=g.home_win_prob>=.5
  const pick=pickHome?g.home_team:g.away_team
  const pct=(Math.max(g.home_win_prob,g.away_win_prob)*100).toFixed(1)
  const edge=(Math.abs(g.home_win_prob-.5)*100).toFixed(1)
  const aT=TEAMS[g.away_team],hT=TEAMS[g.home_team]

  const handlePick=(team:string)=>{
    if(myPick===team) onUnpick(g.game_pk)
    else onPick(g.game_pk,team)
  }

  return (
    <div className="view-enter">
      <div className="detail-back">
        <button className="btn ghost" onClick={onBack}>← 오늘 예측</button>
        <span style={{color:'var(--rule)'}}>·</span>
        <span className="num" style={{fontSize:11,color:'var(--ink-3)',letterSpacing:'.06em'}}>{g.away_team} vs {g.home_team}</span>
        <span style={{flex:1}}/>
        <Conf level={g.confidence}/>
      </div>
      <div style={{background:'var(--navy)',color:'#fff',padding:'26px 32px 22px',borderBottom:'1px solid rgba(255,255,255,.06)'}}>
        <div className="lbl on-dark" style={{marginBottom:16}}>MATCHUP ANALYSIS · 경기 예측</div>
        <div style={{display:'grid',gridTemplateColumns:'1fr auto 1fr',gap:22,alignItems:'center'}}>
          <div style={{display:'flex',gap:14,alignItems:'center'}}>
            <TM code={g.away_team} size="xl"/>
            <div><div className="lbl on-dark">AWAY</div><div className="cond" style={{fontSize:12,color:'#8FA3C0',marginTop:2}}>{aT?.city}</div><div className="cond" style={{fontSize:26,fontWeight:700,textTransform:'uppercase',lineHeight:1.05}}>{aT?.name??g.away_team}</div><div className="num" style={{fontSize:30,fontWeight:600,color:'var(--red)',marginTop:8}}>{(g.away_win_prob*100).toFixed(1)}%</div></div>
          </div>
          <div style={{textAlign:'center'}}><div className="cond" style={{fontSize:44,color:'rgba(255,255,255,.2)',fontWeight:300,lineHeight:1}}>vs</div><div className="num" style={{fontSize:9,color:'#8FA3C0',marginTop:6,letterSpacing:'.1em'}}>PREVIEW</div></div>
          <div style={{display:'flex',gap:14,alignItems:'center',flexDirection:'row-reverse',textAlign:'right'}}>
            <TM code={g.home_team} size="xl"/>
            <div><div className="lbl on-dark">HOME</div><div className="cond" style={{fontSize:12,color:'#8FA3C0',marginTop:2}}>{hT?.city}</div><div className="cond" style={{fontSize:26,fontWeight:700,textTransform:'uppercase',lineHeight:1.05}}>{hT?.name??g.home_team}</div><div className="num" style={{fontSize:30,fontWeight:600,color:'#5B9BDF',marginTop:8}}>{(g.home_win_prob*100).toFixed(1)}%</div></div>
          </div>
        </div>
        <div style={{marginTop:22,height:26,borderRadius:2,overflow:'hidden',display:'flex',fontFamily:'var(--f-cond)',fontWeight:700,fontSize:12,letterSpacing:'.05em'}}>
          <div style={{width:`${g.away_win_prob*100}%`,background:'var(--red)',color:'#fff',display:'flex',alignItems:'center',paddingLeft:10}}>{g.away_team} {(g.away_win_prob*100).toFixed(1)}%</div>
          <div style={{width:`${g.home_win_prob*100}%`,background:'#1A4A8A',color:'#fff',display:'flex',alignItems:'center',justifyContent:'flex-end',paddingRight:10}}>{g.home_team} {(g.home_win_prob*100).toFixed(1)}%</div>
        </div>
        <div style={{marginTop:12,padding:'12px 16px',background:'rgba(255,255,255,.06)',border:'1px solid rgba(255,255,255,.1)',borderRadius:'var(--r-md)',display:'flex',justifyContent:'space-between',alignItems:'center'}}>
          <div><div className="lbl on-dark" style={{marginBottom:5}}>모델 픽 · AI RECOMMENDATION</div><div className="cond" style={{fontSize:24,fontWeight:700,textTransform:'uppercase',color:pickHome?'#5B9BDF':'#E05A7A'}}>{pick} <span style={{fontFamily:'var(--f-mono)',fontWeight:400,fontSize:18,color:'rgba(255,255,255,.5)',marginLeft:6}}>승 {pct}%</span></div></div>
          <div style={{display:'flex',flexDirection:'column',alignItems:'flex-end',gap:8}}>
            <Conf level={g.confidence}/>
            {auth.token&&(
              <div style={{display:'flex',gap:6}}>
                <button className={`pick-btn ${myPick===g.away_team?'picked-away':''}`} disabled={pickLoading} onClick={()=>handlePick(g.away_team)}>{myPick===g.away_team?'✓ ':''}{g.away_team} 픽</button>
                <button className={`pick-btn ${myPick===g.home_team?'picked-home':''}`} disabled={pickLoading} onClick={()=>handlePick(g.home_team)}>{myPick===g.home_team?'✓ ':''}{g.home_team} 픽</button>
              </div>
            )}
          </div>
        </div>
      </div>
      <div style={{background:'var(--surface)',padding:'0 32px',borderBottom:'1px solid var(--rule)'}}>
        <div className="tabs">
          {[{id:'ai',label:'AI 분석 근거'},{id:'shap',label:'SHAP 기여도'},{id:'live',label:'실시간'}].map(t=>(
            <button key={t.id} className={`tab ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id as any)}>{t.label}</button>
          ))}
        </div>
      </div>
      <div style={{padding:'24px 32px 40px'}}>
        <div className="view-enter" key={tab}>
          {tab==='ai'&&<AiTab g={g}/>}
          {tab==='shap'&&<ShapTab features={g.top5_features}/>}
          {tab==='live'&&<LiveTab gamePk={g.game_pk}/>}
        </div>
      </div>
    </div>
  )
}

function AiTab({g}:{g:TodayGame}) {
  const edge=(Math.abs(g.home_win_prob-.5)*100).toFixed(1)
  return (
    <div style={{display:'grid',gridTemplateColumns:'1fr 280px',gap:18}}>
      <div className="panel" style={{padding:'20px 24px'}}><h2 className="sec-title" style={{marginBottom:14}}>AI 분석 근거</h2><p style={{fontSize:15,lineHeight:1.75,color:'var(--ink)'}}>{g.reasoning||'분석 텍스트가 없습니다.'}</p></div>
      <div style={{display:'flex',flexDirection:'column',gap:14}}>
        <div className="panel" style={{padding:'16px 18px'}}>
          <div className="lbl" style={{marginBottom:10}}>예측 요약</div>
          {[['홈팀 승리 확률',`${(g.home_win_prob*100).toFixed(1)}%`],['원정팀 승리 확률',`${(g.away_win_prob*100).toFixed(1)}%`],['신뢰도',g.confidence],['엣지',`+${edge}%p`],['모델',g.model_version||'v1']].map(([k,v])=>(
            <div key={k} style={{display:'flex',justifyContent:'space-between',padding:'7px 0',borderBottom:'1px solid var(--rule-soft)'}}><span className="num" style={{fontSize:11,color:'var(--ink-3)'}}>{k}</span><span className="num" style={{fontSize:13,fontWeight:600}}>{v}</span></div>
          ))}
        </div>
        <div className="panel dark" style={{padding:'16px 18px'}}>
          <div className="lbl on-dark" style={{marginBottom:10}}>AI ENGINE</div>
          {[['LightGBM','그래디언트 부스팅'],['XGBoost','앙상블 보완'],['Isotonic Cal.','확률 보정'],['SHAP','근거 설명']].map(([n,d])=>(
            <div key={n} style={{display:'flex',justifyContent:'space-between',alignItems:'center',paddingBottom:7,marginBottom:7,borderBottom:'1px solid rgba(255,255,255,.06)'}}><div><div style={{fontFamily:'var(--f-cond)',fontSize:14,fontWeight:600,color:'#fff',textTransform:'uppercase'}}>{n}</div><div className="num" style={{fontSize:10,color:'#8FA3C0'}}>{d}</div></div><svg width="13" height="13" viewBox="0 0 14 14" fill="none"><path d="M2 7H12M8 3L12 7L8 11" stroke="#2ECC71" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/></svg></div>
          ))}
        </div>
      </div>
    </div>
  )
}

function ShapTab({features}:{features:ShapFeature[]}) {
  const max=features.length>0?Math.max(...features.map(f=>Math.abs(f.shap_value))):1
  return (
    <div style={{display:'flex',flexDirection:'column',gap:12}}>
      <div className="panel" style={{padding:'20px 24px'}}>
        <div style={{display:'flex',alignItems:'baseline',gap:12,marginBottom:16}}><h2 className="sec-title">피처 기여도 TOP 5</h2><span style={{flex:1}}/><span className="num" style={{fontSize:10,color:'var(--ink-4)'}}>SHAP value 기준</span></div>
        <div style={{display:'flex',justifyContent:'space-between',marginBottom:10,paddingBottom:8,borderBottom:'1px solid var(--rule-soft)'}}><span className="num" style={{fontSize:9,color:'var(--red)',letterSpacing:'.1em'}}>← AWAY 유리</span><span className="num" style={{fontSize:9,color:'var(--navy)',letterSpacing:'.1em'}}>HOME 유리 →</span></div>
        {features.length===0?<div className="lbl" style={{textAlign:'center',padding:24}}>SHAP 데이터 없음</div>:(
          <div style={{display:'flex',flexDirection:'column',gap:7}}>
            {features.map((f,i)=>{const pct=(Math.abs(f.shap_value)/max)*100;const pos=f.shap_value>0;return(
              <div key={i} className="shap-row">
                <span className="num" style={{fontSize:11,color:'var(--ink-4)',letterSpacing:'.08em'}}>#{String(i+1).padStart(2,'0')}</span>
                <div><div className="cond" style={{fontSize:14,fontWeight:600,textTransform:'uppercase'}}>{f.feature}</div><div className="num" style={{fontSize:10,color:'var(--ink-4)',marginTop:2}}>값: {f.value.toFixed(3)}</div></div>
                <div style={{position:'relative',height:14,background:'var(--rule-soft)',borderRadius:1}}><div style={{position:'absolute',left:'50%',top:-3,bottom:-3,width:1.5,background:'var(--ink-3)'}}/><div style={{position:'absolute',left:pos?'50%':`${50-pct/2}%`,width:`${pct/2}%`,top:2,bottom:2,background:pos?'var(--navy)':'var(--red)',borderRadius:1}}/></div>
                <div style={{textAlign:'right'}}><div className="num" style={{fontSize:13,fontWeight:700,color:pos?'var(--navy)':'var(--red)'}}>{pos?'+':''}{f.shap_value.toFixed(4)}</div><div className="cond" style={{fontSize:9,color:'var(--ink-4)',letterSpacing:'.08em',marginTop:1}}>{pos?'→ HOME':'→ AWAY'}</div></div>
              </div>
            )})}
          </div>
        )}
      </div>
    </div>
  )
}

function LiveTab({gamePk}:{gamePk:number}) {
  const [data,setData]=useState<LiveData|null>(null)
  useEffect(()=>{
    const fetch_=()=>fetch(`${API}/live/game/${gamePk}`).then(r=>r.json()).then(setData).catch(()=>{})
    fetch_()
    const t=setInterval(fetch_,10000)
    return ()=>clearInterval(t)
  },[gamePk])
  if(!data) return <Spinner/>
  if(data.status==='OFFLINE'||data.status==='Preview') return <div className="panel" style={{padding:24,textAlign:'center'}}><div className="cond" style={{fontSize:18,color:'var(--ink-3)'}}>경기 아직 시작 전</div><div className="num" style={{fontSize:11,color:'var(--ink-4)',marginTop:8}}>경기 시작 후 실시간 데이터가 업데이트됩니다</div></div>
  return (
    <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:16}}>
      <div className="panel" style={{padding:'20px 24px'}}>
        <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:16}}>
          <div><span className="live-badge"><span className="live-badge-dot"/>LIVE</span><div className="cond" style={{fontSize:22,fontWeight:700,marginTop:4}}>{inningLabel(data)}</div></div>
          <div style={{textAlign:'right'}}><div className="num" style={{fontSize:11,color:'var(--ink-4)'}}>{data.detailed_state}</div><div className="num" style={{fontSize:11,color:'var(--ink-4)',marginTop:2}}>{data.venue}</div></div>
        </div>
        <div style={{display:'grid',gridTemplateColumns:'1fr auto 1fr',alignItems:'center',gap:10,marginBottom:16}}>
          <div style={{textAlign:'center'}}><div className="lbl">AWAY</div><div className="cond" style={{fontSize:18,fontWeight:700,textTransform:'uppercase'}}>{data.away_team}</div><div className="num" style={{fontSize:36,fontWeight:700,marginTop:4}}>{data.runs.away}</div></div>
          <div className="cond" style={{fontSize:24,color:'var(--ink-4)',fontWeight:300}}>–</div>
          <div style={{textAlign:'center'}}><div className="lbl">HOME</div><div className="cond" style={{fontSize:18,fontWeight:700,textTransform:'uppercase'}}>{data.home_team}</div><div className="num" style={{fontSize:36,fontWeight:700,marginTop:4}}>{data.runs.home}</div></div>
        </div>
        <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:8}}>
          {[['H',data.hits.away,data.hits.home],['E',data.errors.away,data.errors.home]].map(([l,a,h])=>(
            <div key={String(l)} style={{textAlign:'center',padding:'8px 0',background:'var(--surface-2)',borderRadius:'var(--r-sm)'}}><div className="lbl" style={{marginBottom:3}}>{l}</div><div className="num" style={{fontSize:14,fontWeight:600}}>{String(a)} – {String(h)}</div></div>
          ))}
        </div>
      </div>
      <div className="panel" style={{padding:'20px 24px'}}>
        <div className="lbl" style={{marginBottom:16}}>현재 상황</div>
        <div style={{display:'flex',gap:24,alignItems:'center',marginBottom:20}}>
          <Diamond runners={data.runners}/>
          <div>
            <div style={{display:'flex',gap:8,marginBottom:8}}>
              {['B','S','O'].map((l,i)=>{const val=[data.balls,data.strikes,data.outs][i];const max=[3,2,2][i];return(
                <div key={l} style={{display:'flex',alignItems:'center',gap:4}}><span className="num" style={{fontSize:11,color:'var(--ink-3)',width:12}}>{l}</span>{Array.from({length:max+1},(_,j)=><span key={j} style={{width:10,height:10,borderRadius:'50%',background:j<val?'var(--amber)':'var(--rule)',display:'inline-block'}}/>)}</div>
              )})}
            </div>
            <div className="num" style={{fontSize:11,color:'var(--ink-4)'}}>볼 · 스트라이크 · 아웃</div>
          </div>
        </div>
        <div style={{display:'flex',gap:8}}>
          {data.runners.first&&<span className="cond" style={{fontSize:12,fontWeight:600,color:'var(--amber)',background:'rgba(212,144,24,.08)',padding:'4px 10px',borderRadius:'var(--r-sm)'}}>1루 주자</span>}
          {data.runners.second&&<span className="cond" style={{fontSize:12,fontWeight:600,color:'var(--amber)',background:'rgba(212,144,24,.08)',padding:'4px 10px',borderRadius:'var(--r-sm)'}}>2루 주자</span>}
          {data.runners.third&&<span className="cond" style={{fontSize:12,fontWeight:600,color:'var(--amber)',background:'rgba(212,144,24,.08)',padding:'4px 10px',borderRadius:'var(--r-sm)'}}>3루 주자</span>}
          {!data.runners.first&&!data.runners.second&&!data.runners.third&&<span className="num" style={{fontSize:11,color:'var(--ink-4)'}}>주자 없음</span>}
        </div>
      </div>
    </div>
  )
}

/* ════════════════════════════════════

/* ════════════════════════════════════
   SCREEN: LIVE (라이브스코어)
════════════════════════════════════ */
function ScreenHistory() {
  const [data, setData] = useState<HistData|null>(null)
  const [loading, setLoading] = useState(true)
  const [days, setDays] = useState(7)
  const [conf, setConf] = useState<'ALL'|'HIGH'|'MED'|'LOW'>('ALL')
  const [sel, setSel] = useState<HistRow|null>(null)

  useEffect(()=>{
    setLoading(true)
    fetch(`${API}/predictions/history?days=${days}`)
      .then(r=>r.json()).then((d:HistData)=>{ setData(d); setLoading(false) })
      .catch(()=>setLoading(false))
  },[days])

  if (sel) return <DetailHistory row={sel} onBack={()=>setSel(null)}/>

  const rows = data?.rows ?? []
  const filtered = conf==='ALL' ? rows : rows.filter(r=>r.confidence===conf)
  const byDate: Record<string,HistRow[]> = {}
  filtered.forEach(r=>{ (byDate[r.date]??=[]).push(r) })
  const dates = Object.keys(byDate).sort((a,b)=>b.localeCompare(a))

  const graded = rows.filter(r=>r.is_correct!==null)
  const correct = rows.filter(r=>r.is_correct===1)

  return (
    <div className="view-enter">
      <div className="subhead">
        <div>
          <div className="kicker">PREDICTION LOG · {data ? `최근 ${days}일` : '—'}</div>
          <h1 className="page-title">결과 <span className="red">LOG</span></h1>
        </div>
        {data && (
          <div className="hero-kpis">
            <div className="hero-kpi"><div className="lbl">총 예측</div><div className="val cond">{data.total}</div><div className="sub">GRADED {data.graded}</div></div>
            <div className="hero-kpi"><div className="lbl">적중</div><div className="val cond green">{data.correct}</div><div className="sub">{data.accuracy!=null?`${data.accuracy}%`:'—'}</div></div>
            <div className="hero-kpi"><div className="lbl">Brier</div><div className={`val cond ${data.brier!=null&&data.brier<.23?'green':'red'}`}>{data.brier??'—'}</div><div className="sub">{data.brier!=null&&data.brier<.23?'○ 목표 달성':'✕ 미달'}</div></div>
            <div className="hero-kpi"><div className="lbl">정확도</div><div className="val cond">{data.accuracy!=null?`${data.accuracy}%`:'—'}</div><div className="sub">vs 베가스 ≈ 58%</div></div>
          </div>
        )}
      </div>

      {/* Confidence breakdown */}
      {data && data.graded>0 && (
        <div style={{display:'flex',alignItems:'center',gap:28,padding:'14px 32px',background:'var(--surface)',borderBottom:'1px solid var(--rule)'}}>
          <span className="lbl" style={{marginRight:4}}>신뢰도별 적중</span>
          {(['HIGH','MED','LOW'] as const).map(lv=>{
            const bucket = rows.filter(r=>r.confidence===lv)
            const hits = bucket.filter(r=>r.is_correct===1)
            const acc = bucket.length>0 ? hits.length/bucket.length : 0
            const color = lv==='HIGH'?'var(--navy)':lv==='MED'?'var(--ink-3)':'var(--red)'
            return (
              <div key={lv} style={{display:'flex',alignItems:'center',gap:12,flex:1,maxWidth:320}}>
                <Conf level={lv}/>
                <div style={{flex:1}}>
                  <div style={{height:6,background:'var(--rule-soft)',borderRadius:1,overflow:'hidden'}}>
                    <div style={{width:`${acc*100}%`,height:'100%',background:color}}/>
                  </div>
                  <div className="num" style={{fontSize:10.5,color:'var(--ink-3)',marginTop:3}}>
                    {hits.length}/{bucket.length} · <strong style={{color:'var(--ink)'}}>{bucket.length>0?(acc*100).toFixed(1):0}%</strong>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Filterbar */}
      <div className="filterbar">
        <span className="lbl" style={{marginRight:4}}>기간</span>
        {[5,7,14,30].map(d=>(
          <button key={d} className={`chip ${days===d?'active':''}`} onClick={()=>setDays(d)}>최근 {d}일</button>
        ))}
        <span style={{width:1,height:18,background:'var(--rule)',margin:'0 6px'}}/>
        {(['ALL','HIGH','MED','LOW'] as const).map(f=>(
          <button key={f} className={`chip ${conf===f?'active':''}`} onClick={()=>setConf(f)}>
            {f==='ALL'?'전체':f} <span className="count">{f==='ALL'?rows.length:rows.filter(r=>r.confidence===f).length}</span>
          </button>
        ))}
        <span style={{marginLeft:'auto',fontFamily:'var(--f-mono)',fontSize:11,color:'var(--ink-3)'}}>
          표시 중 {filtered.length} / 전체 {rows.length}
        </span>
      </div>

      <div style={{padding:'20px 32px 48px'}}>
        {loading && <Loading/>}
        {!loading && filtered.length===0 && <EmptyBox msg="결과 데이터가 없습니다" sub="경기 결과가 DB에 업데이트되면 표시됩니다"/>}
        {!loading && dates.map(date=>{
          const dayRows = byDate[date]
          const d = new Date(date+'T00:00:00')
          const wd = ['SUN','MON','TUE','WED','THU','FRI','SAT'][d.getDay()]
          const mm = String(d.getMonth()+1).padStart(2,'0')
          const dd = String(d.getDate()).padStart(2,'0')
          const graded2 = dayRows.filter(r=>r.is_correct!==null)
          const hits2 = dayRows.filter(r=>r.is_correct===1)
          const acc2 = graded2.length>0 ? hits2.length/graded2.length : null

          return (
            <div key={date} style={{marginBottom:28}}>
              {/* Date header */}
              <div style={{display:'flex',alignItems:'baseline',gap:14,padding:'10px 4px',marginBottom:8}}>
                <span className="cond" style={{fontSize:22,fontWeight:700,letterSpacing:'.01em'}}>{mm}.{dd}</span>
                <span className="cond" style={{fontSize:16,color:'var(--ink-4)',fontWeight:400}}>{wd}</span>
                <span style={{flex:1,height:1,background:'var(--rule)',alignSelf:'center'}}/>
                <span className="num" style={{fontSize:12,color:'var(--ink-3)'}}>
                  {dayRows.length}경기 · 적중 <strong style={{color:acc2!=null&&acc2>=.5?'var(--green)':'var(--red)'}}>{hits2.length}/{graded2.length}</strong>
                  {acc2!=null&&` · ${(acc2*100).toFixed(0)}%`}
                </span>
              </div>

              {/* Table */}
              <div className="log-table">
                <div className="log-hd">
                  {['시각','원정팀','스코어','홈팀','예측','신뢰도','결과',''].map((h,i)=>(
                    <span key={i} style={{textAlign:i===2?'center':i>=5?'right':'left'}}>{h}</span>
                  ))}
                </div>
                {dayRows.map((row,i)=>{
                  const hasScore = row.home_score!==null && row.away_score!==null
                  const homeWon = hasScore && row.home_score!>row.away_score!
                  const pickHome = row.pick_team===row.home_team
                  const timeStr = row.game_datetime
                    ? new Date(row.game_datetime).toLocaleTimeString('ko-KR',{hour:'2-digit',minute:'2-digit',hour12:false,timeZone:'Asia/Seoul'})
                    : '—'
                  return (
                    <div key={row.game_pk} className={`log-trow ${row.is_correct===1?'hit':row.is_correct===0?'miss':''}`}
                      onClick={()=>setSel(row)}>
                      {/* 시각 */}
                      <span className="num" style={{fontSize:12,color:'var(--ink-3)'}}>{timeStr}</span>

                      {/* Away */}
                      <div style={{display:'flex',alignItems:'center',gap:8}}>
                        <TM code={row.away_team} size="sm"/>
                        <div>
                          <div className="cond" style={{fontSize:14,fontWeight:600,textTransform:'uppercase',opacity:homeWon?.55:1}}>{row.away_team}</div>
                          <div className="cond" style={{fontSize:10,color:'var(--ink-4)'}}>{TEAMS[row.away_team]?.name}</div>
                        </div>
                      </div>

                      {/* Score */}
                      <div style={{textAlign:'center'}}>
                        {hasScore ? (
                          <span>
                            <span className="num" style={{fontSize:20,fontWeight:700,color:homeWon?'var(--ink-4)':'var(--ink)'}}>{row.away_score}</span>
                            <span className="num" style={{fontSize:13,color:'var(--ink-4)',margin:'0 5px'}}>–</span>
                            <span className="num" style={{fontSize:20,fontWeight:700,color:homeWon?'var(--ink)':'var(--ink-4)'}}>{row.home_score}</span>
                          </span>
                        ) : (
                          <span className="cond" style={{fontSize:12,color:'var(--ink-4)'}}>{row.status==='Preview'?'예정':row.status}</span>
                        )}
                      </div>

                      {/* Home */}
                      <div style={{display:'flex',alignItems:'center',gap:8,justifyContent:'flex-end'}}>
                        <div style={{textAlign:'right'}}>
                          <div className="cond" style={{fontSize:14,fontWeight:600,textTransform:'uppercase',opacity:homeWon?1:.55}}>{row.home_team}</div>
                          <div className="cond" style={{fontSize:10,color:'var(--ink-4)'}}>{TEAMS[row.home_team]?.name}</div>
                        </div>
                        <TM code={row.home_team} size="sm"/>
                      </div>

                      {/* Pick */}
                      <div>
                        <div className="cond" style={{fontSize:14,fontWeight:700,textTransform:'uppercase',color:pickHome?'var(--navy)':'var(--red)'}}>
                          {row.pick_team} 승 <span style={{color:'var(--ink-3)',fontWeight:400,marginLeft:4}}>{(row.pick_prob*100).toFixed(1)}%</span>
                        </div>
                        <div style={{marginTop:4}}><ProbBar home={row.home_win_prob} h={4}/></div>
                      </div>

                      {/* Conf */}
                      <div style={{textAlign:'right'}}><Conf level={row.confidence}/></div>

                      {/* Result */}
                      <div style={{textAlign:'right'}}><ResultChip v={row.is_correct}/></div>

                      {/* Chevron */}
                      <div style={{textAlign:'right'}} className="row-chevron">
                        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                          <path d="M5 2L10 7L5 12" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>
                        </svg>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

/* ════════════════════════════════════
   SCREEN: HISTORY DETAIL
════════════════════════════════════ */
function DetailHistory({ row:r, onBack }: { row:HistRow; onBack:()=>void }) {
  const [tab, setTab] = useState<'overview'|'stack'>('overview')
  const pickHome = r.pick_team===r.home_team
  const hasScore = r.home_score!==null && r.away_score!==null
  const homeWon = hasScore && r.home_score!>r.away_score!
  const aT = TEAMS[r.away_team], hT = TEAMS[r.home_team]

  return (
    <div className="view-enter">
      <div style={{display:'flex',alignItems:'center',gap:14,padding:'12px 32px',background:'var(--surface)',borderBottom:'1px solid var(--rule)'}}>
        <button className="btn btn-ghost" onClick={onBack}>← 경기 일정</button>
        <span style={{color:'var(--rule)'}}>·</span>
        <span className="num" style={{fontSize:11,color:'var(--ink-3)',letterSpacing:'.06em'}}>{r.date} · {r.away_team} vs {r.home_team}</span>
        <span style={{flex:1}}/>
        {r.is_correct!==null && (
          <span className="cond" style={{fontSize:13,fontWeight:700,letterSpacing:'.08em',padding:'4px 10px',borderRadius:3,background:r.is_correct===1?'var(--green)':'var(--red)',color:'#fff'}}>
            {r.is_correct===1?'○ HIT':'✕ MISS'}
          </span>
        )}
      </div>

      {/* Hero */}
      <div style={{background:'var(--surface)',borderBottom:'1px solid var(--rule)',padding:'28px 32px'}}>
        <div className="kicker lbl" style={{marginBottom:14}}>{r.date} · 경기 결과</div>
        <div style={{display:'grid',gridTemplateColumns:'1fr auto 1fr',gap:24,alignItems:'center'}}>
          {/* Away */}
          <div style={{display:'flex',gap:16,alignItems:'center',opacity:homeWon?.7:1}}>
            <TM code={r.away_team} size="xl"/>
            <div>
              <div className="lbl">{!pickHome&&<span style={{marginRight:8,background:'var(--navy)',color:'#fff',padding:'2px 6px',borderRadius:2,fontSize:10,fontWeight:700}}>모델 픽</span>}AWAY</div>
              <div className="cond" style={{fontSize:13,color:'var(--ink-3)',marginTop:2}}>{aT?.city}</div>
              <div className="cond" style={{fontSize:28,fontWeight:700,textTransform:'uppercase',lineHeight:1.05}}>{aT?.name??r.away_team}</div>
              {hasScore && <div className="num" style={{fontSize:38,fontWeight:700,marginTop:10,color:homeWon?'var(--ink-4)':'var(--ink)'}}>{r.away_score}</div>}
            </div>
          </div>

          <div style={{textAlign:'center'}}>
            <div className="cond" style={{fontSize:38,color:'var(--ink-4)',lineHeight:1,fontWeight:300}}>vs</div>
            <div className="num" style={{fontSize:11,color:'var(--ink-3)',marginTop:10,letterSpacing:'.05em'}}>FINAL</div>
          </div>

          {/* Home */}
          <div style={{display:'flex',gap:16,alignItems:'center',flexDirection:'row-reverse',textAlign:'right',opacity:homeWon?1:.7}}>
            <TM code={r.home_team} size="xl"/>
            <div>
              <div className="lbl">{pickHome&&<span style={{marginRight:8,background:'var(--navy)',color:'#fff',padding:'2px 6px',borderRadius:2,fontSize:10,fontWeight:700}}>모델 픽</span>}HOME</div>
              <div className="cond" style={{fontSize:13,color:'var(--ink-3)',marginTop:2}}>{hT?.city}</div>
              <div className="cond" style={{fontSize:28,fontWeight:700,textTransform:'uppercase',lineHeight:1.05}}>{hT?.name??r.home_team}</div>
              {hasScore && <div className="num" style={{fontSize:38,fontWeight:700,marginTop:10,color:homeWon?'var(--ink)':'var(--ink-4)'}}>{r.home_score}</div>}
            </div>
          </div>
        </div>

        {/* Prediction vs Actual */}
        <div style={{marginTop:24,display:'grid',gridTemplateColumns:'1fr 1fr',gap:16}}>
          <div style={{border:'1px solid var(--rule)',borderRadius:'var(--r-md)',padding:'16px 20px',background:'var(--surface-2)'}}>
            <div style={{display:'flex',justifyContent:'space-between',marginBottom:10}}>
              <span className="lbl">예측 (경기 전)</span>
              <Conf level={r.confidence}/>
            </div>
            <div className="cond" style={{fontSize:22,fontWeight:700,textTransform:'uppercase'}}>
              {r.pick_team} 승 <span style={{color:'var(--ink-3)',fontWeight:400,marginLeft:6}}>{(r.pick_prob*100).toFixed(1)}%</span>
            </div>
            <div style={{marginTop:10,height:22,display:'flex',borderRadius:2,overflow:'hidden',fontFamily:'var(--f-cond)',fontWeight:700,fontSize:11,letterSpacing:'.05em'}}>
              <div style={{width:`${r.away_win_prob*100}%`,background:'var(--prob-away)',color:'#fff',display:'flex',alignItems:'center',paddingLeft:10}}>{r.away_team} {(r.away_win_prob*100).toFixed(1)}%</div>
              <div style={{width:`${r.home_win_prob*100}%`,background:'var(--prob-home)',color:'#fff',display:'flex',alignItems:'center',justifyContent:'flex-end',paddingRight:10}}>{r.home_team} {(r.home_win_prob*100).toFixed(1)}%</div>
            </div>
          </div>
          {hasScore && (
            <div style={{border:`1px solid ${r.is_correct===1?'var(--green)':'var(--red)'}`,borderRadius:'var(--r-md)',padding:'16px 20px',background:r.is_correct===1?'rgba(30,140,90,.04)':'rgba(191,13,62,.03)'}}>
              <div style={{display:'flex',justifyContent:'space-between',marginBottom:10}}>
                <span className="lbl" style={{color:r.is_correct===1?'var(--green)':'var(--red)'}}>실제 결과</span>
                <span className="cond" style={{fontSize:11,fontWeight:700,letterSpacing:'.1em',color:r.is_correct===1?'var(--green)':'var(--red)'}}>
                  {r.is_correct===1?'○ PREDICTION CORRECT':'✕ PREDICTION MISSED'}
                </span>
              </div>
              <div className="cond" style={{fontSize:22,fontWeight:700,textTransform:'uppercase'}}>
                {homeWon?r.home_team:r.away_team} 승 <span className="num" style={{color:'var(--ink-3)',fontWeight:400,marginLeft:6}}>{r.away_score}–{r.home_score}</span>
              </div>
              <div className="num" style={{fontSize:11,color:'var(--ink-3)',marginTop:10}}>
                {r.pick_team} 예측 → {homeWon?r.home_team:r.away_team} 실제 · {r.is_correct===1?'모델 적중':'모델 실패'}
              </div>
            </div>
          )}
        </div>
      </div>

      <div style={{background:'var(--surface)',padding:'0 32px',borderBottom:'1px solid var(--rule)'}}>
        <div className="tabs">
          {[{id:'overview',label:'개요'},{id:'stack',label:'모델 스택'}].map(t=>(
            <button key={t.id} className={`tab ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id as any)}>{t.label}</button>
          ))}
        </div>
      </div>

      <div style={{padding:'26px 32px 40px'}}>
        {tab==='overview' && (
          <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:16}}>
            {[['홈팀 승리 확률',`${(r.home_win_prob*100).toFixed(1)}%`],['원정팀 승리 확률',`${(r.away_win_prob*100).toFixed(1)}%`],['신뢰도',r.confidence],['최종 결과',r.is_correct===1?'HIT':r.is_correct===0?'MISS':'미집계']].map(([k,v])=>(
              <div key={k} className="panel" style={{padding:'18px 20px'}}>
                <div className="lbl">{k}</div>
                <div className="cond" style={{fontSize:28,fontWeight:700,marginTop:6}}>{v}</div>
              </div>
            ))}
          </div>
        )}
        {tab==='stack' && (
          <div className="panel dark" style={{padding:'24px 28px'}}>
            <div className="lbl on-dark" style={{marginBottom:16}}>MODEL STACK · 이 예측에 사용된 알고리즘</div>
            <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:0}}>
              {[{cat:'LGBM',desc:'LightGBM v4.x',sub:'그래디언트 부스팅'},{cat:'XGB',desc:'XGBoost v2.x',sub:'앙상블 보완'},{cat:'CAL',desc:'Isotonic Reg.',sub:'확률 보정'},{cat:'SHAP',desc:'TreeExplainer',sub:'근거 분석'}].map((s,i)=>(
                <div key={s.cat} style={{padding:'0 20px',borderLeft:i?'1px solid rgba(255,255,255,.08)':'none'}}>
                  <div style={{fontFamily:'var(--f-mono)',fontSize:9,letterSpacing:'.18em',color:'var(--red)',marginBottom:8}}>{s.cat}</div>
                  <div className="cond" style={{fontSize:18,fontWeight:700,color:'#fff'}}>{s.desc}</div>
                  <div className="num" style={{fontSize:10,color:'#8FA3C0',marginTop:4}}>{s.sub}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

/* ════════════════════════════════════
   SCREEN: MODEL PERFORMANCE
════════════════════════════════════ */

/* ════════════════════════════════════
   SCREEN: ARCHIVE (캘린더)
════════════════════════════════════ */
function ScreenArchive() {
  const _kstStr=getKstDateStr()
  const [year,setYear]=useState(parseInt(_kstStr.slice(0,4)))
  const [month,setMonth]=useState(parseInt(_kstStr.slice(5,7)))
  const [calData,setCalData]=useState<CalDay[]>([])
  const [selDate,setSelDate]=useState<string|null>(null)
  const [summary,setSummary]=useState<ArchiveSummary|null>(null)
  const [loadSum,setLoadSum]=useState(false)

  useEffect(()=>{
    fetch(`${API}/archive/calendar?year=${year}&month=${month}`).then(r=>r.json()).then(d=>setCalData(d.days||[])).catch(()=>{})
  },[year,month])

  useEffect(()=>{
    if(!selDate) return
    setLoadSum(true);setSummary(null)
    fetch(`${API}/archive/summary?target_date=${selDate}`).then(r=>r.json()).then(d=>{setSummary(d);setLoadSum(false)}).catch(()=>setLoadSum(false))
  },[selDate])

  const calMap:Record<string,CalDay>={}
  calData.forEach(d=>calMap[d.date]=d)
  const firstDay=new Date(year,month-1,1).getDay()
  const daysInMonth=new Date(year,month,0).getDate()
  const todayStr=getKstDateStr()
  const cells:Array<{date:string|null;day:number|null}>=[]
  for(let i=0;i<firstDay;i++) cells.push({date:null,day:null})
  for(let d=1;d<=daysInMonth;d++){
    const dateStr=`${year}-${String(month).padStart(2,'0')}-${String(d).padStart(2,'0')}`
    cells.push({date:dateStr,day:d})
  }
  const prevMonth=()=>{if(month===1){setYear(y=>y-1);setMonth(12)}else setMonth(m=>m-1)}
  const nextMonth=()=>{if(month===12){setYear(y=>y+1);setMonth(1)}else setMonth(m=>m+1)}

  return (
    <div className="view-enter">
      <div className="subhead">
        <div><div className="kicker">ARCHIVE · 캘린더 기반 AI 예측 아카이브</div><h1 className="page-title">예측 <span className="red">아카이브</span></h1></div>
      </div>
      <div style={{display:'grid',gridTemplateColumns:'340px 1fr',flex:1}}>
        <div style={{borderRight:'1px solid var(--rule)',padding:'24px',background:'var(--surface)'}}>
          <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:20}}>
            <button className="btn ghost" style={{padding:'6px 10px'}} onClick={prevMonth}>←</button>
            <div className="cond" style={{fontSize:18,fontWeight:700}}>{year}년 {month}월</div>
            <button className="btn ghost" style={{padding:'6px 10px'}} onClick={nextMonth}>→</button>
          </div>
          <div className="cal-grid" style={{marginBottom:8}}>
            {['일','월','화','수','목','금','토'].map(d=><div key={d} style={{textAlign:'center',fontFamily:'var(--f-mono)',fontSize:10,color:'var(--ink-4)',padding:'4px 0'}}>{d}</div>)}
          </div>
          <div className="cal-grid">
            {cells.map((c,i)=>{
              if(!c.date) return <div key={i}/>
              const cd=calMap[c.date]
              const isToday=c.date===todayStr
              const isSel=c.date===selDate
              const dotClass=cd?.accuracy!=null?(cd.accuracy>=60?'good':cd.accuracy>=40?'med':'bad'):''
              return (
                <div key={c.date} className={`cal-day ${cd?'has-data':''} ${isSel?'selected':''} ${isToday&&!isSel?'today-mark':''} ${dotClass}`}
                  onClick={()=>cd&&setSelDate(c.date!)}>
                  <span>{c.day}</span>
                  {cd&&<div className="cal-dot"/>}
                </div>
              )
            })}
          </div>
          <div style={{marginTop:16,display:'flex',gap:14}}>
            {[['good','var(--green)','60%+ 적중'],['med','var(--amber)','40~60%'],['bad','var(--red)','40% 미만']].map(([cls,color,lbl])=>(
              <div key={cls} style={{display:'flex',alignItems:'center',gap:5}}><div style={{width:8,height:8,borderRadius:'50%',background:color}}/><span className="num" style={{fontSize:10,color:'var(--ink-4)'}}>{lbl}</span></div>
            ))}
          </div>
        </div>
        <div style={{padding:'24px 28px 40px'}}>
          {!selDate&&<div style={{display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center',height:300,color:'var(--ink-4)'}}><div style={{fontSize:40,marginBottom:12}}>📅</div><div className="cond" style={{fontSize:18,fontWeight:700}}>날짜를 선택하세요</div><div className="num" style={{fontSize:11,marginTop:6}}>날짜를 클릭하면 해당 일자의 예측 결과를 확인할 수 있습니다</div></div>}
          {selDate&&loadSum&&<Spinner/>}
          {selDate&&!loadSum&&summary&&(
            <div className="view-enter">
              <div style={{display:'flex',alignItems:'baseline',gap:14,marginBottom:18}}>
                <div className="cond" style={{fontSize:26,fontWeight:700}}>{selDate}</div>
                <span style={{flex:1,height:1,background:'var(--rule)',alignSelf:'center'}}/>
                <span className="num" style={{fontSize:12,color:'var(--ink-3)'}}>{summary.total}경기 · 적중 {summary.correct}/{summary.graded} · {summary.accuracy!=null?`${summary.accuracy}%`:'—'}</span>
              </div>
              {summary.games.length===0?<EmptyBox msg="해당 날짜의 예측 데이터가 없습니다"/>:(
                <div className="log-table">
                  <div style={{display:'grid',gridTemplateColumns:'1.2fr 1.2fr 100px 64px 72px',gap:10,padding:'9px 16px',background:'var(--surface-2)',borderBottom:'1px solid var(--rule)'}}>
                    {['원정팀','홈팀','스코어','신뢰도','결과'].map((h,i)=><span key={i} className="num" style={{fontSize:9,letterSpacing:'.12em',textTransform:'uppercase',color:'var(--ink-4)',textAlign:i>=2?'center':'left'}}>{h}</span>)}
                  </div>
                  {summary.games.map((g:any)=>(
                    <div key={g.game_pk} style={{display:'grid',gridTemplateColumns:'1.2fr 1.2fr 100px 64px 72px',gap:10,padding:'12px 16px',alignItems:'center',borderBottom:'1px solid var(--rule-soft)'}}>
                      <div style={{display:'flex',alignItems:'center',gap:7}}><TM code={g.away_team} size="sm"/><div><div className="cond" style={{fontSize:14,fontWeight:600,textTransform:'uppercase'}}>{g.away_team}</div><div className="num" style={{fontSize:10,color:'var(--red)'}}>{(g.away_win_prob*100).toFixed(1)}%</div></div></div>
                      <div style={{display:'flex',alignItems:'center',gap:7}}><TM code={g.home_team} size="sm"/><div><div className="cond" style={{fontSize:14,fontWeight:600,textTransform:'uppercase'}}>{g.home_team}</div><div className="num" style={{fontSize:10,color:'var(--navy)'}}>{(g.home_win_prob*100).toFixed(1)}%</div></div></div>
                      <div style={{textAlign:'center'}}><span className="num" style={{fontSize:16,fontWeight:700}}>{g.home_score!=null?`${g.away_score}–${g.home_score}`:'—'}</span></div>
                      <div style={{textAlign:'center'}}><Conf level={g.confidence}/></div>
                      <div style={{textAlign:'center'}}><RC v={g.is_correct}/></div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}


/* ════════════════════════════════════
   SCREEN: SCHEDULE (경기 일정)
   - 기존 History/Archive API를 유지하면서 한 화면으로 통합
════════════════════════════════════ */
function ScreenSchedule() {
  const _kstStr = getKstDateStr()
  const initial = _kstStr
  const [year,setYear] = useState(parseInt(_kstStr.slice(0,4)))
  const [month,setMonth] = useState(parseInt(_kstStr.slice(5,7)))
  const [calData,setCalData] = useState<CalDay[]>([])
  const [mlbCal,setMlbCal] = useState<Record<string,number>>({})
  const [selDate,setSelDate] = useState<string>(initial)
  const [summary,setSummary] = useState<ArchiveSummary|null>(null)
  const [mlbGames,setMlbGames] = useState<MlbScheduleGame[]>([])
  const [loadSum,setLoadSum] = useState(false)
  const [refreshTick,setRefreshTick] = useState(0)

  useEffect(()=>{
    fetch(`${API}/archive/calendar?year=${year}&month=${month}`)
      .then(r=>r.json())
      .then(d=>setCalData(d.days||[]))
      .catch(()=>setCalData([]))
  },[year,month])

  useEffect(()=>{
    // KST date = US date + 1 day, so fetch US dates [month-1 last day .. month last day - 1]
    const prevLastDay = new Date(year, month-1, 0)
    const startDate = `${prevLastDay.getFullYear()}-${String(prevLastDay.getMonth()+1).padStart(2,'0')}-${String(prevLastDay.getDate()).padStart(2,'0')}`
    const lastDay = new Date(year, month, 0).getDate()
    const endDate = `${year}-${String(month).padStart(2,'0')}-${String(lastDay-1).padStart(2,'0')}`
    fetch(`https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate=${startDate}&endDate=${endDate}`)
      .then(r=>r.json())
      .then(d=>{
        const map:Record<string,number> = {}
        ;(d?.dates || []).forEach((day:any)=>{
          // Shift US date → KST date (+1 day)
          const [uy,um,ud] = day.date.split('-').map(Number)
          const kst = new Date(uy, um-1, ud+1)
          const kstStr = `${kst.getFullYear()}-${String(kst.getMonth()+1).padStart(2,'0')}-${String(kst.getDate()).padStart(2,'0')}`
          map[kstStr] = day.totalGames || (day.games?.length ?? 0)
        })
        setMlbCal(map)
      })
      .catch(()=>setMlbCal({}))
  },[year,month])

  const fetchDayData = useCallback((d:string, showLoading:boolean)=>{
    if(showLoading){ setLoadSum(true); setSummary(null); setMlbGames([]) }
    Promise.allSettled([
      fetch(`${API}/archive/summary?target_date=${d}`).then(r=>r.json()),
      fetchMlbScheduleByDate(d),
    ]).then(([archiveRes, mlbRes])=>{
      if(archiveRes.status==='fulfilled') setSummary(archiveRes.value)
      else if(showLoading) setSummary({date:d,total:0,graded:0,correct:0,accuracy:null,high_med_accuracy:null,games:[]})
      if(mlbRes.status==='fulfilled') setMlbGames(mlbRes.value)
      else if(showLoading) setMlbGames([])
      setLoadSum(false)
    }).catch(()=>setLoadSum(false))
  },[])

  useEffect(()=>{ fetchDayData(selDate, true) },[selDate, fetchDayData])

  // 오늘 날짜 선택 시 5분마다 자동 갱신
  useEffect(()=>{
    if(selDate!==getKstDateStr()) return
    const id = setInterval(()=>{ setRefreshTick(t=>t+1) }, 5*60*1000)
    return ()=>clearInterval(id)
  },[selDate])
  useEffect(()=>{ if(refreshTick>0) fetchDayData(selDate, false) },[refreshTick, selDate, fetchDayData])

  const predMap:Record<number,any> = {}
  ;(summary?.games || []).forEach((g:any)=>{ predMap[g.game_pk]=g })

  const mergedGames = mlbGames
    .map(m => ({...m, prediction: predMap[m.game_pk] || null}))
    .sort((a:any,b:any)=>String(a.game_datetime||'').localeCompare(String(b.game_datetime||'')))

  const firstDay = new Date(year,month-1,1).getDay()
  const daysInMonth = new Date(year,month,0).getDate()
  const todayStr = getKstDateStr()
  const cells:Array<{date:string|null;day:number|null}> = []
  for(let i=0;i<firstDay;i++) cells.push({date:null,day:null})
  for(let d=1;d<=daysInMonth;d++) cells.push({date:`${year}-${String(month).padStart(2,'0')}-${String(d).padStart(2,'0')}`,day:d})

  const calMap:Record<string,CalDay> = {}
  calData.forEach(d=>calMap[d.date]=d)

  const moveMonth = (delta:number) => {
    const base = new Date(year, month-1+delta, 1)
    setYear(base.getFullYear())
    setMonth(base.getMonth()+1)
  }
  const moveDate = (delta:number) => {
    const [y,m,dy] = selDate.split('-').map(Number)
    const d = new Date(y, m-1, dy+delta)
    const next = `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`
    setSelDate(next); setYear(d.getFullYear()); setMonth(d.getMonth()+1)
  }
  const dayLabel = (() => {
    const d = new Date(selDate+'T00:00:00')
    const wd = ['일','월','화','수','목','금','토'][d.getDay()]
    return `${d.getFullYear()}.${String(d.getMonth()+1).padStart(2,'0')}.${String(d.getDate()).padStart(2,'0')} (${wd})`
  })()

  const totalGames = mergedGames.length
  const predTotal = mergedGames.filter((m:any)=>m.prediction).length

  let _graded = 0, _correct = 0
  mergedGames.forEach((item:any)=>{
    const pred = item.prediction
    if(!pred) return
    const hs = pred?.home_score ?? item.home_score
    const as_ = pred?.away_score ?? item.away_score
    if(hs==null || as_==null) return
    const homeWon = hs > as_
    const pickHome = pred.home_win_prob >= 0.5
    const ic = pred.is_correct ?? (homeWon===pickHome ? 1 : 0)
    _graded++
    if(ic===1) _correct++
  })
  const graded = _graded
  const correct = _correct
  const accuracy = graded>0 ? Math.round(correct/graded*1000)/10 : null

  return (
    <div className="view-enter schedule-page">
      <div className="subhead">
        <div>
          <div className="kicker">SCHEDULE · 날짜별 AI 예측 및 경기 결과</div>
          <h1 className="page-title">경기 <span className="red">일정</span></h1>
        </div>
        <div className="hero-kpis">
          <div className="hero-kpi"><div className="hk-lbl">선택 날짜</div><div className="hk-val cond">{dayLabel.slice(5,10)}</div><div className="hk-sub">{dayLabel.slice(-3)}</div></div>
          <div className="hero-kpi"><div className="hk-lbl">전체 경기</div><div className="hk-val cond">{totalGames}</div><div className="hk-sub">MLB</div></div>
          <div className="hero-kpi"><div className="hk-lbl">AI 예측</div><div className="hk-val cond">{predTotal}</div><div className="hk-sub">PREDICTED</div></div>
          <div className="hero-kpi"><div className="hk-lbl">적중</div><div className="hk-val cond green">{correct}/{graded}</div><div className="hk-sub">{accuracy!=null?`${accuracy}%`:'AI RECORD'}</div></div>
        </div>
      </div>

      <div className="schedule-layout">
        <aside className="schedule-calendar">
          <div className="schedule-cal-top">
            <button className="btn ghost" onClick={()=>moveMonth(-1)}>←</button>
            <div><div className="cond" style={{fontSize:22,fontWeight:700}}>{year}년 {month}월</div><div className="num" style={{fontSize:10,color:'var(--ink-4)',marginTop:3}}>MLB SCHEDULE CALENDAR</div></div>
            <button className="btn ghost" onClick={()=>moveMonth(1)}>→</button>
          </div>
          <div className="cal-grid schedule-week">{['일','월','화','수','목','금','토'].map(d=><div key={d}>{d}</div>)}</div>
          <div className="cal-grid">
            {cells.map((c,i)=>{
              if(!c.date) return <div key={i}/>
              const cd=calMap[c.date]
              const hasMlb=(mlbCal[c.date]||0)>0
              const isToday=c.date===todayStr
              const isSel=c.date===selDate
              const dotClass=cd?.accuracy!=null?(cd.accuracy>=60?'good':cd.accuracy>=40?'med':'bad'):(hasMlb?'mlb':'')
              return <div key={c.date} className={`cal-day schedule-cal-day ${hasMlb?'has-mlb':''} ${cd?'has-data':''} ${isSel?'selected':''} ${isToday&&!isSel?'today-mark':''} ${dotClass}`} onClick={()=>setSelDate(c.date!)}><span>{c.day}</span>{hasMlb&&<div className="cal-dot"/>}{cd&&cd.graded>0&&<small>{cd.correct}/{cd.graded}</small>}</div>
            })}
          </div>
          <div className="schedule-legend">
            <span><i style={{background:'var(--green)'}}/>예측 적중률 60% 이상</span>
            <span><i style={{background:'var(--amber)'}}/>예측 적중률 40~60%</span>
            <span><i style={{background:'var(--red)'}}/>예측 적중률 40% 미만</span>
            <span><i style={{background:'var(--ink-4)'}}/>MLB 경기 있음</span>
          </div>
        </aside>

        <main className="schedule-games">
          <div className="schedule-date-head">
            <div><div className="kicker">SELECTED DATE</div><div className="cond" style={{fontSize:30,fontWeight:700}}>{dayLabel}</div></div>
            <div className="num" style={{fontSize:13,color:'var(--ink-2)',fontWeight:700}}>{totalGames}경기 · AI 예측 {predTotal}경기 · 채점 {graded}경기 · 적중 {correct}경기</div>
          </div>

          {loadSum && <Spinner/>}
          {!loadSum && mergedGames.length===0 && <EmptyBox msg="해당 날짜의 MLB 경기 일정이 없습니다" sub="MLB Stats API 일정이 없거나 아직 공개되지 않은 날짜입니다"/>}

          {!loadSum && mergedGames.length>0 && (
            <div className="schedule-list">
              {mergedGames.map((item:any)=>{
                const pred = item.prediction
                const hasPrediction = !!pred
                const away = normalizeTeamCode(pred?.away_team || item.away_team)
                const home = normalizeTeamCode(pred?.home_team || item.home_team)
                const hasScore = (pred?.home_score ?? item.home_score) != null && (pred?.away_score ?? item.away_score) != null
                const awayScore = pred?.away_score ?? item.away_score
                const homeScore = pred?.home_score ?? item.home_score
                const homeWon = hasScore && homeScore > awayScore
                const pickHome = hasPrediction ? (pred.pick_team ? pred.pick_team===home : pred.home_win_prob>=.5) : false
                const pickTeam = hasPrediction ? (pred.pick_team || (pickHome ? home : away)) : ''
                const pickProb = hasPrediction ? (pred.pick_prob ?? Math.max(pred.home_win_prob,pred.away_win_prob)) : null
                let isCorrect:number|null = pred?.is_correct ?? null
                if(isCorrect===null && hasPrediction && hasScore) isCorrect = (homeWon===pickHome) ? 1 : 0
                const timeStr = formatKstTime(pred?.game_datetime || item.game_datetime)
                const status = statusKo(item.status, item.detailed_state)
                const awayPitcher = hasScore
                  ? (homeWon ? item.loss_pitcher : item.win_pitcher) || pred?.away_starter_name || null
                  : item.away_probable_pitcher || pred?.away_starter_name || null
                const homePitcher = hasScore
                  ? (homeWon ? item.win_pitcher : item.loss_pitcher) || pred?.home_starter_name || null
                  : item.home_probable_pitcher || pred?.home_starter_name || null
                const awayPitcherCls = hasScore ? (!homeWon ? 'win' : 'loss') : 'start'
                const homePitcherCls = hasScore ? (homeWon ? 'win' : 'loss') : 'start'
                return (
                  <div key={item.game_pk} className={`schedule-game-row ${isCorrect===1?'hit':isCorrect===0?'miss':''} ${!hasPrediction?'no-pred':''}`}>
                    <div className="schedule-time"><span className="num">{timeStr}</span><span className="cond">{status}</span></div>
                    <div className="schedule-team away"><TM code={away} size="lg"/><div><div className="lbl">AWAY</div><div className="cond team-code">{away}</div><div className="cond team-name">{TEAMS[away]?.city} {TEAMS[away]?.name || item.away_name}</div>{hasPrediction&&<div className="num team-prob">{(pred.away_win_prob*100).toFixed(1)}%</div>}{awayPitcher&&<div className={`pitcher-label ${awayPitcherCls}`}>{getLastName(awayPitcher)}</div>}</div></div>
                    <div className="schedule-score">{hasScore?(<><span className={`score-num ${!homeWon?'win':''}`}>{awayScore}</span><span className="score-mid">vs</span><span className={`score-num ${homeWon?'win':''}`}>{homeScore}</span></>):(<span className="score-vs">vs</span>)}</div>
                    <div className="schedule-team home"><div><div className="lbl">HOME</div><div className="cond team-code">{home}</div><div className="cond team-name">{TEAMS[home]?.city} {TEAMS[home]?.name || item.home_name}</div>{hasPrediction&&<div className="num team-prob home-prob">{(pred.home_win_prob*100).toFixed(1)}%</div>}{homePitcher&&<div className={`pitcher-label ${homePitcherCls}`}>{getLastName(homePitcher)}</div>}</div><TM code={home} size="lg"/></div>
                    <div className="schedule-pred">
                      {hasPrediction ? (<><div className="lbl">AI 예측</div><div className="cond pred-pick" style={{color:pickHome?'var(--navy)':'var(--red)'}}>{pickTeam} 승 <span>{pickProb!=null?(pickProb*100).toFixed(1):'—'}%</span></div><PBar home={pred.home_win_prob} h={7}/></>) : (<><div className="lbl">AI 예측</div><div className="cond pred-empty">예측 없음</div><div className="num" style={{fontSize:11,color:'var(--ink-4)'}}>MLB 일정만 표시</div></>)}
                    </div>
                    <div className="schedule-result">{hasPrediction ? (<><Conf level={pred.confidence}/><RC v={isCorrect}/></>) : (<span className="rc rc-pend">일정</span>)}</div>
                  </div>
                )
              })}
            </div>
          )}
        </main>
      </div>
    </div>
  )
}

/* 타석 이벤트 → 한국어 변환 */
const EVENT_KO: Record<string,string> = {
  'Single':'안타!','Double':'2루타!','Triple':'3루타!','Home Run':'홈런!',
  'Strikeout':'삼진','Walk':'볼넷','Groundout':'땅볼아웃','Flyout':'플라이아웃',
  'Lineout':'라인드라이브아웃','Pop Out':'내야플라이','Double Play':'병살타!',
  'Triple Play':'삼중살!','Sac Fly':'희생플라이','Sac Bunt':'희생번트',
  'Field Error':'에러','Hit By Pitch':'몸에 맞는 공','Intent Walk':'고의사구',
  'Passed Ball':'패스트볼','Wild Pitch':'폭투','Balk':'보크',
  'Stolen Base 2B':'도루!','Stolen Base 3B':'도루!','Stolen Base Home':'홈도루!!',
  'Caught Stealing 2B':'도루실패','Caught Stealing 3B':'도루실패',
}
const toKoEvent=(e:string)=>EVENT_KO[e]||(e||'')

/* 이벤트 플래시가 인상적인 이벤트 목록 */
const EXCITING = new Set(['Home Run','Triple','Double Play','Triple Play','Stolen Base Home'])

const VENUE_SHORT: Record<string,string> = {
  'Oriole Park at Camden Yards':'Camden Yards',
  'Guaranteed Rate Field':'Guaranteed Rate',
  'American Family Field':'Am. Family Field',
  'loanDepot park':'loanDepot Park',
  'Angel Stadium of Anaheim':'Angel Stadium',
  'Oakland Coliseum':'Coliseum',
  'T-Mobile Park':'T-Mobile Park',
  'Busch Stadium':'Busch Stadium',
  'Tropicana Field':'Tropicana Field',
}
function shortVenue(v:string) { return VENUE_SHORT[v] || v }

function ScreenLive() {
  const [data,setData]=useState<TodayData|null>(null)
  const [lives,setLives]=useState<Record<number,LiveData>>({})
  const [loading,setLoading]=useState(true)
  const [filter,setFilter]=useState<'ALL'|'Live'|'Preview'|'Final'>('ALL')
  const [sseConnected,setSseConnected]=useState<Record<number,boolean>>({})
  const [flashEvent,setFlashEvent]=useState<Record<number,string>>({})
  const prevProbs=useRef<Record<number,number>>({})
  const [probDeltas,setProbDeltas]=useState<Record<number,number>>({})

  // 초기 경기 목록 + 60초 폴백 폴링 (예정/종료 경기용)
  useEffect(()=>{
    fetch(`${API}/predictions/today`).then(r=>r.json()).then((d:TodayData)=>{setData(d);setLoading(false)}).catch(()=>setLoading(false))
  },[])

  // 최초 REST 스냅샷 + 60초 폴링 (SSE 미연결 경기 대상)
  useEffect(()=>{
    if(!data?.games) return
    const fetchAll=()=>{
      data.games.forEach(g=>{
        if(sseConnected[g.game_pk]) return  // SSE 연결된 경기는 스킵
        fetch(`${API}/live/game/${g.game_pk}`).then(r=>r.json()).then(d=>{
          setLives(prev=>({...prev,[g.game_pk]:d}))
          const np=d.live_home_prob as number|undefined
          if(np!=null){const op=prevProbs.current[g.game_pk];if(op!=null&&Math.abs(np-op)>0.001)setProbDeltas(prev=>({...prev,[g.game_pk]:np-op}));prevProbs.current[g.game_pk]=np}
        }).catch(()=>{})
      })
    }
    fetchAll()
    const t=setInterval(fetchAll,60000)
    return ()=>clearInterval(t)
  },[data,sseConnected])

  // 진행 중 경기에 SSE 구독
  useEffect(()=>{
    if(!data?.games) return
    const liveGamePks = data.games
      .filter(g=>lives[g.game_pk]?.status==='Live'||(!lives[g.game_pk]))
      .map(g=>g.game_pk)

    const srcs: Record<number,EventSource> = {}
    liveGamePks.forEach(pk=>{
      if(sseConnected[pk]) return
      const es = new EventSource(`${API}/live/stream/${pk}`)
      es.onmessage=(e)=>{
        try {
          const msg = JSON.parse(e.data)
          if(msg.type==='connected') { setSseConnected(prev=>({...prev,[pk]:true})); return }
          // SSE 데이터로 lives 업데이트
          setLives(prev=>{
            const cur = prev[pk] || {} as LiveData
            return {...prev,[pk]:{
              ...cur,
              status: msg.status,
              inning_state: msg.half==='top'?'Top':'Bottom',
              current_inning: msg.inning,
              outs: msg.outs,
              balls: msg.balls,
              strikes: msg.strikes,
              runs:{home:msg.home_score,away:msg.away_score},
              runners:{first:!!msg.on1,second:!!msg.on2,third:!!msg.on3},
              live_home_prob: msg.live_home_prob,
              play_event: msg.play_event,
              is_new_play: msg.is_new_play,
            }}
          })
          const _np=msg.live_home_prob as number|undefined
          if(_np!=null){const _op=prevProbs.current[pk];if(_op!=null&&Math.abs(_np-_op)>0.001)setProbDeltas(prev=>({...prev,[pk]:_np-_op}));prevProbs.current[pk]=_np}
          // 새 타석 이벤트 플래시
          if(msg.is_new_play && msg.play_event) {
            setFlashEvent(prev=>({...prev,[pk]:msg.play_event}))
            setTimeout(()=>setFlashEvent(prev=>({...prev,[pk]:''})),3000)
          }
        } catch {}
      }
      es.onerror=()=>{
        setSseConnected(prev=>({...prev,[pk]:false}))
        es.close()
      }
      srcs[pk]=es
    })
    return ()=>{ Object.values(srcs).forEach(es=>es.close()) }
  },[data,lives])

  const games=data?.games??[]
  const liveGames=games.filter(g=>lives[g.game_pk]?.status==='Live')
  const previewGames=games.filter(g=>!lives[g.game_pk]||lives[g.game_pk]?.status==='Preview')
  const finalGames=games.filter(g=>lives[g.game_pk]?.status==='Final')
  const filtered = filter==='ALL' ? games : filter==='Live' ? liveGames : filter==='Preview' ? previewGames : finalGames
  const grouped: Record<string, TodayGame[]> = {}
  if(filter==='ALL') {
    if(liveGames.length>0) grouped['진행 중']=liveGames
    if(previewGames.length>0) grouped['예정']=previewGames
    if(finalGames.length>0) grouped['종료']=finalGames
  } else grouped[filter]=filtered

  return (
    <div className="view-enter live-page">
      <div className="subhead">
        <div><div className="kicker">LIVE SCOREBOARD · {getKstDateStr()}</div><h1 className="page-title">라이브 <span className="red">스코어</span></h1></div>
        <div className="hero-kpis">
          <div className="hero-kpi"><div className="hk-lbl">진행 중</div><div className="hk-val cond green">{liveGames.length}</div><div className="hk-sub">LIVE</div></div>
          <div className="hero-kpi"><div className="hk-lbl">예정</div><div className="hk-val cond">{previewGames.length}</div><div className="hk-sub">UPCOMING</div></div>
          <div className="hero-kpi"><div className="hk-lbl">종료</div><div className="hk-val cond">{finalGames.length}</div><div className="hk-sub">FINAL</div></div>
          <div className="hero-kpi"><div className="hk-lbl">총 경기</div><div className="hk-val cond">{games.length}</div><div className="hk-sub">TODAY</div></div>
        </div>
      </div>

      <div className="filterbar">
        {([['ALL','전체',games.length],['Live','진행 중',liveGames.length],['Preview','예정',previewGames.length],['Final','종료',finalGames.length]] as const).map(([f,l,c])=>(
          <button key={f} className={`chip ${filter===f?'active':''}`} onClick={()=>setFilter(f as any)}>{l}<span className="ct">{c}</span></button>
        ))}
        <span style={{marginLeft:'auto',fontFamily:'var(--f-mono)',fontSize:12,color:'var(--ink-2)',fontWeight:700}}>
          {liveGames.length>0?'⚡ 실시간 자동 업데이트':'🔄 60초마다 자동 갱신'}
        </span>
      </div>

      {loading&&<Spinner/>}
      {!loading&&games.length===0&&<div style={{padding:'24px 32px'}}><EmptyBox msg="오늘 경기 데이터가 없습니다"/></div>}

      {!loading&&games.length>0&&(
        <div className="live-board">
          <div className="live-table-head">
            {['⏱ 시각 · 구장','✈ 원정팀 (Away)','⚾ 점수','🏠 홈팀 (Home)','🤖 AI 승리 예측','📊 경기 현황'].map((h,i)=><div key={i} className="num" style={{textAlign:i===2?'center':i>=4?'center':'left'}}>{h}</div>)}
          </div>
          {Object.entries(grouped).map(([groupName, groupGames])=>(
            <div key={groupName}>
              <div className="live-group-head">
                {groupName==='진행 중'&&<span className="live-group-dot"/>}
                <span>{groupName==='진행 중'?'⚾ 진행 중':groupName==='예정'?'⏰ 예정':groupName==='종료'?'✅ 종료':groupName}</span>
                <em>{groupGames.length}경기</em>
              </div>
              {groupGames.map(g=>{
                const ld=lives[g.game_pk]
                const isLive=ld?.status==='Live'
                const isFinal=ld?.status==='Final'
                const hasScore=ld?.runs!=null
                const liveProb=ld?.live_home_prob
                const displayHomeProb = isLive && liveProb!=null ? liveProb : g.home_win_prob
                const displayAwayProb = 1 - displayHomeProb
                const pickHome=displayHomeProb>=.5
                const pickTeam=pickHome?g.home_team:g.away_team
                const pct=(Math.max(displayHomeProb,displayAwayProb)*100).toFixed(1)
                const homeWinning=!!(hasScore&&ld.runs.home>ld.runs.away)
                const awayWinning=!!(hasScore&&ld.runs.away>ld.runs.home)
                const kstTime=formatKstTime(g.game_datetime)
                const inning=inningLabel(ld)
                const flash=flashEvent[g.game_pk]
                const isExciting=flash&&EXCITING.has(flash)
                const delta=probDeltas[g.game_pk]
                const baseChg=isLive&&liveProb!=null&&g.home_win_prob!=null?liveProb-g.home_win_prob:null
                return (
                  <div key={g.game_pk} className="live-game-row" style={{position:'relative'}}>
                    {/* 이벤트 플래시 배너 */}
                    {flash&&(
                      <div style={{
                        position:'absolute',top:0,left:0,right:0,bottom:0,
                        display:'flex',alignItems:'center',justifyContent:'center',
                        pointerEvents:'none',zIndex:10,
                      }}>
                        <span style={{
                          fontFamily:'var(--f-cond)',fontWeight:700,
                          fontSize:isExciting?22:16,letterSpacing:'.06em',
                          color:isExciting?'var(--amber)':'var(--ink-1)',
                          background:'rgba(0,0,0,.72)',
                          padding:isExciting?'6px 18px':'4px 14px',
                          borderRadius:'var(--r-sm)',
                          animation:'flash-in .2s ease',
                          border:isExciting?'1.5px solid var(--amber)':'none',
                        }}>{toKoEvent(flash)}</span>
                      </div>
                    )}
                    <div className="live-time-cell">
                      {isLive
                        ? <span className="live-inning-now">⚾ {inning||'진행 중'}</span>
                        : isFinal
                        ? <><span className="num live-time">{kstTime}</span><span style={{fontFamily:'var(--f-mono)',fontSize:11,fontWeight:800,color:'var(--ink-4)',marginTop:1}}>경기 종료</span></>
                        : <span className="num live-time">{kstTime}</span>
                      }
                      {ld?.venue&&<span className="live-venue">🏟 {shortVenue(ld.venue)}</span>}
                    </div>
                    <div className="live-team away"><TM code={g.away_team} size="lg"/><div><div className="live-team-code" style={{opacity:isFinal&&homeWinning?.55:1}}>{g.away_team}</div><div className="live-team-name">{TEAMS[g.away_team]?.city} {TEAMS[g.away_team]?.name}</div>{isLive&&ld&&<div className="live-mini"><span>🥎 안타 {ld.hits.away}</span><span>⚠ 실책 {ld.errors.away}</span></div>}{isFinal&&ld?.pitchers?.loser&&!homeWinning&&<div className="live-pitcher win">🏆 {getLastName(ld.pitchers.winner)}</div>}{isFinal&&ld?.pitchers?.loser&&homeWinning&&<div className="live-pitcher loss">{getLastName(ld.pitchers.loser)}</div>}{!isFinal&&ld?.pitchers?.away_probable&&<div className="live-pitcher start">🎯 {getLastName(ld.pitchers.away_probable)}</div>}</div></div>
                    <div className="live-score-cell">{hasScore?(<><span className={`live-score-num ${awayWinning?'win':''}`}>{ld.runs.away}</span><span className="live-score-mid">:</span><span className={`live-score-num ${homeWinning?'win':''}`}>{ld.runs.home}</span>{isLive&&ld&&<div className="live-outs"><span className="live-outs-lbl">아웃</span>{[0,1,2].map(i=><span key={i} className={i<ld.outs?'on':''}/>)}</div>}</>):<span className="live-vs">vs</span>}</div>
                    <div className="live-team home"><div><div className="live-team-code" style={{opacity:isFinal&&awayWinning?.55:1}}>{g.home_team}</div><div className="live-team-name">{TEAMS[g.home_team]?.city} {TEAMS[g.home_team]?.name}</div>{isLive&&ld&&<div className="live-mini right"><span>🥎 안타 {ld.hits.home}</span><span>⚠ 실책 {ld.errors.home}</span></div>}{isFinal&&ld?.pitchers?.winner&&homeWinning&&<div className="live-pitcher win">🏆 {getLastName(ld.pitchers.winner)}</div>}{isFinal&&ld?.pitchers?.loser&&!homeWinning&&<div className="live-pitcher loss">{getLastName(ld.pitchers.loser)}</div>}{!isFinal&&ld?.pitchers?.home_probable&&<div className="live-pitcher start">🎯 {getLastName(ld.pitchers.home_probable)}</div>}</div><TM code={g.home_team} size="lg"/></div>
                    <div className="live-pred">
                      <div style={{display:'flex',alignItems:'center',justifyContent:'center',gap:6}}>
                        <div className="cond" style={{color:pickHome?'var(--navy)':'var(--red)',transition:'color .3s'}}>
                          {pickTeam} 승 <span style={{fontWeight:900}}>{pct}%</span>
                        </div>
                        {isLive&&delta!=null&&Math.abs(delta)>0.001&&(
                          <span className={`prob-delta ${delta>0?'up':'down'}`}>{delta>0?'▲ +':'▼ -'}{(Math.abs(delta)*100).toFixed(1)}%p</span>
                        )}
                      </div>
                      <PBar home={displayHomeProb} h={6}/>
                      {isLive&&liveProb!=null&&g.home_win_prob!=null&&(
                        <div style={{fontSize:11,color:'var(--ink-3)',marginTop:3,fontFamily:'var(--f-mono)',textAlign:'center',fontWeight:600}}>
                          예측 {(g.home_win_prob*100).toFixed(1)}%
                          {baseChg!=null&&Math.abs(baseChg)>0.002&&<span style={{marginLeft:5,fontWeight:800,fontSize:12,color:baseChg>0?'var(--green)':'var(--red)'}}>{baseChg>0?'↑+':'↓-'}{(Math.abs(baseChg)*100).toFixed(1)}%p</span>}
                        </div>
                      )}
                    </div>
                    <div className="live-status">
                      <Conf level={g.confidence}/>
                      {isLive&&ld
                        ? <>
                            <div style={{textAlign:'center',fontFamily:'var(--f-mono)',fontSize:11,color:'var(--ink-4)',fontWeight:700,letterSpacing:'.06em'}}>주루 현황</div>
                            <Diamond runners={ld.runners}/>
                          </>
                        : <span>{isFinal?'✅ 최종':'⏰ 대기 중'}</span>
                      }
                    </div>
                  </div>
                )
              })}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

/* ════════════════════════════════════
   SCREEN: STANDINGS
════════════════════════════════════ */
interface StandingEntry {
  team_id:number; team_name:string; city:string; abbr:string
  division:string; league:string; wins:number; losses:number; win_pct:number
  gb:string; streak:string; home_w:number; home_l:number; away_w:number; away_l:number
  l10_w:number; l10_l:number; div_rank:number; wc_rank:number; wc_gb:string
}
interface NewStandingsData {
  AL: Record<string,StandingEntry[]>
  NL: Record<string,StandingEntry[]>
  wildcard: {AL:StandingEntry[];NL:StandingEntry[]}
  error?: string
}

function DivTable({ teams, title, showWC=false }: { teams:StandingEntry[]; title:string; showWC?:boolean }) {
  if(!teams || teams.length===0) return null
  return (
    <div style={{marginBottom:24}}>
      {/* 디비전 헤더 */}
      <div style={{
        display:'flex', justifyContent:'space-between', alignItems:'baseline',
        padding:'10px 16px', background:'var(--navy)', borderRadius:'var(--r-md) var(--r-md) 0 0'
      }}>
        <span className="cond" style={{fontSize:16,fontWeight:700,color:'#fff',letterSpacing:'.04em',textTransform:'uppercase'}}>{title}</span>
        <span className="num" style={{fontSize:11,color:'rgba(255,255,255,.4)'}}>{teams.length}팀</span>
      </div>

      {/* 테이블 헤더 */}
      <div style={{
        display:'grid',
        gridTemplateColumns:'28px 180px 40px 40px 64px 64px 72px 72px 60px 60px',
        gap:0, padding:'8px 16px', background:'var(--surface-2)',
        borderLeft:'1px solid var(--rule)', borderRight:'1px solid var(--rule)'
      }}>
        {['#','팀','W','L','PCT','GB','HOME','AWAY','L10','연속'].map((h,i)=>(
          <span key={i} className="num" style={{
            fontSize:9, letterSpacing:'.12em', textTransform:'uppercase', color:'var(--ink-4)',
            textAlign: i>=2 ? 'center' : 'left'
          }}>{h}</span>
        ))}
      </div>

      {/* 팀 행 */}
      <div style={{border:'1px solid var(--rule)',borderTop:'none',borderRadius:'0 0 var(--r-md) var(--r-md)',overflow:'hidden',background:'var(--surface)'}}>
        {teams.map((t, i) => {
          const inPlayoff = i < 3
          const isFirst = i === 0
          const streakWin = String(t.streak).startsWith('W')
          const streakLoss = String(t.streak).startsWith('L')
          return (
            <div key={t.team_id} style={{
              display:'grid',
              gridTemplateColumns:'28px 180px 40px 40px 64px 64px 72px 72px 60px 60px',
              gap:0, padding:'12px 16px', alignItems:'center',
              borderBottom: i < teams.length-1 ? '1px solid var(--rule-soft)' : 'none',
              background: isFirst ? 'rgba(4,30,66,.025)' : 'transparent',
              borderLeft: `3px solid ${inPlayoff ? 'var(--navy)' : 'transparent'}`,
              marginLeft: inPlayoff ? 0 : 0,
            }}>
              {/* 순위 */}
              <span className="num" style={{
                fontSize:12, fontWeight:isFirst?700:400,
                color: isFirst ? 'var(--navy)' : inPlayoff ? 'var(--navy-3)' : 'var(--ink-4)'
              }}>{showWC ? t.wc_rank : i+1}</span>

              {/* 팀 */}
              <div style={{display:'flex',alignItems:'center',gap:8}}>
                <TM code={t.abbr} size="sm"/>
                <div>
                  <div className="cond" style={{fontSize:14,fontWeight:700,textTransform:'uppercase',lineHeight:1.1}}>{t.abbr}</div>
                  <div style={{fontSize:10,color:'var(--ink-4)',lineHeight:1}}>{t.city} {t.team_name}</div>
                </div>
              </div>

              {/* 통계 */}
              {[
                {v:String(t.wins), bold:true},
                {v:String(t.losses), bold:false},
                {v:t.win_pct.toFixed(3), bold:false},
                {v:showWC ? String(t.wc_gb) : String(t.gb), bold:false},
                {v:`${t.home_w}-${t.home_l}`, bold:false},
                {v:`${t.away_w}-${t.away_l}`, bold:false},
                {v:`${t.l10_w}-${t.l10_l}`, bold:false},
                {v:String(t.streak), bold:false, color: streakWin?'var(--green)':streakLoss?'var(--red)':'var(--ink)'},
              ].map((s,j)=>(
                <div key={j} style={{textAlign:'center'}}>
                  <span className="num" style={{
                    fontSize:13, fontWeight:s.bold?700:400,
                    color: s.color || 'var(--ink)'
                  }}>{s.v||'—'}</span>
                </div>
              ))}
            </div>
          )
        })}
      </div>
    </div>
  )
}

function ScreenStandings() {
  const [data,setData]=useState<NewStandingsData|null>(null)
  const [loading,setLoading]=useState(true)
  const [lg,setLg]=useState<'AL'|'NL'>('AL')
  const [tab,setTab]=useState<'division'|'wildcard'>('division')

  useEffect(()=>{
    fetch(`${API}/standings`)
      .then(r=>r.json())
      .then(d=>{setData(d);setLoading(false)})
      .catch(()=>setLoading(false))
  },[])

  const divData = data?.[lg] ?? {East:[],Central:[],West:[]}
  const wcData  = data?.wildcard?.[lg] ?? []
  const alTotal = Object.values(data?.AL ?? {}).reduce((a,b)=>a+b.length,0)
  const nlTotal = Object.values(data?.NL ?? {}).reduce((a,b)=>a+b.length,0)

  return (
    <div className="view-enter">
      <div className="subhead">
        <div>
          <div className="kicker">STANDINGS · 2026 MLB 시즌 순위표</div>
          <h1 className="page-title">팀 <span className="red">순위</span></h1>
        </div>
        <div className="hero-kpis">
          <div className="hero-kpi"><div className="hk-lbl">AL 팀</div><div className="hk-val cond">{alTotal||'—'}</div><div className="hk-sub">TEAMS</div></div>
          <div className="hero-kpi"><div className="hk-lbl">NL 팀</div><div className="hk-val cond">{nlTotal||'—'}</div><div className="hk-sub">TEAMS</div></div>
        </div>
      </div>

      <div className="filterbar">
        <span className="lbl" style={{marginRight:4}}>리그</span>
        <button className={`chip ${lg==='AL'?'active':''}`} onClick={()=>setLg('AL')}>AL 리그</button>
        <button className={`chip ${lg==='NL'?'active':''}`} onClick={()=>setLg('NL')}>NL 리그</button>
        <span style={{width:1,height:16,background:'var(--rule)',margin:'0 8px'}}/>
        <span className="lbl" style={{marginRight:4}}>보기</span>
        <button className={`chip ${tab==='division'?'active':''}`} onClick={()=>setTab('division')}>디비전별</button>
        <button className={`chip ${tab==='wildcard'?'active':''}`} onClick={()=>setTab('wildcard')}>와일드카드</button>
      </div>

      {loading && <Spinner/>}

      {!loading && data?.error && (
        <div style={{margin:'24px 32px',padding:20,background:'#FEF3F2',border:'1px solid #FCA5A5',borderRadius:'var(--r-md)'}}>
          <div className="cond" style={{fontSize:15,fontWeight:700,color:'var(--red)'}}>순위 데이터 로딩 실패</div>
          <div className="num" style={{fontSize:11,color:'var(--ink-3)',marginTop:6}}>{data.error}</div>
        </div>
      )}

      {!loading && (
        <div style={{padding:'20px 32px 48px'}}>
          {/* 플레이오프 진출권 안내 */}
          <div style={{display:'flex',alignItems:'center',gap:20,padding:'10px 16px',background:'var(--surface)',border:'1px solid var(--rule)',borderRadius:'var(--r-md)',marginBottom:20}}>
            <div style={{display:'flex',alignItems:'center',gap:8}}>
              <div style={{width:14,height:14,borderRadius:2,background:'var(--navy)'}}/>
              <span style={{fontSize:12,color:'var(--ink-2)'}}>디비전 1위 — 플레이오프 자동 진출</span>
            </div>
            <div style={{display:'flex',alignItems:'center',gap:8}}>
              <div style={{width:14,height:14,borderRadius:2,background:'rgba(4,30,66,.25)'}}/>
              <span style={{fontSize:12,color:'var(--ink-2)'}}>2–3위 — 와일드카드 진출권 (리그별 상위 3팀)</span>
            </div>
          </div>

          {tab==='division' && (
            <div>
              <DivTable teams={divData['East']??[]} title={`${lg} East · 동부지구`}/>
              <DivTable teams={divData['Central']??[]} title={`${lg} Central · 중부지구`}/>
              <DivTable teams={divData['West']??[]} title={`${lg} West · 서부지구`}/>
              {alTotal===0 && nlTotal===0 && (
                <EmptyBox msg="순위 데이터를 불러오는 중입니다" sub="MLB Stats API에서 데이터를 가져옵니다. 잠시 후 새로고침 해주세요."/>
              )}
            </div>
          )}

          {tab==='wildcard' && (
            <div>
              <div style={{padding:'12px 16px',background:'var(--surface)',border:'1px solid var(--rule)',borderRadius:'var(--r-md)',marginBottom:16,fontSize:12,color:'var(--ink-2)'}}>
                <strong>와일드카드</strong> — 각 리그 디비전 1위 3팀을 제외한 나머지 팀 중 승률 기준 상위 3팀이 포스트시즌에 진출합니다.
              </div>
              <DivTable teams={wcData} title={`${lg} 와일드카드`} showWC={true}/>
              {wcData.length===0 && <EmptyBox msg="와일드카드 데이터가 없습니다" sub="디비전별 탭에서 데이터가 로딩된 후 확인해주세요."/>}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function ScreenModel() {
  const [data,setData]=useState<HistData|null>(null)
  useEffect(()=>{fetch(`${API}/predictions/history?days=30`).then(r=>r.json()).then(setData).catch(()=>{})},[])
  const rows=data?.rows??[]
  const graded=rows.filter(r=>r.is_correct!==null)
  const byConf=(['HIGH','MED','LOW'] as const).map(lv=>{const b=graded.filter(r=>r.confidence===lv),h=b.filter(r=>r.is_correct===1);return{level:lv,n:b.length,correct:h.length,acc:b.length>0?h.length/b.length:0}})
  return (
    <div className="view-enter">
      <div className="subhead">
        <div><div className="kicker">MODEL PERFORMANCE · 공개 트랙 레코드 · 최근 30일</div><h1 className="page-title">모델 <span className="red">성능</span></h1></div>
        <div style={{fontFamily:'var(--f-mono)',fontSize:11,color:'var(--ink-3)',textAlign:'right',lineHeight:1.8}}>학습: 2023–2024 시즌<br/>운영: 2026.04~<br/>VERSION v1/v2</div>
      </div>
      <div style={{padding:'24px 32px 0'}}>
        <div className="kpi-grid" style={{marginBottom:20}}>
          {[{lbl:'BRIER SCORE',v:data?.brier,t:.23,lower:true,fmt:(x:number)=>x.toFixed(3)},{lbl:'ACCURACY',v:data?.accuracy!=null?data.accuracy/100:null,t:.55,lower:false,fmt:(x:number)=>`${(x*100).toFixed(1)}%`},{lbl:'총 예측',v:data?.total,t:null,lower:null,fmt:(x:number)=>String(x)},{lbl:'GRADED',v:data?.graded,t:null,lower:null,fmt:(x:number)=>String(x)}].map(k=>{
            const pass=k.t!=null&&k.lower!=null&&k.v!=null?(k.lower?k.v<k.t:k.v>k.t):null
            return (
              <div key={k.lbl} className="kpi-cell">
                <div className="kpi-lbl">{k.lbl}{pass!==null&&<span style={{width:8,height:8,borderRadius:'50%',background:pass?'var(--green)':'var(--red)',marginLeft:'auto',display:'inline-block'}}/>}</div>
                <div className="kpi-val" style={{color:pass===true?'var(--green)':pass===false?'var(--red)':'var(--ink-4)'}}>{k.v!=null?k.fmt(k.v):'—'}</div>
                {k.t!=null&&<div className="kpi-sub">목표 {k.lower?'<':'>'} {k.fmt(k.t)} · <span style={{color:pass?'var(--green)':'var(--red)',fontWeight:700}}>{pass?'PASS':'MISS'}</span></div>}
              </div>
            )
          })}
        </div>
        <div className="panel" style={{padding:'20px 24px',marginBottom:20}}>
          <h2 className="sec-title" style={{marginBottom:16}}>신뢰도 구간별 정확도</h2>
          <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:14}}>
            {byConf.map(b=>{const color=b.level==='HIGH'?'var(--navy)':b.level==='MED'?'var(--ink-3)':'var(--red)';return(
              <div key={b.level} style={{border:'1px solid var(--rule)',borderRadius:'var(--r-md)',padding:'16px 18px'}}>
                <div style={{display:'flex',alignItems:'center',gap:8}}><Conf level={b.level}/><span className="num" style={{fontSize:11,color:'var(--ink-3)',marginLeft:'auto'}}>n={b.n}</span></div>
                <div className="cond" style={{fontSize:36,fontWeight:700,marginTop:10,lineHeight:1,color}}>{b.n>0?`${(b.acc*100).toFixed(1)}%`:'—'}</div>
                <div className="num" style={{fontSize:11,color:'var(--ink-3)',marginTop:5}}>{b.correct} HIT · {b.n-b.correct} MISS</div>
                <div style={{marginTop:10,height:4,background:'var(--rule-soft)',borderRadius:1,overflow:'hidden'}}><div style={{width:`${b.acc*100}%`,height:'100%',background:color}}/></div>
              </div>
            )})}
          </div>
        </div>
        <div className="panel dark" style={{padding:'24px 26px',marginBottom:40}}>
          <div className="lbl on-dark" style={{marginBottom:16}}>TECH STACK</div>
          <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:0}}>
            {[{cat:'ML',items:['LightGBM','XGBoost','scikit-learn','SHAP','Isotonic Cal.']},{cat:'DATA',items:['pybaseball','MLB Stats API','Baseball Ref','pandas','numpy']},{cat:'BACKEND',items:['FastAPI','SQLAlchemy','PostgreSQL','Redis','uvicorn']},{cat:'FRONTEND',items:['Next.js 14','TypeScript','Barlow Cond.','JetBrains Mono']}].map((col,i)=>(
              <div key={col.cat} style={{padding:'0 18px',borderLeft:i?'1px solid rgba(255,255,255,.08)':'none'}}>
                <div style={{fontFamily:'var(--f-mono)',fontSize:9,letterSpacing:'.18em',color:'var(--red)',marginBottom:10}}>{col.cat}</div>
                {col.items.map(it=><div key={it} className="cond" style={{fontSize:14,fontWeight:600,color:'#C8D8EC',marginBottom:5,letterSpacing:'.02em'}}>{it}</div>)}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

/* ════════════════════════════════════
   SCREEN: MY PAGE
════════════════════════════════════ */
function ScreenMy({auth}:{auth:AuthState}) {
  const [info,setInfo]=useState<UserInfo|null>(null)
  const [picks,setPicks]=useState<UserPickItem[]>([])
  const [loading,setLoading]=useState(true)
  const [tab,setTab]=useState<'overview'|'picks'>('overview')

  useEffect(()=>{
    if(!auth.token) return
    Promise.all([
      fetch(`${API}/users/me`,{headers:{'Authorization':`Bearer ${auth.token}`}}).then(r=>r.json()),
      fetch(`${API}/users/me/picks`,{headers:{'Authorization':`Bearer ${auth.token}`}}).then(r=>r.json()),
    ]).then(([me,pk])=>{setInfo(me);setPicks(pk.picks||[]);setLoading(false)}).catch(()=>setLoading(false))
  },[auth.token])

  if(!auth.token) return <div style={{padding:'60px 32px',textAlign:'center'}}><EmptyBox msg="로그인이 필요합니다" sub="상단의 로그인 버튼을 눌러주세요"/></div>

  return (
    <div className="view-enter">
      <div className="my-hero">
        <div className="lbl on-dark" style={{marginBottom:8}}>MY PAGE · 나의 픽 기록</div>
        <div className="cond" style={{fontSize:30,fontWeight:700,textTransform:'uppercase'}}>{auth.username}</div>
        {info&&(
          <div className="my-stats">
            {[['총 픽',info.total,''],['채점 완료',info.graded,'GRADED'],['적중',info.correct,'HIT'],['적중률',info.accuracy!=null?`${info.accuracy}%`:'—','ACCURACY'],['연속 적중',info.streak,'STREAK']].map(([l,v,s],i)=>(
              <div key={i} className="my-stat">
                <div className="msl">{l}</div>
                <div className={`msv ${i===3&&info.accuracy!=null?(info.accuracy>=55?'green':info.accuracy>=45?'amber':'red'):''}  ${i===4&&info.streak>0?'green':''}`}>{v}</div>
                {s&&<div style={{fontFamily:'var(--f-mono)',fontSize:9,color:'rgba(255,255,255,.4)',marginTop:3,letterSpacing:'.12em'}}>{s}</div>}
              </div>
            ))}
          </div>
        )}
      </div>

      {info&&(
        <div style={{background:'var(--surface)',borderBottom:'1px solid var(--rule)',padding:'12px 32px'}}>
          <div style={{display:'flex',gap:20}}>
            {(['HIGH','MED','LOW'] as const).map(lv=>{const b=info.by_conf[lv];const color=lv==='HIGH'?'var(--navy)':lv==='MED'?'var(--ink-3)':'var(--red)';return(
              <div key={lv} style={{display:'flex',alignItems:'center',gap:10,flex:1,maxWidth:280}}>
                <Conf level={lv}/>
                <div style={{flex:1}}>
                  <div style={{height:5,background:'var(--rule-soft)',borderRadius:1,overflow:'hidden'}}><div style={{width:`${(b.acc||0)}%`,height:'100%',background:color}}/></div>
                  <div className="num" style={{fontSize:10,color:'var(--ink-3)',marginTop:3}}>{b.correct}/{b.n} · <strong>{b.acc!=null?`${b.acc}%`:'—'}</strong></div>
                </div>
              </div>
            )})}
          </div>
        </div>
      )}

      <div style={{background:'var(--surface)',padding:'0 32px',borderBottom:'1px solid var(--rule)'}}>
        <div className="tabs">
          {[{id:'overview',label:'개요'},{id:'picks',label:`픽 기록 (${picks.length})`}].map(t=>(
            <button key={t.id} className={`tab ${tab===t.id?'active':''}`} onClick={()=>setTab(t.id as any)}>{t.label}</button>
          ))}
        </div>
      </div>

      <div style={{padding:'24px 32px 48px'}}>
        {loading&&<Spinner/>}
        {!loading&&tab==='overview'&&info&&(
          <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:16}}>
            {[['총 픽',String(info.total),'전체 제출한 픽 수'],['채점 완료',String(info.graded),'결과가 확정된 경기 수'],['적중',String(info.correct),'맞춘 경기 수'],['적중률',info.accuracy!=null?`${info.accuracy}%`:'—','전체 적중률'],['연속 적중',String(info.streak),'현재 연속 적중 수'],['미채점',String(info.total-info.graded),'결과 대기 중']].map(([k,v,d])=>(
              <div key={k} className="panel" style={{padding:'18px 20px'}}>
                <div className="lbl" style={{marginBottom:6}}>{k}</div>
                <div className="cond" style={{fontSize:32,fontWeight:700,lineHeight:1}}>{v}</div>
                <div style={{fontSize:11,color:'var(--ink-4)',marginTop:6}}>{d}</div>
              </div>
            ))}
          </div>
        )}
        {!loading&&tab==='picks'&&(
          picks.length===0?<EmptyBox msg="픽 기록이 없습니다" sub="오늘 예측 탭에서 경기를 픽해보세요"/>:(
            <div style={{display:'flex',flexDirection:'column',gap:8}}>
              {picks.map(p=>(
                <div key={p.id} className={`panel ${p.is_correct===1?'hit':p.is_correct===0?'miss':'pend'}`} style={{padding:'14px 18px',borderLeft:p.is_correct===1?'3px solid var(--green)':p.is_correct===0?'3px solid var(--red)':'3px solid var(--rule)'}}>
                  <div style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}>
                    <div style={{display:'flex',alignItems:'center',gap:10}}>
                      <TM code={p.away_team} size="sm"/>
                      <span className="cond" style={{fontSize:13,color:'var(--ink-3)'}}>vs</span>
                      <TM code={p.home_team} size="sm"/>
                      <div>
                        <div className="num" style={{fontSize:11,color:'var(--ink-4)',letterSpacing:'.06em'}}>{p.game_date}</div>
                        <div className="cond" style={{fontSize:14,fontWeight:700,textTransform:'uppercase'}}>
                          내 픽: <span style={{color:p.pick_team===p.home_team?'var(--navy)':'var(--red)'}}>{p.pick_team}</span>
                          {p.pick_prob&&<span style={{color:'var(--ink-3)',fontWeight:400,marginLeft:6}}>{(p.pick_prob*100).toFixed(1)}%</span>}
                        </div>
                      </div>
                    </div>
                    <div style={{display:'flex',alignItems:'center',gap:8}}>
                      <Conf level={p.confidence}/>
                      <RC v={p.is_correct}/>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )
        )}
      </div>
    </div>
  )
}

/* ════════════════════════════════════
   APP ROOT
════════════════════════════════════ */
export default function App() {
  const [view,setView]=useState<'today'|'live'|'schedule'|'standings'|'model'|'my'>('today')
  const [gameCount,setGameCount]=useState(0)
  const [auth,setAuth]=useState<AuthState>({token:null,username:null,userId:null})
  const [authModal,setAuthModal]=useState<'login'|'register'|null>(null)

  // 토큰 로컬 복원
  useEffect(()=>{
    const t=localStorage.getItem('simmlb_token')
    const u=localStorage.getItem('simmlb_username')
    const id=localStorage.getItem('simmlb_userid')
    if(t&&u&&id) setAuth({token:t,username:u,userId:parseInt(id)})
  },[])

  useEffect(()=>{
    fetch(`${API}/predictions/today`).then(r=>r.json()).then(d=>setGameCount(d?.games?.length??0)).catch(()=>{})
  },[])

  const handleAuthSuccess=(token:string,username:string,userId:number)=>{
    localStorage.setItem('simmlb_token',token)
    localStorage.setItem('simmlb_username',username)
    localStorage.setItem('simmlb_userid',String(userId))
    setAuth({token,username,userId})
    setAuthModal(null)
  }
  const handleLogout=()=>{
    localStorage.removeItem('simmlb_token')
    localStorage.removeItem('simmlb_username')
    localStorage.removeItem('simmlb_userid')
    setAuth({token:null,username:null,userId:null})
    if(view==='my') setView('today')
  }

  return (
    <div className="app">
      <Topbar view={view} setView={v=>setView(v as any)} gameCount={gameCount} auth={auth} onAuthClick={m=>setAuthModal(m)} onLogout={handleLogout}/>
      {view==='today'    &&<ScreenToday auth={auth}/>}
      {view==='live'     &&<ScreenLive/>}
      {view==='schedule' &&<ScreenSchedule/>}
      {view==='standings'&&<ScreenStandings/>}
      {view==='model'    &&<ScreenModel/>}
      {view==='my'       &&<ScreenMy auth={auth}/>}
      {authModal&&<AuthModal mode={authModal} onClose={()=>setAuthModal(null)} onSuccess={handleAuthSuccess}/>}
    </div>
  )
}
