"""Builds dashboard/index.html: one self-contained page (no CDN, no server) from results/*.json."""
import glob
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results")
KEEP_STEP = ("step", "reward", "correct", "tool_calls", "absn", "absp", "absh", "abstot", "absn_fresh", "n_fresh",
             "clip", "w", "items", "stale_items", "rerendered_items", "dropped_nonrenderable", "dropped_old_harness",
             "elapsed")


def main():
    runs = []
    for p in sorted(glob.glob(os.path.join(RES, "*_s*.json"))):
        r = json.load(open(p))
        runs.append({"arm": r["config"]["arm"], "seed": r["config"]["seed"], "config": r["config"],
                     "steps": [{k: s.get(k) for k in KEEP_STEP} for s in r["steps"]], "evals": r["evals"],
                     "edits": r["edits"], "final": r.get("final"), "time": r["time"], "demo": r.get("demo")})
    summ = json.load(open(os.path.join(RES, "summary.json"))) if os.path.exists(os.path.join(RES, "summary.json")) else {}
    html = TEMPLATE.replace("/*DATA*/null", json.dumps({"runs": runs, "summary": summ}))
    out = os.path.join(ROOT, "dashboard", "index.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write(html)
    print(out, f"{len(runs)} runs")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>harnesslag / runs</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#070707;--p:#0f0f0f;--p2:#151515;--b:#232323;--b2:#2e2e2e;--t:#ececec;--d:#8c8c8c;--dd:#555;
--mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace;--sans:Inter,"Helvetica Neue",system-ui,sans-serif}
*{box-sizing:border-box}html,body{margin:0;background:var(--bg);color:var(--t);font:13px/1.4 var(--sans)}
.app{display:grid;grid-template-columns:56px 232px 1fr;min-height:100vh}
.rail{border-right:1px solid var(--b);display:flex;flex-direction:column;align-items:center;gap:14px;padding:14px 0}
.rail .logo{width:28px;height:28px;border:1.5px solid var(--t);border-radius:7px;display:grid;place-items:center;font:700 13px var(--mono)}
.rail i{width:34px;height:34px;border-radius:9px;display:grid;place-items:center;color:var(--d);font-style:normal;font:12px var(--mono)}
.rail i.on{background:var(--p2);color:var(--t);border:1px solid var(--b2)}
.side{border-right:1px solid var(--b);padding:16px 14px;display:flex;flex-direction:column;gap:18px;background:#0a0a0a}
.side h2{font-size:14px;margin:0;font-weight:600}.lbl{font:10px var(--mono);color:var(--d);text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px}
.seg{display:flex;background:#000;border:1px solid var(--b);border-radius:9px;padding:3px}
.seg button{flex:1;background:none;border:0;color:var(--d);font:11px var(--mono);padding:6px 0;border-radius:6px;cursor:pointer}
.seg button.on{background:var(--p2);color:var(--t)}
.arm{display:flex;align-items:center;gap:8px;padding:7px 8px;border-radius:8px;cursor:pointer;color:var(--d)}
.arm.on{background:var(--p2);color:var(--t)}.arm .sw{width:18px;height:0;border-top:2px solid}.arm .n{margin-left:auto;font:10px var(--mono);
background:#000;border:1px solid var(--b);border-radius:5px;padding:1px 6px}
.main{display:flex;flex-direction:column;min-width:0}
.top{height:52px;border-bottom:1px solid var(--b);display:flex;align-items:center;gap:14px;padding:0 20px}
.top .crumb{font-weight:600}.top .tag{font:10px var(--mono);color:var(--d);border:1px solid var(--b2);border-radius:20px;padding:3px 9px}
.top .sp{flex:1}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:12px;padding:16px 20px 24px}
.card{background:var(--p);border:1px solid var(--b);border-radius:12px;padding:12px 14px;min-width:0}
.card h3{margin:0 0 8px;font:500 11px var(--mono);color:var(--d);text-transform:uppercase;letter-spacing:.07em;display:flex;gap:8px}
.card h3 b{margin-left:auto;color:var(--dd);font-weight:400}
.kpi .v{font:600 26px var(--mono);letter-spacing:-.02em}.kpi .s{font:11px var(--mono);color:var(--d);margin-top:2px}
.kpi .bars{display:flex;gap:2px;align-items:flex-end;height:22px;margin-top:8px}.kpi .bars span{flex:1;background:#3a3a3a;border-radius:1px}
.c3{grid-column:span 3}.c4{grid-column:span 4}.c5{grid-column:span 5}.c6{grid-column:span 6}.c7{grid-column:span 7}.c8{grid-column:span 8}.c12{grid-column:span 12}
svg{display:block;width:100%}svg text{font:10px var(--mono);fill:var(--d)}
table{width:100%;border-collapse:collapse;font:11px var(--mono)}th{color:var(--dd);font-weight:400;text-align:left;padding:5px 6px;border-bottom:1px solid var(--b)}
td{padding:5px 6px;border-bottom:1px solid #191919;color:var(--d)}td.t{color:var(--t)}tr.acc td{color:var(--t)}
.pill{border:1px solid var(--b2);border-radius:4px;padding:0 5px}.pill.w{background:var(--t);color:#000;border-color:var(--t)}
.legend{display:flex;gap:14px;font:10px var(--mono);color:var(--d);margin-top:6px;flex-wrap:wrap}.legend span:before{content:"";display:inline-block;width:14px;margin-right:5px;vertical-align:middle;border-top:2px solid var(--c)}
.tokens{display:flex;flex-wrap:wrap;gap:1px;font:11px var(--mono);max-height:190px;overflow:auto}
.tokens span{padding:1px 2px;border-radius:2px;white-space:pre}
.ctx{font:10px/1.35 var(--mono);color:var(--d);white-space:pre-wrap;background:#000;border:1px solid var(--b);border-radius:8px;padding:8px;max-height:120px;overflow:auto}
.ctx em{color:var(--t);font-style:normal;background:#262626}
.two{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.foot{font:10px var(--mono);color:var(--dd);padding:0 20px 20px}
</style></head><body>
<div class="app">
 <nav class="rail"><div class="logo">h</div><i class="on">≡</i><i>∿</i><i>⌗</i><i>◎</i></nav>
 <aside class="side">
  <h2>harnesslag</h2>
  <div><div class="lbl">seed</div><div class="seg" id="seeds"></div></div>
  <div><div class="lbl">arms</div><div id="arms"></div></div>
  <div><div class="lbl">mismatch panel</div><div class="seg" id="marm"></div></div>
  <div><div class="lbl">run</div><div id="runinfo" style="font:11px var(--mono);color:var(--d);line-height:1.7"></div></div>
 </aside>
 <main class="main">
  <div class="top"><span class="crumb">runs</span><span style="color:var(--dd)">/</span><span>gsm8k · calculator · grpo</span>
   <span class="tag" id="model"></span><span class="tag" id="label"></span><span class="sp"></span><span class="tag" id="nruns"></span></div>
  <div class="grid" id="grid"></div>
  <div class="foot">every number on this page is read from results/*.json produced by real runs; rebuild with python scripts/dashboard.py</div>
 </main>
</div>
<script>
const D=/*DATA*/null;
const NAMES={A:"discard stale",B:"naive reuse",C:"re-render, no IS",D:"harnesslag",E:"re-render, one cap"};
const COL={A:"#ffffff",B:"#7d7d7d",C:"#b5b5b5",D:"#ffffff",E:"#5e5e5e"};
const DASH={A:"",B:"2 3",C:"6 3",D:"",E:"8 3 2 3"};
const WID={A:1.2,B:1.2,C:1.2,D:2.4,E:1.2};
const st={seed:"all",arms:new Set("ABCDE"),marm:"D"};
const runs=D.runs, seeds=[...new Set(runs.map(r=>r.seed))].sort();
const el=(t,a={},h="")=>{const e=document.createElement(t);Object.entries(a).forEach(([k,v])=>e.setAttribute(k,v));e.innerHTML=h;return e};
const f=(x,d=2)=>x==null||isNaN(x)?"–":(+x).toFixed(d);
const mean=a=>a.length?a.reduce((s,x)=>s+x,0)/a.length:NaN;
function sel(arm){return runs.filter(r=>r.arm===arm&&(st.seed==="all"||r.seed==+st.seed))}
function ci(xs){if(xs.length<2)return NaN;const m=mean(xs),sd=Math.sqrt(xs.reduce((s,x)=>s+(x-m)**2,0)/(xs.length-1));const t={2:12.706,3:4.303,4:3.182,5:2.776}[xs.length]||2;return t*sd/Math.sqrt(xs.length)}
function chart(series,o={}){
 const W=o.w||640,H=o.h||220,P={l:42,r:10,t:10,b:24};const xs=series.flatMap(s=>s.pts.map(p=>p[0])),ys=series.flatMap(s=>s.pts.map(p=>p[1])).filter(y=>y!=null&&isFinite(y)&&(!o.log||y>0));
 if(!xs.length||!ys.length)return `<svg viewBox="0 0 ${W} ${H}"><text x="${W/2}" y="${H/2}" text-anchor="middle">waiting for data</text></svg>`;
 let x0=Math.min(...xs),x1=Math.max(...xs),y0=o.log?Math.min(...ys):Math.min(0,...ys),y1=Math.max(...ys);if(x1==x0)x1=x0+1;if(y1==y0)y1=y0+1;
 const ly=v=>o.log?Math.log10(v):v;const Y0=ly(y0),Y1=ly(y1)+(ly(y1)-ly(y0))*0.08;
 const X=v=>P.l+(v-x0)/(x1-x0)*(W-P.l-P.r),Yp=v=>H-P.b-(ly(v)-Y0)/(Y1-Y0)*(H-P.t-P.b);
 let g="";for(let i=0;i<=4;i++){const v=o.log?10**(Y0+(Y1-Y0)*i/4):y0+(Y1-y0)*i/4;const y=Yp(v);g+=`<line x1="${P.l}" x2="${W-P.r}" y1="${y}" y2="${y}" stroke="#1c1c1c"/><text x="${P.l-6}" y="${y+3}" text-anchor="end">${o.log?v.toExponential(0):f(v*(o.pct?100:1),(Y1-y0)*(o.pct?100:1)<5?2:1)}</text>`}
 for(let i=0;i<=5;i++){const v=x0+(x1-x0)*i/5;g+=`<text x="${X(v)}" y="${H-6}" text-anchor="middle">${x1-x0<10?f(v,1):Math.round(v)}</text>`}
 (o.marks||[]).forEach(m=>{g+=`<line x1="${X(m.x)}" x2="${X(m.x)}" y1="${P.t}" y2="${H-P.b}" stroke="#fff" stroke-opacity=".35" stroke-dasharray="2 2"/><text x="${X(m.x)+3}" y="${P.t+9}" fill="#bbb">${m.t}</text>`});
 series.forEach(s=>{if(s.band){const up=s.band.map(p=>`${X(p[0])},${Yp(p[2])}`),dn=s.band.slice().reverse().map(p=>`${X(p[0])},${Yp(p[1])}`);g+=`<polygon points="${up.concat(dn).join(" ")}" fill="${s.c}" fill-opacity=".07"/>`}
  const pts=s.pts.filter(p=>p[1]!=null&&isFinite(p[1])&&(!o.log||p[1]>0));if(!pts.length)return;
  g+=`<polyline fill="none" stroke="${s.c}" stroke-width="${s.w||1.3}" stroke-dasharray="${s.dash||""}" points="${pts.map(p=>X(p[0])+","+Yp(p[1])).join(" ")}"/>`;
  if(s.dots)pts.forEach(p=>g+=`<circle cx="${X(p[0])}" cy="${Yp(p[1])}" r="2.4" fill="#000" stroke="${s.c}"/>`)});
 return `<svg viewBox="0 0 ${W} ${H}">${g}<text x="${W-P.r}" y="${H-P.b-4}" text-anchor="end" fill="#555">${o.xl||""}</text></svg>`}
function diff(a,b,k){return Object.keys(a).filter(x=>a[x]!==b[x]).map(x=>x+"="+(k?b[x]:a[x])).join(", ")}
function legend(items){return `<div class="legend">${items.map(([n,c])=>`<span style="--c:${c}">${n}</span>`).join("")}</div>`}
function card(cls,title,body,aside=""){return `<div class="card ${cls}"><h3>${title}<b>${aside}</b></h3>${body}</div>`}
function editMarks(seed){const a=runs.find(r=>r.arm==="A"&&r.seed==seed);return a?a.edits.filter(e=>e.harness).map(e=>({x:e.step,t:"edit"})):[]}
function render(){
 const g=document.getElementById("grid");let h="";const S=D.summary||{};
 // KPIs
 const fin=a=>sel(a).filter(r=>r.final).map(r=>r.final.acc*100);
 ["A","B","C","D","E"].forEach(a=>{if(a==="E")return;});
 const kp=(t,v,s,bars)=>card("c3 kpi",t,`<div class="v">${v}</div><div class="s">${s}</div>${bars?`<div class="bars">${bars.map(b=>`<span style="height:${Math.max(2,b*100)}%"></span>`).join("")}</div>`:""}`);
 const fd=fin("D"),fa=fin("A"),fb=fin("B");
 const dr=sel("D").flatMap(r=>r.steps.map(s=>s.reward));
 h+=kp("harnesslag final acc",fd.length?f(mean(fd))+"%":"–",fd.length?`± ${f(ci(fd))} · ${fd.length} seed(s) · n=${sel("D").find(r=>r.final)?.final.n}`:"running",
   sel("D")[0]?sel("D")[0].steps.map(s=>s.reward/1.1):null);
 h+=kp("vs discard stale (A)",fd.length&&fa.length?(mean(fd)-mean(fa)>=0?"+":"")+f(mean(fd)-mean(fa))+" pt":"–",fa.length?`A = ${f(mean(fa))}% · B = ${fb.length?f(mean(fb)):"–"}%`:"running");
 const hy=S.hypothesis||{};
 h+=kp("fresh rollouts saved",hy.rollout_saving!=null?f(100*hy.rollout_saving,0)+"%":"none",hy.A_rollouts?`D reaches A's final curve acc at ${hy.D_rollouts_to_match_A??"never"} / ${hy.A_rollouts} · noisy n=200 curve, not significant`:"needs A and D curves");
 const ed=runs.filter(r=>r.arm==="A").flatMap(r=>r.edits);
 const tsec=runs.reduce((s,r)=>s+r.time.total*(r.config.threads||1),0);
 h+=kp("accepted harness edits",`${ed.filter(e=>e.harness).length}<span style="font-size:14px;color:#777"> / ${ed.length}</span>`,`edit points across ${runs.filter(r=>r.arm==="A").length} seeds · ${f(tsec/3600,1)} CPU-h total`);
 // accuracy curves
 const ser=[];[..."ABCDE"].filter(a=>st.arms.has(a)).forEach(a=>{const rs=sel(a);if(!rs.length)return;const n=Math.min(...rs.map(r=>r.evals.length));
  const pts=[],band=[];for(let i=0;i<n;i++){const v=rs.map(r=>r.evals[i].acc);pts.push([rs[0].evals[i].fresh_rollouts,mean(v)]);band.push([rs[0].evals[i].fresh_rollouts,Math.min(...v),Math.max(...v)])}
  ser.push({pts,band:rs.length>1?band:null,c:COL[a],dash:DASH[a],w:WID[a],dots:true})});
 h+=card("c8","accuracy vs fresh rollouts",chart(ser,{h:240,w:760,pct:true,xl:"fresh rollouts",marks:st.seed==="all"?[]:editMarks(+st.seed)})+legend([..."ABCDE"].filter(a=>st.arms.has(a)).map(a=>[a+" "+NAMES[a],COL[a]])),
   `test subset n=${runs[0]?.config.test_n} · greedy · mean${st.seed==="all"?" of seeds, band min..max":""}`);
 // final bars
 let bars="";const W=360,H=240;const vals=[..."ABCDE"].map(a=>({a,v:fin(a)}));const mx=Math.max(1,...vals.flatMap(x=>x.v))*1.25;
 vals.forEach((x,i)=>{const bx=30+i*66,m=mean(x.v),c=ci(x.v),yy=v=>H-28-(v/mx)*(H-46);if(x.v.length){bars+=`<rect x="${bx}" y="${yy(m)}" width="36" height="${H-28-yy(m)}" rx="3" fill="${x.a==="D"?"#fff":"#333"}"/>`;
  if(isFinite(c))bars+=`<line x1="${bx+18}" x2="${bx+18}" y1="${yy(m+c)}" y2="${yy(Math.max(0,m-c))}" stroke="#999"/><line x1="${bx+12}" x2="${bx+24}" y1="${yy(m+c)}" y2="${yy(m+c)}" stroke="#999"/>`;
  x.v.forEach((v,j)=>bars+=`<circle cx="${bx+10+j*8}" cy="${yy(v)}" r="2.5" fill="#000" stroke="#ddd"/>`);bars+=`<text x="${bx+18}" y="${yy(m)-6}" text-anchor="middle" fill="#ddd">${f(m)}</text>`}
  bars+=`<text x="${bx+18}" y="${H-12}" text-anchor="middle">${x.a}</text>`});
 h+=card("c4","final test accuracy %",`<svg viewBox="0 0 ${W} ${H}">${bars}</svg>`,`full split · 95% CI over seeds`);
 // mismatch
 const ms=runs.filter(r=>r.arm===st.marm&&(st.seed==="all"||r.seed==+st.seed));const mser=[];
 [["absn","numerics","#6a6a6a","2 3"],["absp","policy lag","#a8a8a8","6 3"],["absh","harness lag","#ffffff",""]].forEach(([k,n,c,d])=>{
  if(!ms.length)return;const L=Math.min(...ms.map(r=>r.steps.length));const pts=[];for(let i=0;i<L;i++){const v=ms.map(r=>r.steps[i][k]).filter(x=>x!=null);pts.push([i,v.length?mean(v):null])}
  mser.push({pts,c,dash:d,w:k==="absh"?2:1.3})});
 const marks=st.seed==="all"?[...new Set(runs.filter(r=>r.arm==="A").flatMap(r=>r.edits.filter(e=>e.harness).map(e=>e.step)))].map(x=>({x,t:"edit"})):editMarks(+st.seed);
 h+=card("c7",`per-source mismatch on stale tokens · arm ${st.marm}`,chart(mser,{h:230,w:700,log:true,xl:"rl step",marks})+legend([["numerics  |log π_trainer − log μ_sampler|","#6a6a6a"],["policy lag","#a8a8a8"],["harness lag","#fff"]]),"mean |term| per token · log scale");
 // clip + reward
 const cser=[..."ADE"].filter(a=>st.arms.has(a)).map(a=>{const rs=sel(a);if(!rs.length)return null;const L=Math.min(...rs.map(r=>r.steps.length));const pts=[];for(let i=0;i<L;i++){const v=rs.map(r=>r.steps[i].clip).filter(x=>x!=null);pts.push([i,v.length?mean(v):null])}return {pts,c:COL[a],dash:DASH[a],w:WID[a]}}).filter(Boolean);
 const rser=[..."ABCDE"].filter(a=>st.arms.has(a)).map(a=>{const rs=sel(a);if(!rs.length)return null;const L=Math.min(...rs.map(r=>r.steps.length));const pts=[];for(let i=0;i<L;i++){pts.push([i,mean(rs.map(r=>r.steps[i].reward))])}return {pts,c:COL[a],dash:DASH[a],w:WID[a]}}).filter(Boolean);
 h+=card("c5","share of stale tokens clipped",chart(cser,{h:100,w:480,pct:true})+`<h3 style="margin-top:10px">train reward / rollout</h3>`+chart(rser,{h:100,w:480}),"A · D · E");
 // demo tokens
 const dm=(runs.find(r=>r.arm===st.marm&&r.demo&&(st.seed==="all"||r.seed==+st.seed))||runs.find(r=>r.demo))?.demo;
 let tb="waiting for a re-rendered stale trajectory";
 if(dm){const hi=(a,b)=>{let i=0;while(i<a.length&&a[i]===b[i])i++;return [a.slice(0,i),a.slice(i)]};const esc=s=>s.replace(/&/g,"&amp;").replace(/</g,"&lt;");
  const sys=s=>s.split("<|im_start|>user")[0];
  tb=`<div class="two"><div><div class="lbl">old harness · ${diff(dm.old_harness,dm.new_harness,0)}</div><div class="ctx">${esc(sys(dm.old_context))}</div></div><div><div class="lbl">current harness · ${diff(dm.old_harness,dm.new_harness,1)}</div><div class="ctx">${esc(sys(dm.new_context))}</div></div></div>
  <div class="lbl" style="margin-top:10px">sampled action tokens · shade = |harness lag| · underline = IS weight &lt; 1 · step ${dm.step}</div><div class="tokens">${dm.tokens.map((t,i)=>{const v=Math.min(1,Math.abs(dm.harness_lag[i])/1.0);const w=dm.weight[i];
   return `<span title="numerics ${f(dm.numerics[i],4)} · policy ${f(dm.policy_lag[i],4)} · harness ${f(dm.harness_lag[i],4)} · w ${f(w,3)}" style="background:rgba(255,255,255,${0.04+0.8*v});color:${v>0.5?"#000":"#ddd"};${w<0.999?"text-decoration:underline":""}">${esc(t)}</span>`}).join("")}</div>`}
 h+=card("c7","one stale trajectory, re-rendered",tb,dm?`reward ${dm.reward}`:"");
 // edit log
 let rows="";runs.filter(r=>r.arm==="A"&&(st.seed==="all"||r.seed==+st.seed)).forEach(r=>r.edits.forEach(e=>(e.candidates||[]).forEach(c=>{rows+=`<tr class="${e.accepted===c.edit?"acc":""}"><td>s${r.seed}</td><td>${e.step}</td><td class="t">${c.edit}</td><td>${f(c.delta,3)}</td><td>${f(c.p,3)}</td><td>${e.accepted===c.edit?'<span class="pill w">accepted</span>':'<span class="pill">no</span>'}</td></tr>`})));
 h+=card("c5","harness edit proposals (paired test, p &lt; 0.1)",`<div style="max-height:260px;overflow:auto"><table><tr><th>seed</th><th>step</th><th>edit</th><th>Δ reward</th><th>p</th><th></th></tr>${rows||'<tr><td colspan=6>none yet</td></tr>'}</table></div>`,"selection split n="+(runs[0]?.config.select_n));
 // runs table
 let rr="";runs.forEach(r=>{const last=r.steps[r.steps.length-1]||{};rr+=`<tr><td class="t">${r.arm}</td><td>${r.seed}</td><td>${r.steps.length}/${r.config.steps}</td><td>${f(last.reward,3)}</td><td>${r.evals.length?f(100*r.evals[r.evals.length-1].acc):"–"}</td><td class="t">${r.final?f(100*r.final.acc):"running"}</td><td>${f(mean(r.steps.map(s=>s.tool_calls||0)),2)}</td><td>${r.steps.reduce((s,x)=>s+(x.dropped_nonrenderable||0),0)}</td><td>${f(r.time.total/3600,2)}h</td></tr>`});
 h+=card("c12","runs",`<table><tr><th>arm</th><th>seed</th><th>steps</th><th>last reward</th><th>curve acc %</th><th>final acc %</th><th>tool calls / rollout</th><th>dropped non-renderable</th><th>wall</th></tr>${rr}</table>`,`${runs.length} runs`);
 g.innerHTML=h;
}
function side(){
 const sd=document.getElementById("seeds");sd.innerHTML="";["all",...seeds].forEach(s=>{const b=el("button",{class:st.seed==s?"on":""},s);b.onclick=()=>{st.seed=String(s);side();render()};sd.appendChild(b)});
 const ar=document.getElementById("arms");ar.innerHTML="";[..."ABCDE"].forEach(a=>{const d=el("div",{class:"arm"+(st.arms.has(a)?" on":"")},`<span class="sw" style="border-color:${COL[a]};border-top-style:${DASH[a]?"dashed":"solid"};border-top-width:${a==="D"?3:2}px"></span>${a} · ${NAMES[a]}<span class="n">${runs.filter(r=>r.arm===a).length}</span>`);d.onclick=()=>{st.arms.has(a)?st.arms.delete(a):st.arms.add(a);side();render()};ar.appendChild(d)});
 const mr=document.getElementById("marm");mr.innerHTML="";[..."BCDE"].forEach(a=>{const b=el("button",{class:st.marm==a?"on":""},a);b.onclick=()=>{st.marm=a;side();render()};mr.appendChild(b)});
 const c=runs[0]?.config||{};document.getElementById("runinfo").innerHTML=`model ${c.model?.split("/")[1]||"–"}<br>steps ${c.steps} · ${c.prompts}×${c.group} rollouts<br>buffer ${c.buffer} batches · cap ${c.cap}<br>edits every ${c.edit_every} steps<br>lr ${c.lr}`;
 document.getElementById("model").textContent=c.model||"";document.getElementById("label").textContent=c.label||"";document.getElementById("nruns").textContent=`${runs.filter(r=>r.final).length}/${runs.length} runs complete`;
}
side();render();
</script></body></html>"""

if __name__ == "__main__":
    main()
