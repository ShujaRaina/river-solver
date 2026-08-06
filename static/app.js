"use strict";

const RANKS = "AKQJT98765432";
const SUITS = [["s", "♠", "spade"], ["h", "♥", "heart"],
               ["d", "♦", "diamond"], ["c", "♣", "club"]];
// 7 steps so check + five bet sizes + all-in each get a distinct colour
// (calm green -> escalating warm -> deep maroon for the biggest sizings).
const PALETTE = ["#3fb27f", "#8ab861", "#e9c46a", "#f4a261",
                 "#e76f51", "#c62d42", "#7a1420"];

// pretty action labels: "bet33" -> "Bet 33", "allin" -> "All In"
function displayName(a) {
  const fixed = { check: "Check", call: "Call", fold: "Fold", allin: "All In" };
  if (fixed[a]) return fixed[a];
  const m = a.match(/^(bet|raise)(\d+)$/);
  if (m) return m[1][0].toUpperCase() + m[1].slice(1) + " " + m[2];
  return a;
}

// stable id so the server scopes "cancel my previous solve" to this browser
// (re-clicking Solve cancels only my own in-flight solve, never another user's
// on the shared deploy). Persisted in localStorage so a REFRESH keeps the same
// id -- otherwise a fresh id would fail to cancel the solve the old page left
// running, and that orphan would hold the single concurrency slot.
const CLIENT_ID = (() => {
  let id = localStorage.getItem("rs_client");
  if (!id) {
    id = Math.random().toString(36).slice(2) + Date.now().toString(36);
    localStorage.setItem("rs_client", id);
  }
  return id;
})();

// Cancel any in-flight solve when the page is hidden/closed/refreshed, so it
// frees the slot immediately instead of running on to the backstop orphaned.
window.addEventListener("pagehide", () => {
  navigator.sendBeacon("/river/cancel", JSON.stringify({ client: CLIENT_ID }));
});

// bet sizes come from the four editable % inputs (fractions of pot); all-in is
// auto-added by the engine. Blank/out-of-range boxes are dropped; valid range
// is 5%-300% (matches the server's fraction bounds).
// the slider picks how many bet-size boxes are active (1-5); read those.
function betSizes() {
  const n = +document.getElementById("nsizes").value;
  return [...document.querySelectorAll(".betsize")].slice(0, n)
    .map(el => +el.value / 100)
    .filter(f => f >= 0.05 && f <= 3.0);
}

// reflect the slider: show N size boxes, update the count, warn at 4-5 (a
// bigger tree slows solves, especially on the small hosted box).
function updateSizeCount() {
  const slider = document.getElementById("nsizes");
  const n = +slider.value;
  document.querySelectorAll(".betsize").forEach((el, i) => { el.hidden = i >= n; });
  const count = document.getElementById("nsizes-count");
  count.textContent = n;
  // place the number directly under the slider thumb (tracks it across 1-5)
  const w = slider.offsetWidth || 260, thumb = 16;
  count.style.left = (((n - 1) / 4) * (w - thumb) + thumb / 2) + "px";
  // toggle visibility (not display) so the warning's space is always reserved
  document.getElementById("sizes-warn").style.visibility = n >= 4 ? "visible" : "hidden";
}

// A canonical signature of everything that defines the solve/tree (NOT iters --
// adding iterations is exactly what Resume is for). If any of these changed
// since the solve started, Resume must start fresh instead of continuing a
// solver built for the old inputs. Board and range keys are sorted so a
// re-pick in a different order doesn't look like a change.
function solveSignature() {
  const rangeKey = r => Object.keys(r).sort().map(k => `${k}:${r[k]}`).join(",");
  return JSON.stringify({
    board: [...state.board].sort(),
    r0: rangeKey(state.ranges[0]), r1: rangeKey(state.ranges[1]),
    pot: +document.getElementById("pot").value,
    stack: +document.getElementById("stack").value,
    sizes: betSizes(),
  });
}

const state = {
  street: "river",           // "river" (5 cards) or "turn" (4 cards)
  board: [],                 // card strings, e.g. "As"
  ranges: [{}, {}],          // ranges[player] = { "AKs": weight, ... }
  paintWeight: 1.0,
  painting: false,
  selection: { player: null, labels: new Set() },  // hands the weight slider edits
  currentJob: null,          // id of the active/last solve (for Stop/Resume)
  resumable: false,          // last job kept a resumable solver
  solveSignature: null,      // inputs the current solver was built for
};

function slotCount() { return state.street === "turn" ? 4 : 5; }

// class label for grid cell (row, col), matching the backend's all_classes()
function classLabel(row, col) {
  if (row === col) return RANKS[row] + RANKS[row];
  if (row < col) return RANKS[row] + RANKS[col] + "s";
  return RANKS[col] + RANKS[row] + "o";
}

// ---- board ----------------------------------------------------------------
function buildBoard() {
  const slots = document.getElementById("board-slots");
  slots.innerHTML = "";
  for (let i = 0; i < slotCount(); i++) {
    const s = document.createElement("div");
    s.className = "slot";
    const card = state.board[i];
    if (card) {
      s.textContent = card[0] + SUITS.find(x => x[0] === card[1])[1];
      s.classList.add(SUITS.find(x => x[0] === card[1])[2]);
      s.style.color = { s: "#e6edf3", h: "#ff6b6b", d: "#5aa9ff", c: "#4cd07d" }[card[1]];
    }
    slots.appendChild(s);
  }

  const picker = document.getElementById("card-picker");
  if (picker.childElementCount === 0) {
    for (const [su, sym, cls] of SUITS) {
      for (const r of RANKS) {
        const card = r + su;
        const b = document.createElement("button");
        b.className = "card " + cls;
        b.textContent = r + sym;
        b.dataset.card = card;
        b.onclick = () => toggleBoardCard(card);
        picker.appendChild(b);
      }
    }
  }
  for (const b of picker.children)
    b.classList.toggle("picked", state.board.includes(b.dataset.card));
}

function toggleBoardCard(card) {
  const i = state.board.indexOf(card);
  if (i >= 0) state.board.splice(i, 1);
  else if (state.board.length < slotCount()) state.board.push(card);
  buildBoard();
}

// ---- range grids ----------------------------------------------------------
function buildGrid(player) {
  const grid = document.getElementById("grid-" + player);
  grid.innerHTML = "";
  for (let row = 0; row < 13; row++) {
    for (let col = 0; col < 13; col++) {
      const label = classLabel(row, col);
      const c = document.createElement("div");
      c.className = "cell" + (row === col ? " pair" : "");
      c.dataset.label = label;
      c.dataset.player = player;
      c.innerHTML = `<div class="fill"></div><span>${label}</span>`;
      c.addEventListener("mousedown", e => { e.preventDefault(); startPaint(player, label); });
      c.addEventListener("mouseenter", () => { if (state.painting) applyPaint(player, label); });
      c.addEventListener("dblclick", e => { e.preventDefault(); toggleFull(player, label); });
      grid.appendChild(c);
    }
  }
  paintGrid(player);
}

// Select-first workflow: clicking/dragging only SELECTS hands -- it never
// changes a hand's weight. The slider syncs to the selection's current weight
// (0 for an untouched square), and moving it sets the weight (adding the hand
// to the range once it's > 0). Clicking outside the grids clears the selection.
function startPaint(player, label) {
  state.painting = true;
  clearSelection();
  state.selection.player = player;
  applyPaint(player, label);
}
function applyPaint(player, label) {
  if (state.selection.player !== player) return;      // don't cross grids mid-stroke
  state.selection.labels.add(label);                  // select only; weight unchanged
  markSelected(player, label, true);
  paintCell(player, label);
}
// double-click toggles a square: empty -> 100%, any fill -> 0% (removed). It
// also makes the square the active single selection so the slider tracks it.
function toggleFull(player, label) {
  if ((state.ranges[player][label] || 0) > 0) delete state.ranges[player][label];
  else state.ranges[player][label] = 1.0;
  clearSelection();
  state.selection.player = player;
  state.selection.labels.add(label);
  markSelected(player, label, true);
  paintCell(player, label);
  syncWeightSlider();
}
function clearSelection() {
  const { player, labels } = state.selection;
  if (player != null) for (const lab of labels) markSelected(player, lab, false);
  state.selection = { player: null, labels: new Set() };
}
function markSelected(player, label, on) {
  const cell = document.querySelector(`#grid-${player} .cell[data-label="${label}"]`);
  if (cell) cell.classList.toggle("selected", on);
}
// reflect the selection's weight on the slider so adjusting starts from the
// current value (uniform selection -> that weight; mixed -> 100%).
function syncWeightSlider() {
  const { player, labels } = state.selection;
  if (player == null || labels.size === 0) return;
  const ws = [...labels].map(l => state.ranges[player][l] ?? 0);
  const w = ws.every(x => x === ws[0]) ? ws[0] : 1.0;
  state.paintWeight = w;
  document.getElementById("weight").value = Math.round(w * 100);
  document.getElementById("weight-label").textContent = Math.round(w * 100) + "%";
}
function paintCell(player, label) {
  const cell = document.querySelector(`#grid-${player} .cell[data-label="${label}"]`);
  const w = state.ranges[player][label] || 0;
  cell.classList.toggle("on", w > 0);
  // fill left-to-right by weight (like PioSolver / PokerCruncher), not by opacity
  cell.querySelector(".fill").style.width = (w * 100) + "%";
  cell.querySelector(".fill").style.background = player === 0 ? "#4c9be8" : "#e35d5d";
}
function paintGrid(player) {
  for (let row = 0; row < 13; row++)
    for (let col = 0; col < 13; col++) paintCell(player, classLabel(row, col));
}

// ---- solve + results ------------------------------------------------------
// Button state: while solving -> only Stop; when idle -> Solve, plus Resume if
// the last job kept a resumable solver.
function setSolving(on) {
  document.getElementById("solve").disabled = on;
  document.getElementById("stop").disabled = !on;
  document.getElementById("resume").disabled = on || !(state.resumable && state.currentJob);
}

async function solve() {
  if (state.street === "turn") return solveTurn();
  const status = document.getElementById("status");
  if (state.board.length !== 5) { status.textContent = "pick 5 board cards"; return; }
  if (!betSizes().length) { status.textContent = "enter at least one bet size (5–300%)"; return; }
  state.solveSignature = solveSignature();     // remember what this solve is for
  setSolving(true); status.textContent = "starting…";
  let start;
  try {
    start = await (await fetch("/river/solve", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        client: CLIENT_ID,
        board: state.board, range0: state.ranges[0], range1: state.ranges[1],
        pot: +document.getElementById("pot").value,
        stack: +document.getElementById("stack").value,
        iters: +document.getElementById("iters").value,
        fractions: betSizes(),
      }),
    })).json();
  } catch (e) { status.textContent = "request failed: " + e.message; state.resumable = false; setSolving(false); return; }
  if (start.error) { status.textContent = "error: " + start.error; state.resumable = false; setSolving(false); return; }
  state.currentJob = start.id;
  pollJob(start.id);
}

// Poll a job to the ticking counter; on finish, flip buttons and note whether
// it can be resumed (stopped early or under the iteration cap).
function pollJob(id) {
  const status = document.getElementById("status");
  const poll = async () => {
    if (state.currentJob !== id) return;      // a newer solve superseded this poll
    let d;
    try { d = await (await fetch("/river/progress/" + id)).json(); }
    catch (e) { status.textContent = "poll failed: " + e.message; setSolving(false); return; }
    if (d.error) { status.textContent = "error: " + d.error; state.resumable = false; setSolving(false); return; }
    const reached = d.iter >= d.target;
    status.textContent = d.timeout
      ? `stopped at time limit — ${d.iter} iterations (spot too deep to fully converge)`
      : d.done
      ? `${reached ? "done" : "stopped"} — ${d.iter} iterations`
      : `solving…  ${d.iter} / ${d.target} iterations`;
    if (d.actions && d.actions.length) renderResult(d);
    if (d.done) { state.resumable = !!d.resumable; setSolving(false); return; }
    setTimeout(poll, 200);
  };
  poll();
}

async function stopSolve() {
  document.getElementById("stop").disabled = true;   // the poll loop finalizes the rest
  try {
    await fetch("/river/cancel", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ client: CLIENT_ID }),
    });
  } catch (e) { /* the worker will stop on its next chunk regardless */ }
}

async function resumeSolve() {
  if (!state.currentJob) return;
  // inputs changed since this solve was built -> the stored solver is stale;
  // start a fresh solve for the new board/pot/stack/ranges/sizes instead.
  if (solveSignature() !== state.solveSignature) return solve();
  const status = document.getElementById("status");
  setSolving(true); status.textContent = "resuming…";
  let r;
  try {
    r = await (await fetch("/river/resume", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: state.currentJob, client: CLIENT_ID,
        iters: +document.getElementById("iters").value,
      }),
    })).json();
  } catch (e) { status.textContent = "request failed: " + e.message; setSolving(false); return; }
  if (r.error) { status.textContent = "error: " + r.error; state.resumable = false; setSolving(false); return; }
  pollJob(state.currentJob);
}

async function solveTurn() {
  const status = document.getElementById("status");
  if (state.board.length !== 4) { status.textContent = "pick 4 board cards"; return; }
  const btn = document.getElementById("solve");
  btn.disabled = true; status.textContent = "starting turn solve…";
  let start;
  try {
    start = await (await fetch("/turn/solve", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        board: state.board, range0: state.ranges[0], range1: state.ranges[1],
        pot: +document.getElementById("pot").value,
        stack: +document.getElementById("stack").value,
        iters: +document.getElementById("iters").value,
      }),
    })).json();
  } catch (e) { status.textContent = "request failed: " + e.message; btn.disabled = false; return; }
  if (start.error) { status.textContent = "error: " + start.error; btn.disabled = false; return; }

  const id = start.id;                       // then poll for live progress
  const poll = async () => {
    let d;
    try { d = await (await fetch("/turn/progress/" + id)).json(); }
    catch (e) { status.textContent = "poll failed: " + e.message; btn.disabled = false; return; }
    if (d.error) { status.textContent = "error: " + d.error; btn.disabled = false; return; }
    status.textContent = d.done
      ? `turn solve done — ${d.iter} iters`
      : `solving turn… iter ${d.iter}/${d.target} — watch it converge`;
    if (d.actions && d.actions.length) renderResult(d);
    if (d.done) { btn.disabled = false; return; }
    setTimeout(poll, 1500);
  };
  poll();
}

function renderResult(data) {
  document.getElementById("results").hidden = false;
  const parts = [];
  if (data.iter != null) parts.push(`${data.iter}${data.target ? " / " + data.target : ""} iterations`);
  if (data.exploitability_bb != null) parts.push(`exploitability ${data.exploitability_bb.toFixed(2)} bb`);
  document.getElementById("expl").textContent = parts.join("  ·  ");

  const legend = document.getElementById("legend");
  legend.innerHTML = data.actions.map((a, i) =>
    `<span><i style="background:${PALETTE[i % PALETTE.length]}"></i>` +
    `${displayName(a)} ${Math.round(data.mix[i] * 100)}%</span>`).join("");

  const grid = document.getElementById("grid-result");
  grid.innerHTML = "";
  for (let row = 0; row < 13; row++) {
    for (let col = 0; col < 13; col++) {
      const label = classLabel(row, col);
      const c = document.createElement("div");
      c.className = "cell" + (row === col ? " pair" : "");
      const strat = data.strategy[label];
      if (strat) {
        const bars = strat.map((f, i) =>
          `<i style="width:${f * 100}%;background:${PALETTE[i % PALETTE.length]}"></i>`).join("");
        c.innerHTML = `<div class="bars">${bars}</div><span>${label}</span>`;
        c.classList.add("on");
      } else {
        c.innerHTML = `<span>${label}</span>`;
      }
      grid.appendChild(c);
    }
  }
}

// ---- wiring ---------------------------------------------------------------
document.addEventListener("mouseup", () => {
  if (state.painting) syncWeightSlider();   // reflect the finished selection's weight
  state.painting = false;
});
// clicking outside the range grids clears the selection -- but not on the Hand
// weight control, which is the tool used to edit the current selection.
document.addEventListener("mousedown", e => {
  if (e.target.closest("#grid-0, #grid-1, .weight-row")) return;
  clearSelection();
});
document.getElementById("weight").addEventListener("input", e => {
  state.paintWeight = e.target.value / 100;
  document.getElementById("weight-label").textContent = e.target.value + "%";
  // apply the weight to the currently selected hands (select-then-weight)
  const { player, labels } = state.selection;
  if (player != null) for (const lab of labels) {
    if (state.paintWeight > 0) state.ranges[player][lab] = state.paintWeight;
    else delete state.ranges[player][lab];       // dialing to 0% removes the hand
    paintCell(player, lab);
  }
});
for (const b of document.querySelectorAll(".clear"))
  b.onclick = () => { state.ranges[b.dataset.player] = {}; clearSelection(); paintGrid(+b.dataset.player); };
document.getElementById("solve").onclick = solve;
document.getElementById("stop").onclick = stopSolve;
document.getElementById("resume").onclick = resumeSolve;
document.getElementById("nsizes").oninput = updateSizeCount;
for (const r of document.querySelectorAll('input[name="street"]')) {
  r.onchange = () => {
    state.street = r.value;
    document.getElementById("board-hint").textContent = "pick " + slotCount();
    if (state.board.length > slotCount()) state.board = state.board.slice(0, slotCount());
    document.getElementById("iters").value = state.street === "turn" ? 60 : 1000;
    buildBoard();
  };
}

// ---- defaults so Solve works immediately ----------------------------------
// Both players start with the same range (weights are frequencies 0-1).
function preset() {
  state.board = ["As", "Kd", "7s", "2c", "9h"];
  const full = (
    "66 77 88 99 TT JJ QQ KK AA " +            // 66-AA
    "A3s A4s A5s A6s A7s A8s A9s ATs AJs AQs AKs " +  // A3s-AKs
    "K9s KTs KJs KQs Q9s QTs QJs J9s JTs T9s " +      // K9s-KQs, Q9s-QJs, J9s-JTs, T9s
    "AJo AQo AKo KQo").split(" ");              // AJo-AKo, KQo
  const w = {};
  for (const h of full) w[h] = 1;
  for (const h of "ATo KJo K8s 55".split(" ")) w[h] = 0.75;
  for (const h of "QJo K7s T8s 98s 44".split(" ")) w[h] = 0.5;
  for (const h of "K6s 87s 76s 65s 33 22".split(" ")) w[h] = 0.25;
  state.ranges = [{ ...w }, { ...w }];
}

preset();
buildBoard();
buildGrid(0);
buildGrid(1);
updateSizeCount();
