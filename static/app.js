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

// bet sizes offered to the solver (fractions of pot); all-in is auto-added by
// the engine, so this menu of 4 -> five actions (33/66/100/200/all-in).
const BET_SIZES = [0.33, 0.66, 1.0, 2.0];

const state = {
  street: "river",           // "river" (5 cards) or "turn" (4 cards)
  board: [],                 // card strings, e.g. "As"
  ranges: [{}, {}],          // ranges[player] = { "AKs": weight, ... }
  paintWeight: 1.0,
  painting: false,
  paintMode: "paint",        // or "erase"
  currentJob: null,          // id of the active/last solve (for Stop/Resume)
  resumable: false,          // last job kept a resumable solver
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
      grid.appendChild(c);
    }
  }
  paintGrid(player);
}

function startPaint(player, label) {
  state.painting = true;
  state.paintMode = state.ranges[player][label] ? "erase" : "paint";
  applyPaint(player, label);
}
function applyPaint(player, label) {
  if (state.paintMode === "paint") state.ranges[player][label] = state.paintWeight;
  else delete state.ranges[player][label];
  paintCell(player, label);
}
function paintCell(player, label) {
  const cell = document.querySelector(`#grid-${player} .cell[data-label="${label}"]`);
  const w = state.ranges[player][label] || 0;
  cell.classList.toggle("on", w > 0);
  cell.querySelector(".fill").style.opacity = w * 0.85;
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
        fractions: BET_SIZES,
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
document.addEventListener("mouseup", () => { state.painting = false; });
document.getElementById("weight").addEventListener("input", e => {
  state.paintWeight = e.target.value / 100;
  document.getElementById("weight-label").textContent = e.target.value + "%";
});
for (const b of document.querySelectorAll(".clear"))
  b.onclick = () => { state.ranges[b.dataset.player] = {}; paintGrid(+b.dataset.player); };
document.getElementById("solve").onclick = solve;
document.getElementById("stop").onclick = stopSolve;
document.getElementById("resume").onclick = resumeSolve;
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
function preset() {
  state.board = ["As", "Kd", "7s", "2c", "9h"];
  const r0 = "AA KK QQ AKs AQs AJs KQs KJs QJs JTs T9s 99 88 77 A5s A4s".split(" ");
  const r1 = "AA KK 99 77 AKs AKo AQo AJo KQo QJs JTs T9s T8s 54s".split(" ");
  for (const c of r0) state.ranges[0][c] = 1;
  for (const c of r1) state.ranges[1][c] = 1;
}

preset();
buildBoard();
buildGrid(0);
buildGrid(1);
