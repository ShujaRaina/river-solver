"use strict";

const RANKS = "AKQJT98765432";
const SUITS = [["s", "♠", "spade"], ["h", "♥", "heart"],
               ["d", "♦", "diamond"], ["c", "♣", "club"]];
const PALETTE = ["#3fb27f", "#e9c46a", "#f4a261", "#e35d5d", "#b07de8"];

const state = {
  board: [],                 // up to 5 card strings, e.g. "As"
  ranges: [{}, {}],          // ranges[player] = { "AKs": weight, ... }
  paintWeight: 1.0,
  painting: false,
  paintMode: "paint",        // or "erase"
};

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
  for (let i = 0; i < 5; i++) {
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
  else if (state.board.length < 5) state.board.push(card);
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
  cell.querySelector(".fill").style.background = player === 0 ? "#4c9be8" : "#b07de8";
}
function paintGrid(player) {
  for (let row = 0; row < 13; row++)
    for (let col = 0; col < 13; col++) paintCell(player, classLabel(row, col));
}

// ---- solve + results ------------------------------------------------------
async function solve() {
  const status = document.getElementById("status");
  if (state.board.length !== 5) { status.textContent = "pick 5 board cards"; return; }
  const btn = document.getElementById("solve");
  btn.disabled = true; status.textContent = "solving…";
  try {
    const res = await fetch("/solve", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        board: state.board,
        range0: state.ranges[0], range1: state.ranges[1],
        pot: +document.getElementById("pot").value,
        stack: +document.getElementById("stack").value,
        iters: +document.getElementById("iters").value,
      }),
    });
    const data = await res.json();
    if (data.error) { status.textContent = "error: " + data.error; return; }
    status.textContent = "";
    renderResult(data);
  } catch (e) {
    status.textContent = "request failed: " + e.message;
  } finally {
    btn.disabled = false;
  }
}

function renderResult(data) {
  document.getElementById("results").hidden = false;
  document.getElementById("expl").textContent =
    `exploitability ${data.exploitability_pct}% of pot`;

  const legend = document.getElementById("legend");
  legend.innerHTML = data.actions.map((a, i) =>
    `<span><i style="background:${PALETTE[i % PALETTE.length]}"></i>${a}</span>`).join("");

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
