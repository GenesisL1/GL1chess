# GenesisL1 Chess EVM AI Engine
### Chess EVM Convolutional Neural Network (Onchain Policy Net + Offchain Search UI)

### https://GenesisL1.com/Chess.html

A **serverless, single-page chess app** that plays chess using a **quantized convolutional neural network deployed as an EVM smart contract**. The model produces **policy logits** over a **4672-move** action space. The frontend combines:

- **On-chain inference** (`eth_call`) for policy evaluation  
- **Off-chain legality, move generation, and fast heuristics** (via `chess.js`)  
- A practical **search loop** (1-ply refine / widened 1-ply / optional 2-ply)  
- Optional **engine assistance** (suggest move without playing it)  
- **FIDE-style chess clocks** (90+30)  
- **Serverless P2P** (WebRTC data channel, no app server)

> **TL;DR:** The blockchain hosts the neural network. The browser hosts the chess rules + search orchestration.

---

## Table of Contents

- [What this is](#what-this-is)  
- [Key features](#key-features)  
- [Architecture overview](#architecture-overview)  
- [How on-chain inference works](#how-on-chain-inference-works)  
  - [BoardState encoding](#boardstate-encoding)  
  - [Legal move mask](#legal-move-mask)  
  - [Move encoding: 4672 = 64 × 73](#move-encoding-4672--64--73)  
  - [Model topology](#model-topology)  
  - [Weights storage & EIP-170](#weights-storage--eip170)  
- [Frontend search modes](#frontend-search-modes)  
  - [1-ply split refine](#1ply-split-refine)  
  - [1-ply widened](#1ply-widened)  
  - [2-ply](#2ply)  
  - [Mate-in-1 + opening book overrides](#matein1--opening-book-overrides)  
- [Run the UI locally](#run-the-ui-locally)  
- [Connect to GenesisL1 (defaults)](#connect-to-genesisl1-defaults)  
- [Contract API](#contract-api)  
- [P2P play (serverless WebRTC)](#p2p-play-serverless-webrtc)  
- [Debugging](#debugging)  
- [Deploying your own network](#deploying-your-own-network)  
- [Performance notes & RPC limits](#performance-notes--rpc-limits)  
- [Security & trust](#security--trust)  
- [License](#license)

---

## What this is

This repo contains two main parts:

1. **`ChessPolicyNetV1` (Solidity)**  
   A quantized CNN policy network that runs **entirely inside the EVM** and outputs logits for chess moves. It exposes:
   - `inferTopK(state, legalMask, k)` → get best *k* policy moves  
   - `inferBest(state, legalMask)` → get argmax policy move  
   - `weightsReady()` → model deployed & wired  

2. **A single-page web UI (HTML/JS)**  
   - Renders a chessboard with drag-and-drop  
   - Uses `chess.js` for legality + PGN/FEN  
   - Uses `ethers.js` to call the contract (`eth_call`)  
   - Implements a practical “**policy + tactics + refinement**” search loop

This isn’t Stockfish-on-chain. It’s a **policy network** + **light search logic** designed to make on-chain inference usable in real time.

---

## Key features

### Engine / AI
- ✅ On-chain CNN inference (`inferTopK`, `inferBest`)  
- ✅ Deterministic play after opening variety (no midgame randomness)  
- ✅ Candidate filtering + tactical injection (captures/checks are never ignored)  
- ✅ Optional 2-ply line simulation (my → opp → my)

### UI / UX
- ✅ Drag & drop only lands on legal squares  
- ✅ FEN display + copy/load  
- ✅ PGN copy/paste/download  
- ✅ Captured pieces tracker  
- ✅ FIDE 90+30 clocks (start on first move, flag detection)  
- ✅ Engine move + engine suggest (assist mode: off / once / always)  
- ✅ Debug dock + copy full debug JSON

### Multiplayer
- ✅ **Serverless P2P**: WebRTC offer/answer over any chat  
- ✅ Host-controlled setup (new game / undo / load FEN/PGN)  
- ✅ Clock and move history synced via the data channel

---

## Architecture overview

```
┌───────────────────────────┐
│ Browser (UI + chess.js)   │
│                           │
│ - legal moves             │
│ - move encoding (4672)    │
│ - legalMask (584 bytes)   │
│ - search + heuristics     │
│ - optional 2-ply          │
│ - P2P sync via WebRTC     │
└──────────────┬────────────┘
               │ eth_call (ethers.js)
               ▼
┌───────────────────────────┐
│ EVM / GenesisL1           │
│ ChessPolicyNetV1.sol      │
│                           │
│ - quantized CNN forward   │
│ - policy logits (4672)    │
│ - inferTopK / inferBest   │
│ - weights stored as blobs │
└───────────────────────────┘
```

---

## How on-chain inference works

### BoardState encoding

The contract takes a compact `BoardState`:

```solidity
struct BoardState {
  uint64[12] bb;   // bitboards: a1=LSB, h8=MSB
  uint8 stm;       // side to move: 0=white, 1=black
  uint8 castling;  // bit0 WK, bit1 WQ, bit2 BK, bit3 BQ
  int8 epFile;     // -1 none, else 0..7 (file a..h)
}
```

The frontend builds this from `chess.js` by iterating squares a1..h8 and setting bits in one of 12 planes:

- 6 piece planes for White: `P,N,B,R,Q,K`
- 6 piece planes for Black: `P,N,B,R,Q,K`

Then it sets “state” planes in the CNN input:
- Side-to-move plane  
- 4 castling planes  
- En-passant plane (single square encoded from `epFile`)

---

### Legal move mask

Both `inferTopK` and `inferBest` require a `legalMask`:

- **Length:** `584 bytes`
- **Why:** `4672 moves / 8 bits = 584 bytes`

The UI computes all legal moves (via chess.js), encodes each move into `[0..4671]`, and sets the corresponding bit in the mask. The contract ignores any move whose bit is not set.

---

### Move encoding: 4672 = 64 × 73

This engine uses an AlphaZero-style move encoding:

- 64 from-squares  
- 73 move planes per square  

Planes:
- 56 sliding directions (8 × 7)  
- 8 knight jumps  
- 9 underpromotion planes (N/B/R × forward/capture-left/capture-right)

Total: `64 × 73 = 4672`

The frontend and contract implement **identical encoding logic**.

---

### Model topology

`ChessPolicyNetV1` is a compact policy CNN designed for EVM execution:

- Input planes: **18**
- Channels: **24**
- Residual blocks: **4**
- Policy head: **2 × 64**
- Output: **4672 logits**

Pipeline:
1. Input encoding (binary planes)  
2. 3×3 stem convolution  
3. 4 residual blocks (3×3 + 3×3)  
4. 1×1 policy head  
5. Fully-connected projection to 4672 moves

All math is **int8 / int32 quantized** with fixed-point scaling.

📌 648,314 trainable parameters

| Component       | Parameters  |
| --------------- | ----------- |
| Stem            | 3,912       |
| Residual trunk  | 41,664      |
| Policy head     | 50          |
| Fully connected | 602,688     |
| **TOTAL**       | **648,314** |


---

### Weights storage & EIP-170

Due to EVM bytecode size limits:

- All weights are stored in **separate blob contracts**
- Fully-connected weights are split into **25 chunks**
- `setWeights(...)` wires everything together
- `weightsReady = true` signals the model is live

---

## Frontend search modes

### 1-ply split refine (default)
- Root: `inferTopK`
- Refine only top `refineN` candidates with `inferBest`
- Best strength-to-RPC ratio

### 1-ply widened
- Refine **all** candidates
- Stronger, more RPC calls

### 2-ply
- My move → opponent reply → my reply
- Uses NN on both sides
- More expensive but safer tactically

### Mate-in-1 + opening book overrides
- Mate-in-1 detected locally (no RPC)
- Opening book used only for first move variety
- Deterministic play afterward

---

## Run the UI locally

Serve as static HTML (do not use `file://`):

```bash
python -m http.server 8080
# open http://localhost:8080
```

or

```bash
npx serve .
```

---

## Connect to GenesisL1 (defaults)

- **RPC:** `https://rpc.genesisl1.org`
- **Contract:** `0x37D6518FbABd982e1908251A9cb4aD97b48BB989`

Click **Connect** in the UI sidebar.

---

## Contract API

- `weightsReady() → bool`
- `inferTopK(BoardState, legalMask, k)`
- `inferBest(BoardState, legalMask)`
- `inferBest1Ply(SearchQuery)` (stack-safe helper)

---

## P2P play (serverless WebRTC)

- WebRTC data channel
- No servers
- Host controls setup
- Moves + clocks synced deterministically

---

## Debugging

- Live debug dock
- Copy full debug JSON
- Logs all inference steps, timings, and errors

---

## Deploying your own network

1. Deploy weight blob contracts  
2. Deploy `ChessPolicyNetV1(shift)`  
3. Call `setWeights(...)`  
4. Point UI to your RPC + contract

---

## Performance notes & RPC limits

- Calls per move:
  - 1-ply: `1 + refineN`
  - 1-ply wide: `1 + candN`
  - 2-ply: `1 + 2*candN`
- Keep `candN` small on public RPCs
- Prefer split-refine for best UX

---

## Security & trust

- Trust the **contract address**
- Owner can update weights unless frozen
- Inference is deterministic for fixed weights

---

## License

MIT © 2025 Decentralized Science Labs
