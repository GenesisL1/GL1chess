// SPDX-License-Identifier: MIT
//
// Copyright (c) 2025 Decentralized Science Labs
// Developed by Decentralized Science Labs
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
//

pragma solidity ^0.8.20;

import {CodeBlobReader} from "./CodeBlobReader.sol";
import {BitMask} from "./BitMask.sol";

contract ChessPolicyNetV1 {
    using BitMask for bytes;

    // -----------------------------
    // Model constants
    // -----------------------------
    uint8 public constant CIN = 18;
    uint8 public constant CHANNELS = 24;
    uint8 public constant BLOCKS = 4;
    uint8 public constant POLICY_COUT = 2;
    uint16 public constant MOVES = 4672;

    uint256 public constant SQ = 64;

    uint256 public constant INPUT_LEN = uint256(CIN) * SQ;          // 1152
    uint256 public constant POLICY_LEN = uint256(POLICY_COUT) * SQ; // 128

    uint256 public constant FC_IN = POLICY_LEN;                     // 128
    uint256 public constant FC_B_LEN = uint256(MOVES) * 4;          // 18,688

    // Expected blob sizes (data bytes)
    uint256 public constant STEM_W_LEN = uint256(CHANNELS) * uint256(CIN) * 9;       // 3888
    uint256 public constant STEM_B_LEN = uint256(CHANNELS) * 4;                      // 96
    uint256 public constant BLK_W_LEN  = uint256(CHANNELS) * uint256(CHANNELS) * 9;  // 5184
    uint256 public constant BLK_B_LEN  = uint256(CHANNELS) * 4;                      // 96
    uint256 public constant POL_W_LEN  = uint256(POLICY_COUT) * uint256(CHANNELS);   // 48
    uint256 public constant POL_B_LEN  = uint256(POLICY_COUT) * 4;                   // 8

    // -----------------------------
    // FC chunking (EIP-170 safe)
    // -----------------------------
    uint256 public constant FC_W_CHUNK_LEN = 24448;         // 191 rows * 128 bytes
    uint256 public constant FC_ROWS_PER_CHUNK = 191;

    uint256 public constant FC_W_CHUNKS = 25;
    uint256 public constant FC_W_LAST_LEN = 11264;          // 88 rows * 128
    uint256 public constant FC_ROWS_LAST_CHUNK = 88;

    uint16 public constant MAX_TOPK = 32;

    // Material heuristic scales (unsigned to keep arithmetic simple)
    uint16 internal constant MY_CAP_SCALE  = 64; // bonus per 1 material-unit
    uint16 internal constant OPP_CAP_SCALE = 48; // penalty per 1 material-unit

    // Refine only top3 by opponent NN reply
    uint8 internal constant REFINE_K = 3;

    uint8 public immutable SHIFT;

    // -----------------------------
    // Weight blob pointers
    // -----------------------------
    address public owner;

    address public stemW;
    address public stemB;

    address[BLOCKS] public blockC1W;
    address[BLOCKS] public blockC1B;
    address[BLOCKS] public blockC2W;
    address[BLOCKS] public blockC2B;

    address public policyW;
    address public policyB;

    address[25] public fcWChunks;
    address public fcB;

    bool public weightsReady;

    // -----------------------------
    // Board input
    // -----------------------------
    struct BoardState {
        uint64[12] bb;   // a1=LSB, h8=MSB
        uint8 stm;       // 0=white, 1=black
        uint8 castling;  // bit0 WK, bit1 WQ, bit2 BK, bit3 BQ
        int8 epFile;     // -1 none, else 0..7
    }

    struct SearchQuery {
        BoardState root;
        bytes rootLegalMask;        // 584 bytes

        uint16[] candIdx;           // <=32
        int32[]  candLogits;        // same length

        BoardState[] nextStates;    // same length
        bytes[] oppMasks;           // same length; each 584 bytes

        uint8 alpha;                // 0..255
        int32 randMargin;           // >=0 recommended
        uint256 seed;               // 0 = auto
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor(uint8 shift_) {
        owner = msg.sender;
        SHIFT = shift_;
    }

    // -----------------------------
    // Admin: set weights
    // -----------------------------
    function setWeights(
        address stemW_,
        address stemB_,
        address[BLOCKS] calldata c1w,
        address[BLOCKS] calldata c1b,
        address[BLOCKS] calldata c2w,
        address[BLOCKS] calldata c2b,
        address policyW_,
        address policyB_,
        address[25] calldata fcWChunks_,
        address fcB_
    ) external onlyOwner {
        require(CodeBlobReader.size(stemW_) == STEM_W_LEN, "bad stemW size");
        require(CodeBlobReader.size(stemB_) == STEM_B_LEN, "bad stemB size");

        for (uint256 i = 0; i < BLOCKS; i++) {
            require(CodeBlobReader.size(c1w[i]) == BLK_W_LEN, "bad c1w size");
            require(CodeBlobReader.size(c1b[i]) == BLK_B_LEN, "bad c1b size");
            require(CodeBlobReader.size(c2w[i]) == BLK_W_LEN, "bad c2w size");
            require(CodeBlobReader.size(c2b[i]) == BLK_B_LEN, "bad c2b size");
        }

        require(CodeBlobReader.size(policyW_) == POL_W_LEN, "bad policyW size");
        require(CodeBlobReader.size(policyB_) == POL_B_LEN, "bad policyB size");

        for (uint256 i = 0; i < FC_W_CHUNKS; i++) {
            uint256 expected = (i + 1 == FC_W_CHUNKS) ? FC_W_LAST_LEN : FC_W_CHUNK_LEN;
            require(CodeBlobReader.size(fcWChunks_[i]) == expected, "bad fcW chunk size");
        }
        require(CodeBlobReader.size(fcB_) == FC_B_LEN, "bad fcB size");

        stemW = stemW_;
        stemB = stemB_;

        for (uint256 i = 0; i < BLOCKS; i++) {
            blockC1W[i] = c1w[i];
            blockC1B[i] = c1b[i];
            blockC2W[i] = c2w[i];
            blockC2B[i] = c2b[i];
        }

        policyW = policyW_;
        policyB = policyB_;

        for (uint256 i = 0; i < FC_W_CHUNKS; i++) {
            fcWChunks[i] = fcWChunks_[i];
        }
        fcB = fcB_;

        weightsReady = true;
    }

    // -----------------------------
    // Public inference
    // -----------------------------
    function inferBest(BoardState calldata s, bytes calldata legalMask)
        external
        view
        returns (uint16 bestIdx, int32 bestLogit)
    {
        require(weightsReady, "weights not set");
        require(legalMask.length == 584, "legalMask must be 584 bytes");
        return _inferBestUnchecked(s, legalMask);
    }

    function inferTopK(BoardState calldata s, bytes calldata legalMask, uint16 k)
        external
        view
        returns (uint16[] memory idx, int32[] memory logits)
    {
        require(weightsReady, "weights not set");
        require(legalMask.length == 584, "legalMask must be 584 bytes");
        require(k > 0 && k <= MAX_TOPK, "bad k");

        bytes memory x = _encodeInput(s);
        bytes memory y = _stemForward(x);
        y = _trunkForward(y);
        bytes memory p = _policyForward(y);

        return _fcTopKChunked(p, legalMask, k);
    }

    /// @notice Stack-safe 1-ply: compute fast score for all candidates, pick top3, refine only those with opponent NN reply.
    function inferBest1Ply(SearchQuery calldata q)
        external
        view
        returns (
            uint16 bestIdx,
            int32 bestScore,
            int32 bestMyLogit,
            int32 bestOppLogit,
            uint16 bestMyCap,
            uint16 bestOppCap
        )
    {
        require(weightsReady, "weights not set");
        require(q.rootLegalMask.length == 584, "root mask must be 584");

        uint256 n = q.candIdx.length;
        require(n > 0 && n <= MAX_TOPK, "bad cand count");
        require(q.candLogits.length == n, "candLogits len");
        require(q.nextStates.length == n, "nextStates len");
        require(q.oppMasks.length == n, "oppMasks len");

        // --- top3 by fastScore (no arrays to avoid stack spill)
        uint8 t0 = 255; int32 s0 = type(int32).min; uint16 my0 = 0; uint16 op0 = 0;
        uint8 t1 = 255; int32 s1 = type(int32).min; uint16 my1 = 0; uint16 op1 = 0;
        uint8 t2 = 255; int32 s2 = type(int32).min; uint16 my2 = 0; uint16 op2 = 0;

        for (uint256 i = 0; i < n; i++) {
            uint16 mi = q.candIdx[i];
            require(q.rootLegalMask.isSet(mi), "cand not legal");
            require(q.oppMasks[i].length == 584, "opp mask must be 584");

            uint16 myCap = _captureValueFromDelta(q.root, q.nextStates[i]);
            uint16 oppCap = _maxCaptureValue(q.nextStates[i], q.oppMasks[i]);

            int32 sc = q.candLogits[i];
            sc += int32(uint32(myCap) * uint32(MY_CAP_SCALE));
            sc -= int32(uint32(oppCap) * uint32(OPP_CAP_SCALE));

            // insert into top3
            if (sc > s0) {
                t2=t1; s2=s1; my2=my1; op2=op1;
                t1=t0; s1=s0; my1=my0; op1=op0;
                t0=uint8(i); s0=sc; my0=myCap; op0=oppCap;
            } else if (sc > s1) {
                t2=t1; s2=s1; my2=my1; op2=op1;
                t1=uint8(i); s1=sc; my1=myCap; op1=oppCap;
            } else if (sc > s2) {
                t2=uint8(i); s2=sc; my2=myCap; op2=oppCap;
            }
        }

        // refine up to 3
        uint8 rk = uint8(n);
        if (rk > REFINE_K) rk = REFINE_K;

        // refined scores + opp logits (again: no arrays)
        int32 r0 = type(int32).min; int32 o0 = 0;
        int32 r1 = type(int32).min; int32 o1 = 0;
        int32 r2 = type(int32).min; int32 o2 = 0;

        if (rk >= 1 && t0 != 255) { (r0, o0) = _refineOne(q.nextStates[t0], q.oppMasks[t0], s0, q.alpha); }
        if (rk >= 2 && t1 != 255) { (r1, o1) = _refineOne(q.nextStates[t1], q.oppMasks[t1], s1, q.alpha); }
        if (rk >= 3 && t2 != 255) { (r2, o2) = _refineOne(q.nextStates[t2], q.oppMasks[t2], s2, q.alpha); }

        // choose with optional randomness among near-best
        int32 margin = q.randMargin;
        if (margin < 0) margin = 0;

        bestScore = r0;
        if (rk >= 2 && r1 > bestScore) bestScore = r1;
        if (rk >= 3 && r2 > bestScore) bestScore = r2;

        int32 thr = bestScore - margin;

        uint256 cnt = 0;
        if (rk >= 1 && r0 >= thr) cnt++;
        if (rk >= 2 && r1 >= thr) cnt++;
        if (rk >= 3 && r2 >= thr) cnt++;

        uint256 seed = _autoSeed(q.seed, q.candIdx);
        uint256 pick = (cnt == 0) ? 0 : (seed % cnt);

        // pick in order t0,t1,t2
        if (rk >= 1 && r0 >= thr) {
            if (pick == 0) return _pack(q, t0, r0, o0, my0, op0);
            pick--;
        }
        if (rk >= 2 && r1 >= thr) {
            if (pick == 0) return _pack(q, t1, r1, o1, my1, op1);
            pick--;
        }
        if (rk >= 3 && r2 >= thr) {
            return _pack(q, t2, r2, o2, my2, op2);
        }

        // fallback
        return _pack(q, t0, r0, o0, my0, op0);
    }

    function _pack(
        SearchQuery calldata q,
        uint8 pos,
        int32 score,
        int32 oppLogit,
        uint16 myCap,
        uint16 oppCap
    ) internal pure returns (
        uint16 bestIdx,
        int32 bestScore,
        int32 bestMyLogit,
        int32 bestOppLogit,
        uint16 bestMyCap,
        uint16 bestOppCap
    ) {
        bestIdx = q.candIdx[pos];
        bestScore = score;
        bestMyLogit = q.candLogits[pos];
        bestOppLogit = oppLogit;
        bestMyCap = myCap;
        bestOppCap = oppCap;
    }

    function _refineOne(
        BoardState calldata child,
        bytes calldata oppMask,
        int32 fastScore,
        uint8 alpha
    ) internal view returns (int32 refinedScore, int32 oppLogitRaw) {
        (, oppLogitRaw) = _inferBestUnchecked(child, oppMask);

        uint32 opp = 0;
        if (oppLogitRaw > 0) opp = uint32(uint32(int32(oppLogitRaw)));

        uint32 pen = (uint32(alpha) * opp) / 255;
        refinedScore = fastScore - int32(pen);
    }

    function _autoSeed(uint256 seed, uint16[] calldata candIdx) internal view returns (uint256) {
        if (seed != 0) return seed;
        bytes32 bh = blockhash(block.number - 1);
        return uint256(keccak256(abi.encodePacked(bh, address(this), msg.sender, candIdx.length, candIdx[0])));
    }

    // -----------------------------
    // Internal full inference
    // -----------------------------
    function _inferBestUnchecked(BoardState calldata s, bytes calldata legalMask)
        internal
        view
        returns (uint16 bestIdx, int32 bestLogit)
    {
        bytes memory x = _encodeInput(s);
        bytes memory y = _stemForward(x);
        y = _trunkForward(y);
        bytes memory p = _policyForward(y);
        return _fcArgmaxChunked(p, legalMask);
    }

    // -----------------------------
    // Forward helpers
    // -----------------------------
    function _stemForward(bytes memory x) internal view returns (bytes memory y) {
        bytes memory stemWb = CodeBlobReader.read(stemW, 0, STEM_W_LEN);
        int32[] memory stemBias = _decodeBias(CodeBlobReader.read(stemB, 0, STEM_B_LEN), CHANNELS);
        y = _conv3x3(x, stemWb, stemBias, CIN, CHANNELS, true);
    }

    function _trunkForward(bytes memory y) internal view returns (bytes memory) {
        for (uint256 i = 0; i < BLOCKS; i++) {
            bytes memory w1 = CodeBlobReader.read(blockC1W[i], 0, BLK_W_LEN);
            int32[] memory b1 = _decodeBias(CodeBlobReader.read(blockC1B[i], 0, BLK_B_LEN), CHANNELS);
            bytes memory t = _conv3x3(y, w1, b1, CHANNELS, CHANNELS, true);

            bytes memory w2 = CodeBlobReader.read(blockC2W[i], 0, BLK_W_LEN);
            int32[] memory b2 = _decodeBias(CodeBlobReader.read(blockC2B[i], 0, BLK_B_LEN), CHANNELS);
            bytes memory u = _conv3x3(t, w2, b2, CHANNELS, CHANNELS, false);

            y = _addReluClamp(u, y);
        }
        return y;
    }

    function _policyForward(bytes memory y) internal view returns (bytes memory p) {
        bytes memory polWb = CodeBlobReader.read(policyW, 0, POL_W_LEN);
        int32[] memory polBias = _decodeBias(CodeBlobReader.read(policyB, 0, POL_B_LEN), POLICY_COUT);
        p = _conv1x1(y, polWb, polBias, CHANNELS, POLICY_COUT, true);
    }

    // -----------------------------
    // Input encoding
    // -----------------------------
    function _encodeInput(BoardState calldata s) internal pure returns (bytes memory out) {
        out = new bytes(INPUT_LEN);

        for (uint256 p = 0; p < 12; p++) {
            uint64 bb = s.bb[p];
            uint256 base = p * SQ;
            for (uint256 sq = 0; sq < SQ; sq++) {
                if (((bb >> sq) & 1) != 0) {
                    out[base + sq] = bytes1(uint8(1));
                }
            }
        }

        if (s.stm == 0) {
            uint256 base12 = 12 * SQ;
            for (uint256 sq = 0; sq < SQ; sq++) {
                out[base12 + sq] = bytes1(uint8(1));
            }
        }

        if ((s.castling & 1) != 0) _fillPlane(out, 13, 1);
        if ((s.castling & 2) != 0) _fillPlane(out, 14, 1);
        if ((s.castling & 4) != 0) _fillPlane(out, 15, 1);
        if ((s.castling & 8) != 0) _fillPlane(out, 16, 1);

        if (s.epFile >= 0) {
            uint256 file = uint256(uint8(s.epFile));
            uint256 rank = (s.stm == 0) ? 5 : 2;
            uint256 sq = file + rank * 8;
            out[17 * SQ + sq] = bytes1(uint8(1));
        }
    }

    function _fillPlane(bytes memory out, uint256 plane, uint8 value) internal pure {
        uint256 base = plane * SQ;
        for (uint256 sq = 0; sq < SQ; sq++) {
            out[base + sq] = bytes1(value);
        }
    }

    // -----------------------------
    // Layers
    // -----------------------------
    function _conv3x3(
        bytes memory x,
        bytes memory w,
        int32[] memory bias,
        uint8 cin,
        uint8 cout,
        bool relu
    ) internal view returns (bytes memory y) {
        y = new bytes(uint256(cout) * SQ);

        for (uint256 oc = 0; oc < cout; oc++) {
            uint256 ocBase = oc << 6;

            for (uint256 sq = 0; sq < SQ; sq++) {
                uint256 r0 = sq >> 3;
                uint256 f0 = sq & 7;

                int32 acc = 0;

                for (uint256 ic = 0; ic < cin; ic++) {
                    uint256 xBase = ic << 6;
                    uint256 wBase = (oc * uint256(cin) + ic) * 9;
                    acc += _acc3x3Ic(x, w, xBase, wBase, sq, r0, f0);
                }

                acc += bias[oc];
                acc = _rshiftRound(acc, SHIFT);

                int8 outv = _clampI8(acc);
                if (relu && outv < 0) outv = 0;

                y[ocBase + sq] = bytes1(uint8(outv));
            }
        }
    }

    function _acc3x3Ic(
        bytes memory x,
        bytes memory w,
        uint256 xBase,
        uint256 wBase,
        uint256 sq,
        uint256 r0,
        uint256 f0
    ) internal pure returns (int32 acc) {
        if (r0 != 0) {
            uint256 up = sq - 8;
            if (f0 != 0) acc += _mulB1(x[xBase + up - 1], w[wBase + 0]);
            acc += _mulB1(x[xBase + up],     w[wBase + 1]);
            if (f0 != 7) acc += _mulB1(x[xBase + up + 1], w[wBase + 2]);
        }

        if (f0 != 0) acc += _mulB1(x[xBase + sq - 1], w[wBase + 3]);
        acc += _mulB1(x[xBase + sq],     w[wBase + 4]);
        if (f0 != 7) acc += _mulB1(x[xBase + sq + 1], w[wBase + 5]);

        if (r0 != 7) {
            uint256 dn = sq + 8;
            if (f0 != 0) acc += _mulB1(x[xBase + dn - 1], w[wBase + 6]);
            acc += _mulB1(x[xBase + dn],     w[wBase + 7]);
            if (f0 != 7) acc += _mulB1(x[xBase + dn + 1], w[wBase + 8]);
        }
    }

    function _conv1x1(
        bytes memory x,
        bytes memory w,
        int32[] memory bias,
        uint8 cin,
        uint8 cout,
        bool relu
    ) internal view returns (bytes memory y) {
        y = new bytes(uint256(cout) * SQ);

        for (uint256 oc = 0; oc < cout; oc++) {
            uint256 ocBase = oc << 6;

            for (uint256 sq = 0; sq < SQ; sq++) {
                int32 acc = _dot1x1(x, w, cin, oc, sq);
                acc += bias[oc];
                acc = _rshiftRound(acc, SHIFT);

                int8 outv = _clampI8(acc);
                if (relu && outv < 0) outv = 0;

                y[ocBase + sq] = bytes1(uint8(outv));
            }
        }
    }

    function _dot1x1(bytes memory x, bytes memory w, uint8 cin, uint256 oc, uint256 sq)
        internal
        pure
        returns (int32 acc)
    {
        uint256 wBase = oc * uint256(cin);
        for (uint256 ic = 0; ic < cin; ic++) {
            uint256 idx = ic * SQ + sq;
            acc += _mulB1(x[idx], w[wBase + ic]);
        }
    }

    function _addReluClamp(bytes memory a, bytes memory b) internal pure returns (bytes memory y) {
        require(a.length == b.length, "len mismatch");
        y = new bytes(a.length);

        for (uint256 i = 0; i < a.length; i++) {
            int32 sum = int32(int8(uint8(a[i]))) + int32(int8(uint8(b[i])));
            int8 outv = _clampI8(sum);
            if (outv < 0) outv = 0;
            y[i] = bytes1(uint8(outv));
        }
    }

    // -----------------------------
    // FC argmax (chunked)
    // -----------------------------
    function _fcArgmaxChunked(bytes memory p, bytes calldata legalMask)
        internal
        view
        returns (uint16 bestIdx, int32 bestLogit)
    {
        require(p.length == FC_IN, "bad policy len");

        bytes memory fcBb = CodeBlobReader.read(fcB, 0, FC_B_LEN);

        bestIdx = 0;
        bestLogit = type(int32).min;

        for (uint256 ci = 0; ci < FC_W_CHUNKS; ci++) {
            uint256 chunkLen = (ci + 1 == FC_W_CHUNKS) ? FC_W_LAST_LEN : FC_W_CHUNK_LEN;
            uint256 rows = (ci + 1 == FC_W_CHUNKS) ? FC_ROWS_LAST_CHUNK : FC_ROWS_PER_CHUNK;

            bytes memory wChunk = CodeBlobReader.read(fcWChunks[ci], 0, chunkLen);

            uint256 miBase = ci * FC_ROWS_PER_CHUNK;
            (bestIdx, bestLogit) = _fcScanRows(miBase, rows, p, wChunk, fcBb, legalMask, bestIdx, bestLogit);
        }
    }

    function _fcScanRows(
        uint256 miBase,
        uint256 rows,
        bytes memory p,
        bytes memory wChunk,
        bytes memory fcBb,
        bytes calldata legalMask,
        uint16 bestIdx,
        int32 bestLogit
    ) internal pure returns (uint16, int32) {
        for (uint256 r = 0; r < rows; r++) {
            uint256 miU = miBase + r;
            if (miU >= MOVES) break;

            uint16 mi = uint16(miU);
            if (!legalMask.isSet(mi)) continue;

            int32 logit = _fcRowLogit(mi, r, p, wChunk, fcBb);
            if (logit > bestLogit) {
                bestLogit = logit;
                bestIdx = mi;
            }
        }
        return (bestIdx, bestLogit);
    }

    function _fcRowLogit(
        uint16 mi,
        uint256 r,
        bytes memory p,
        bytes memory wChunk,
        bytes memory fcBb
    ) internal pure returns (int32) {
        uint256 rowOff = r * FC_IN;
        int32 acc = _dot128(p, wChunk, rowOff);
        int32 bias = _readI32LE(fcBb, uint256(mi) * 4);
        return acc + bias;
    }

    function _dot128(bytes memory p, bytes memory wChunk, uint256 rowOff)
        internal
        pure
        returns (int32 acc)
    {
        for (uint256 j = 0; j < FC_IN; j++) {
            int32 xv = int32(int8(uint8(p[j])));
            int32 wv = int32(int8(uint8(wChunk[rowOff + j])));
            acc += xv * wv;
        }
    }

    // -----------------------------
    // FC topK (chunked)
    // -----------------------------
    function _fcTopKChunked(bytes memory p, bytes calldata legalMask, uint16 k)
        internal
        view
        returns (uint16[] memory idx, int32[] memory logits)
    {
        require(p.length == FC_IN, "bad policy len");

        idx = new uint16[](k);
        logits = new int32[](k);
        for (uint256 i = 0; i < k; i++) {
            idx[i] = 0;
            logits[i] = type(int32).min;
        }

        bytes memory fcBb = CodeBlobReader.read(fcB, 0, FC_B_LEN);

        for (uint256 ci = 0; ci < FC_W_CHUNKS; ci++) {
            uint256 chunkLen = (ci + 1 == FC_W_CHUNKS) ? FC_W_LAST_LEN : FC_W_CHUNK_LEN;
            uint256 rows = (ci + 1 == FC_W_CHUNKS) ? FC_ROWS_LAST_CHUNK : FC_ROWS_PER_CHUNK;

            bytes memory wChunk = CodeBlobReader.read(fcWChunks[ci], 0, chunkLen);

            uint256 miBase = ci * FC_ROWS_PER_CHUNK;
            _fcTopKScanRows(miBase, rows, p, wChunk, fcBb, legalMask, idx, logits);
        }
    }

    function _fcTopKScanRows(
        uint256 miBase,
        uint256 rows,
        bytes memory p,
        bytes memory wChunk,
        bytes memory fcBb,
        bytes calldata legalMask,
        uint16[] memory idx,
        int32[] memory logits
    ) internal pure {
        uint256 k = idx.length;

        for (uint256 r = 0; r < rows; r++) {
            uint256 miU = miBase + r;
            if (miU >= MOVES) break;

            uint16 mi = uint16(miU);
            if (!legalMask.isSet(mi)) continue;

            int32 logit = _fcRowLogit(mi, r, p, wChunk, fcBb);
            _insertTopK(mi, logit, idx, logits, k);
        }
    }

    function _insertTopK(
        uint16 mi,
        int32 logit,
        uint16[] memory idx,
        int32[] memory logits,
        uint256 k
    ) internal pure {
        for (uint256 i = 0; i < k; i++) {
            if (logit > logits[i]) {
                for (uint256 j = k - 1; j > i; j--) {
                    idx[j] = idx[j - 1];
                    logits[j] = logits[j - 1];
                }
                idx[i] = mi;
                logits[i] = logit;
                return;
            }
        }
    }

    // -----------------------------
    // Material heuristics
    // -----------------------------
    function _captureValueFromDelta(BoardState calldata root, BoardState calldata child)
        internal
        pure
        returns (uint16)
    {
        uint8 oppSide = (root.stm == 0) ? 1 : 0;
        uint256 base = (oppSide == 0) ? 0 : 6;

        if ((root.bb[base + 4] & ~child.bb[base + 4]) != 0) return 900;
        if ((root.bb[base + 3] & ~child.bb[base + 3]) != 0) return 500;
        if ((root.bb[base + 2] & ~child.bb[base + 2]) != 0) return 330;
        if ((root.bb[base + 1] & ~child.bb[base + 1]) != 0) return 320;
        if ((root.bb[base + 0] & ~child.bb[base + 0]) != 0) return 100;
        return 0;
    }

    function _maxCaptureValue(BoardState calldata s, bytes calldata mask)
        internal
        pure
        returns (uint16 maxVal)
    {
        uint8 victimSide = (s.stm == 0) ? 1 : 0;

        for (uint256 bi = 0; bi < 584; bi++) {
            uint8 b = uint8(mask[bi]);
            if (b == 0) continue;

            uint16 baseMove = uint16(bi * 8);
            for (uint8 bit = 0; bit < 8; bit++) {
                if ((b & (uint8(1) << bit)) == 0) continue;

                uint16 mi = baseMove + bit;
                if (mi >= MOVES) break;

                uint16 v = _captureValueForMove(s, mi, victimSide);
                if (v > maxVal) {
                    maxVal = v;
                    if (maxVal == 900) return 900;
                }
            }
        }
    }

    function _captureValueForMove(BoardState calldata s, uint16 mi, uint8 victimSide)
        internal
        pure
        returns (uint16)
    {
        uint8 from = uint8(uint256(mi) / 73);
        uint8 plane = uint8(uint256(mi) % 73);

        uint8 ff = from & 7;
        uint8 fr = from >> 3;

        int8 dx;
        int8 dy;
        uint8 dist = 1;

        if (plane < 56) {
            uint8 dir = plane / 7;
            dist = (plane % 7) + 1;
            (dx, dy) = _dirVec(dir);
        } else if (plane < 64) {
            uint8 k = plane - 56;
            (dx, dy) = _knightVec(k);
        } else {
            uint8 ud = plane - 64; // 0..8
            uint8 dir3 = ud % 3;   // 0 forward, 1 cap-left, 2 cap-right
            dy = (s.stm == 0) ? int8(1) : int8(-1);
            if (dir3 == 0) dx = 0;
            else if (dir3 == 1) dx = (s.stm == 0) ? int8(-1) : int8(1);
            else dx = (s.stm == 0) ? int8(1) : int8(-1);
        }

        int8 tf = int8(uint8(ff)) + dx * int8(uint8(dist));
        int8 tr = int8(uint8(fr)) + dy * int8(uint8(dist));
        if (tf < 0 || tf > 7 || tr < 0 || tr > 7) return 0;

        uint8 to = uint8(uint8(tr) * 8 + uint8(tf));

        uint16 v = _pieceValueAt(s, victimSide, to);
        if (v != 0) return v;

        // en passant capture
        if (s.epFile >= 0) {
            uint8 epFile = uint8(uint8(s.epFile));
            uint8 epRank = (s.stm == 0) ? 5 : 2;
            uint8 epSq = epFile + epRank * 8;

            if (to == epSq) {
                uint256 pawnPlane = (s.stm == 0) ? 0 : 6;
                if (((s.bb[pawnPlane] >> from) & 1) != 0) {
                    uint8 tfu = uint8(tf);
                    uint8 tru = uint8(tr);
                    uint8 df = (tfu > ff) ? (tfu - ff) : (ff - tfu);

                    bool forwardOk = (s.stm == 0) ? (tru == fr + 1) : (fr == tru + 1);
                    if (df == 1 && forwardOk) return 100;
                }
            }
        }

        return 0;
    }

    function _pieceValueAt(BoardState calldata s, uint8 side, uint8 sq)
        internal
        pure
        returns (uint16)
    {
        uint256 base = (side == 0) ? 0 : 6;

        if (((s.bb[base + 4] >> sq) & 1) != 0) return 900;
        if (((s.bb[base + 3] >> sq) & 1) != 0) return 500;
        if (((s.bb[base + 2] >> sq) & 1) != 0) return 330;
        if (((s.bb[base + 1] >> sq) & 1) != 0) return 320;
        if (((s.bb[base + 0] >> sq) & 1) != 0) return 100;
        return 0;
    }

    function _dirVec(uint8 d) internal pure returns (int8 dx, int8 dy) {
        if (d == 0) return (0, 1);
        if (d == 1) return (1, 1);
        if (d == 2) return (1, 0);
        if (d == 3) return (1, -1);
        if (d == 4) return (0, -1);
        if (d == 5) return (-1, -1);
        if (d == 6) return (-1, 0);
        return (-1, 1);
    }

    function _knightVec(uint8 k) internal pure returns (int8 dx, int8 dy) {
        if (k == 0) return (1, 2);
        if (k == 1) return (2, 1);
        if (k == 2) return (2, -1);
        if (k == 3) return (1, -2);
        if (k == 4) return (-1, -2);
        if (k == 5) return (-2, -1);
        if (k == 6) return (-2, 1);
        return (-1, 2);
    }

    // -----------------------------
    // Bias + quant helpers
    // -----------------------------
    function _decodeBias(bytes memory bb, uint256 n) internal pure returns (int32[] memory out) {
        require(bb.length == n * 4, "bad bias blob");
        out = new int32[](n);
        for (uint256 i = 0; i < n; i++) out[i] = _readI32LE(bb, i * 4);
    }

    function _readI32LE(bytes memory data, uint256 off) internal pure returns (int32 v) {
        uint32 u =
            uint32(uint8(data[off])) |
            (uint32(uint8(data[off + 1])) << 8) |
            (uint32(uint8(data[off + 2])) << 16) |
            (uint32(uint8(data[off + 3])) << 24);
        v = int32(u);
    }

    function _rshiftRound(int32 x, uint8 shift) internal pure returns (int32) {
        if (shift == 0) return x;
        int32 off = int32(1) << (shift - 1);
        if (x >= 0) x += off;
        else x -= off;
        return x >> shift;
    }

    function _clampI8(int32 x) internal pure returns (int8) {
        if (x > 127) return 127;
        if (x < -128) return -128;
        return int8(x);
    }

    function _mulB1(bytes1 xb, bytes1 wb) internal pure returns (int32) {
        return int32(int8(uint8(xb))) * int32(int8(uint8(wb)));
    }
}

