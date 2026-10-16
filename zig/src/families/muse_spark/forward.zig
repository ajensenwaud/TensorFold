//! Muse Spark's decode forward on one GPU: windows of decode rows (any slots, any positions), EXL3 projections and
//! experts (the MiMo-V2 kernels), hand CUDA attention and glue (kernels/cuda/muse_spark). Under TP=2 each rank runs
//! its heads, latent half and expert width; four all-gathers a layer (docs/ZIG-MUSE.md in the recipe).

const std = @import("std");
const cuda = @import("cuda");
const tp = @import("tp");
const load_ = @import("load.zig");
const exl3 = cuda.exl3;
const Weights = load_.Weights;
const Stream = cuda.Stream;

pub const CK = 256; // keys an attention chunk
pub const DECODE_ROWS = 128; // the EXL3 GEMV's row limit
pub const NORM_THREADS = 1024;
pub const BUCKET = 4096; // graphs: the smallest key bucket (full layers read bucket / CK chunks)
pub const MAX_GRAPH_ROWS = 32; // wider windows (prompt chunks) run eagerly

/// One decode row: a stream's cache slot, its position, its input token.
pub const Row = struct { slot: u32, pos: u32, token: i64 };

/// Sliding cache rows a slot keeps: the window, the row itself and a window's other rows of the same slot.
pub fn ringSize(window: usize, rows: usize) usize {
    return std.mem.alignForward(usize, window + 1 + rows, 64);
}

fn u(x: usize) u32 {
    return @intCast(x);
}
fn ci(x: usize) c_int {
    return @intCast(x);
}

const Bufs = struct {
    ids: u64, // i64 [R]
    pos: u64, // i32 [R]
    slot: u64, // i32 [R]
    res: u64, // f32 [R, D]
    x: u64, // f16 [R, D]: the normed sublayer input
    q: u64, // f16 [R, H * 64]
    k: u64, // f16 [R, HK * 64]
    v: u64,
    g: u64, // f16 [R, H * 64]: the output gate's logits
    qo: u64, // f16 [R, H * 64]
    att: u64, // f16 [R, H * 64]: gated attention output
    po: u64, // f32 [R, H, NCH, 64]
    pm: u64,
    pl: u64,
    xh: u64, // f16 rot_in scratch [R, max K]
    z: u64, // f32 split partials
    part: u64, // f32 [R, D]
    logits: u64, // f32 [R, E]
    pick: u64, // i32 [R, K]
    wts: u64, // f32 [R, K]
    lat: u64, // f16 [R, latent]
    latn: u64, // f16 [R, latent]
    acc: u64, // f16 [R, latent]
    head: u64, // f32 [R, head stride]: this rank's head rows
    recv: u64, // f32 [world, R, max(D, K * latent, head stride)]: all-gathers
    latg: u64, // f16 [world, R, latent / world]: the gathered latent halves
    am: u64, // i32 [R]
    sc: exl3.Scratch,
};

pub const Model = struct {
    w: *const Weights,
    d: *const cuda.Driver,
    k3: *const exl3.Kernels,
    comm: ?tp.Collective,
    glue: cuda.Module,
    attn: cuda.Module,
    f_embed: cuda.Function,
    f_rms: cuda.Function,
    f_gate: cuda.Function,
    f_router: cuda.Function,
    f_topk: cuda.Function,
    f_esum: cuda.Function,
    f_softcap: cuda.Function,
    f_argmax: cuda.Function,
    f_prep: cuda.Function,
    f_rows: cuda.Function,
    f_merge: cuda.Function,
    context: usize,
    graphs: std.AutoHashMapUnmanaged([2]u32, cuda.graph.Exec) = .empty,
    replays: usize = 0,
    captures: usize = 0,
    ring: usize,
    rows: usize,
    slots: usize,
    nch_max: usize,
    kc: []u64,
    vc: []u64,
    cos: u64, // fp32 [context, 32] (one theta: every RoPE layer's)
    sin: u64,
    theta: f64,
    b: Bufs,
    arena: load_.Arena,
    gpa: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const Weights, k3: *const exl3.Kernels, comm: ?tp.Collective, context: usize, rows: usize, slots: usize, s: Stream) !Model {
        const cfg = w.cfg;
        if (w.world > 1 and comm == null) return error.NeedComm;
        if (rows > DECODE_ROWS or rows == 0) return error.TooManyRows;
        var m: Model = undefined;
        m.w = w;
        m.d = d;
        m.k3 = k3;
        m.comm = if (w.world > 1) comm else null;
        m.context = context;
        m.graphs = .empty;
        m.replays = 0;
        m.captures = 0;
        m.ring = ringSize(cfg.window, rows);
        m.rows = rows;
        m.slots = slots;
        m.nch_max = @max((context + CK - 1) / CK, (cfg.window + 1 + CK - 1) / CK);
        m.gpa = gpa;
        m.arena = .{ .d = d, .gpa = gpa };
        errdefer m.arena.deinit();
        m.glue = try cuda.Module.load(d, cuda.kernels.muse_glue);
        errdefer m.glue.unload();
        m.attn = try cuda.Module.load(d, cuda.kernels.muse_attention);
        errdefer m.attn.unload();
        m.f_embed = try m.glue.function("muse_embed_norm");
        m.f_rms = try m.glue.function("muse_rms");
        m.f_gate = try m.glue.function("muse_gate_norm");
        m.f_router = try m.glue.function("muse_router");
        m.f_topk = try m.glue.function("muse_topk");
        m.f_esum = try m.glue.function("muse_expert_sum");
        m.f_softcap = try m.glue.function("muse_softcap");
        m.f_argmax = try m.glue.function("muse_argmax");
        m.f_prep = try m.attn.function("muse_qk_prep");
        m.f_rows = try m.attn.function("muse_attn_rows");
        m.f_merge = try m.attn.function("muse_attn_merge");
        const ar = &m.arena;
        const L = w.layers.len;
        const HK = cfg.kv_heads;
        const hd = cfg.head_dim;
        m.kc = try gpa.alloc(u64, L);
        m.vc = try gpa.alloc(u64, L);
        for (0..L) |i| {
            const n = if (cfg.full[i]) context else m.ring;
            m.kc[i] = try ar.zeros(slots * n * HK * hd * 2);
            m.vc[i] = try ar.zeros(slots * n * HK * hd * 2);
        }
        // RoPE tables: every RoPE layer of Muse Spark has the same theta (5e5); refuse a second one
        m.theta = 0;
        for (0..L) |i| if (cfg.rope(i)) {
            if (m.theta != 0 and m.theta != cfg.theta[i]) return error.SeveralThetas;
            m.theta = cfg.theta[i];
        };
        m.cos = try ar.alloc(context * (hd / 2) * 4);
        m.sin = try ar.alloc(context * (hd / 2) * 4);
        if (m.theta != 0) {
            const f = try m.glue.function("muse_rope_tables");
            var a: cuda.Args = .{};
            a.add(m.cos);
            a.add(m.sin);
            a.add(ci(context));
            a.add(m.theta);
            a.add(ci(hd));
            try cuda.launch.launch(f, .{ .grid = .{ .x = 256, .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
        }
        const R = rows;
        const D = cfg.hidden;
        const HQ = cfg.heads * hd;
        const HKD = HK * hd;
        const lat = cfg.latent;
        const vr = w.head_stride;
        var max_z: usize = zBytes(w.head, R);
        for (w.layers) |ly| inline for (.{ "q", "k", "v", "g", "o", "pre", "post" }) |f| {
            max_z = @max(max_z, zBytes(@field(ly, f), R));
        };
        var b: Bufs = undefined;
        b.ids = try ar.zeros(R * 8);
        b.pos = try ar.zeros(R * 4);
        b.slot = try ar.zeros(R * 4);
        b.res = try ar.zeros(R * D * 4);
        b.x = try ar.zeros(R * D * 2);
        b.q = try ar.zeros(R * HQ * 2);
        b.k = try ar.zeros(R * HKD * 2);
        b.v = try ar.zeros(R * HKD * 2);
        b.g = try ar.zeros(R * HQ * 2);
        b.qo = try ar.zeros(R * HQ * 2);
        b.att = try ar.zeros(R * HQ * 2);
        b.po = try ar.zeros(R * cfg.heads * m.nch_max * hd * 4);
        b.pm = try ar.zeros(R * cfg.heads * m.nch_max * 4);
        b.pl = try ar.zeros(R * cfg.heads * m.nch_max * 4);
        b.xh = try ar.zeros(R * @max(D, HQ) * 2);
        b.z = try ar.zeros(@max(max_z, 4));
        b.part = try ar.zeros(R * D * 4);
        b.logits = try ar.zeros(R * cfg.experts * 4);
        b.pick = try ar.zeros(R * cfg.top_k * 4);
        b.wts = try ar.zeros(R * cfg.top_k * 4);
        b.lat = try ar.zeros(R * lat * 2);
        b.latn = try ar.zeros(R * lat * 2);
        b.acc = try ar.zeros(R * lat * 2);
        b.head = try ar.zeros(R * vr * 4);
        b.recv = if (w.world > 1) try ar.zeros(w.world * R * @max(@max(D, cfg.top_k * lat), vr) * 4) else 0;
        b.latg = if (w.world > 1) try ar.zeros(R * lat * 2) else 0;
        b.am = try ar.zeros(R * 4);
        const first = w.layers[0].experts;
        const cfg_gu = try exl3.defaultConfig(first.dims, first.width, true);
        const cfg_d = try exl3.defaultConfig(first.width, first.dims, false);
        const sz = exl3.Scratch.bytes(first, R, cfg.top_k, cfg_gu, cfg_d);
        b.sc = .{ .xg = try ar.zeros(sz.xg), .xu = try ar.zeros(sz.xg), .xd = try ar.zeros(sz.xd), .z = try ar.zeros(sz.z), .y = try ar.zeros(sz.y), .ids = try ar.zeros(sz.ids), .count = try ar.zeros(4), .members = try ar.alloc(sz.members), .rows = R, .slots = cfg.top_k, .cfg_gu = cfg_gu, .cfg_d = cfg_d };
        try d.check(d.api.cuMemsetD32_v2(b.sc.members, 0xFFFF_FFFF, sz.members / 4), "members -1");
        m.b = b;
        try s.synchronize();
        return m;
    }

    fn zBytes(l: exl3.Linear, R: usize) usize {
        return if (l.split[0] > 1) l.split[0] * R * l.n * 4 else 0;
    }

    pub fn deinit(m: *Model) void {
        var it = m.graphs.valueIterator();
        while (it.next()) |e| e.deinit();
        m.graphs.deinit(m.gpa);
        m.gpa.free(m.kc);
        m.gpa.free(m.vc);
        m.glue.unload();
        m.attn.unload();
        m.arena.deinit();
    }

    pub const Tap = enum { input, attn, mlp };
    /// A hook per layer: `.input` before the layer (res may be overwritten: teacher forcing renormalises after it),
    /// `.attn` after the attention gate, `.mlp` the block output. `dev`: fp32 [n, D].
    pub const Capture = struct {
        hook: *const fn (ctx: *anyopaque, what: Tap, layer: usize, dev: u64, n: usize) anyerror!void,
        ctx: *anyopaque,
        teacher: bool = false,
    };

    /// Rows' tokens, positions and slots into the fixed device inputs.
    pub fn stage(m: *Model, rows: []const Row) !void {
        if (rows.len == 0 or rows.len > m.rows) return error.WindowTooWide;
        var ids: [DECODE_ROWS]i64 = undefined;
        var pos: [DECODE_ROWS]i32 = undefined;
        var slot: [DECODE_ROWS]i32 = undefined;
        var per: [DECODE_ROWS]usize = @splat(0);
        for (rows, 0..) |r, i| {
            if (r.slot >= m.slots) return error.BadSlot;
            if (r.pos >= m.context) return error.PastContext;
            if (r.token < 0 or r.token >= m.w.full.vocab) return error.BadToken;
            ids[i] = r.token;
            pos[i] = @intCast(r.pos);
            slot[i] = @intCast(r.slot);
            per[r.slot] += 1;
        }
        for (per[0..m.slots]) |c| if (c + m.w.cfg.window + 1 > m.ring) return error.RingOverrun;
        const n = rows.len;
        try upload(m.d, m.b.ids, std.mem.sliceAsBytes(ids[0..n]));
        try upload(m.d, m.b.pos, std.mem.sliceAsBytes(pos[0..n]));
        try upload(m.d, m.b.slot, std.mem.sliceAsBytes(slot[0..n]));
    }

    /// Eager decode window: softcapped fp32 logits of every row into `out` [n, vocab] (or none).
    pub fn forward(m: *Model, s: Stream, rows: []const Row, out: ?[]f32, cap: ?Capture) !void {
        try m.stage(rows);
        var end: usize = 0;
        for (rows) |r| end = @max(end, r.pos);
        try m.run(s, rows.len, end / CK + 1, cap);
        if (out) |o| try m.logitsTo(s, rows.len, o) else try s.synchronize();
    }

    /// Graphed decode window: a graph per (rows, key bucket), captured after an eager warm-up of the same window;
    /// full layers read bucket / CK chunks (empty chunks merge to nothing, so replay bits equal eager bits).
    pub fn graphed(m: *Model, s: Stream, rows: []const Row, out: ?[]f32) !void {
        if (rows.len > MAX_GRAPH_ROWS) return m.forward(s, rows, out, null); // prompt windows: eager, the same bits
        var end: usize = 0;
        for (rows) |r| end = @max(end, r.pos + 1);
        const full = (m.context + CK - 1) / CK * CK;
        const bucket = @min(full, @max(BUCKET, std.math.ceilPowerOfTwo(usize, end) catch unreachable));
        const key: [2]u32 = .{ @intCast(rows.len), @intCast(bucket) };
        try m.stage(rows);
        if (m.graphs.get(key)) |exec| {
            try exec.launchOn(s);
            m.replays += 1;
        } else {
            try m.run(s, rows.len, bucket / CK, null); // warm-up: writes the same cache rows the replay will
            try s.synchronize();
            try cuda.graph.beginCapture(s, .thread_local);
            m.run(s, rows.len, bucket / CK, null) catch |e| {
                if (cuda.graph.endCapture(s)) |g| {
                    var gg = g;
                    gg.deinit();
                } else |_| {}
                return e;
            };
            var g = try cuda.graph.endCapture(s);
            defer g.deinit();
            const exec = try g.instantiate();
            try m.graphs.put(m.gpa, key, exec);
            try exec.launchOn(s);
            m.captures += 1;
        }
        if (out) |o| try m.logitsTo(s, rows.len, o) else try s.synchronize();
    }

    fn norm(m: *Model, s: Stream, x: u64, x_half: bool, w: u64, out: u64, n: usize, dim: usize, eps: f64) !void {
        return m.normParts(s, x, x_half, w, out, n, dim, eps, 0);
    }

    /// RMS norm of n rows; `parts` > 0: x is the ranks' gathered column slices of that width.
    fn normParts(m: *Model, s: Stream, x: u64, x_half: bool, w: u64, out: u64, n: usize, dim: usize, eps: f64, parts: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(c_int, @intFromBool(x_half)));
        a.add(w);
        a.add(out);
        a.add(ci(dim));
        a.add(@as(f32, @floatCast(eps)));
        a.add(ci(parts));
        try cuda.launch.launch(m.f_rms, .{ .grid = .{ .x = u(n), .y = 1, .z = 1 }, .block = .{ .x = NORM_THREADS, .y = 1, .z = 1 } }, s, &a);
    }

    /// The ranks' fp32 partials [n, count]: all-gathered into recv (rank-major); returns (rank 0's, rank 1's or 0).
    fn gather(m: *Model, s: Stream, part: u64, count: usize) ![2]u64 {
        const c = m.comm orelse return .{ part, 0 };
        try c.allGather(part, m.b.recv, count, .f32, s.handle);
        return .{ m.b.recv, m.b.recv + count * 4 };
    }

    /// res = res * alpha + rms(y') * beta with y = the ranks' partials summed in rank order, then x = half(rms(res) *
    /// w_next) (muse_gate_norm).
    fn gateNorm(m: *Model, s: Stream, part: u64, scale: u64, ab: u64, w_next: u64, n: usize) !void {
        const cfg = m.w.cfg;
        const y = try m.gather(s, part, n * cfg.hidden);
        var a: cuda.Args = .{};
        a.add(m.b.res);
        a.add(y[0]);
        a.add(y[1]);
        a.add(scale);
        a.add(@as(c_int, @intFromBool(scale != 0)));
        a.add(@as(f32, @floatCast(cfg.post_eps)));
        a.add(ab);
        a.add(w_next);
        a.add(@as(f32, @floatCast(cfg.eps)));
        a.add(m.b.x);
        a.add(ci(cfg.hidden));
        try cuda.launch.launch(m.f_gate, .{ .grid = .{ .x = u(n), .y = 1, .z = 1 }, .block = .{ .x = NORM_THREADS, .y = 1, .z = 1 } }, s, &a);
    }

    /// An EXL3 projection of n rows: x [n, K] fp16 -> out [n, N] fp16 or fp32.
    fn proj(m: *Model, s: Stream, l: exl3.Linear, x: u64, out: u64, out_f32: bool, n: usize) !void {
        try m.k3.linear(s, l, x, .f16, m.b.xh, out, if (out_f32) .f32 else .f16, if (l.split[0] > 1) m.b.z else 0, n);
    }

    fn attention(m: *Model, s: Stream, i: usize, n: usize, chunks: usize) !void {
        const cfg = m.w.cfg;
        const H = cfg.heads;
        const HK = cfg.kv_heads;
        const full = cfg.full[i];
        const len = if (full) m.context else m.ring;
        {
            var a: cuda.Args = .{};
            inline for (.{ m.b.q, m.b.k, m.b.v, m.cos, m.sin }) |p| a.add(p);
            a.add(@as(c_int, @intFromBool(cfg.rope(i))));
            inline for (.{ m.b.pos, m.b.slot, m.b.qo, m.kc[i], m.vc[i] }) |p| a.add(p);
            a.add(ci(len));
            a.add(ci(H));
            a.add(ci(HK));
            a.add(@as(f32, @floatCast(cfg.eps)));
            try cuda.launch.launch(m.f_prep, .{ .grid = .{ .x = u(n), .y = u(H + HK), .z = 1 }, .block = .{ .x = 32, .y = 1, .z = 1 } }, s, &a);
        }
        const win: usize = if (full) 0 else cfg.window;
        const nch: usize = if (full) chunks else (cfg.window + 1 + CK - 1) / CK;
        {
            var a: cuda.Args = .{};
            inline for (.{ m.b.qo, m.kc[i], m.vc[i], m.b.pos, m.b.slot, m.b.po, m.b.pm, m.b.pl }) |p| a.add(p);
            a.add(ci(len));
            a.add(ci(nch));
            a.add(ci(CK));
            a.add(ci(win));
            a.add(ci(H));
            a.add(ci(HK));
            a.add(@as(f32, @floatCast(cfg.smScale())));
            try cuda.launch.launch(m.f_rows, .{ .grid = .{ .x = u(n), .y = u(HK), .z = u(nch) }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
        }
        {
            var a: cuda.Args = .{};
            inline for (.{ m.b.po, m.b.pm, m.b.pl }) |p| a.add(p);
            a.add(ci(nch));
            a.add(m.b.g);
            a.add(m.b.att);
            a.add(ci(H));
            a.add(@as(f32, @floatCast(cfg.eps)));
            a.add(@as(c_int, @intFromBool(full)));
            try cuda.launch.launch(m.f_merge, .{ .grid = .{ .x = u(n), .y = u(H / 8), .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
        }
    }

    /// MuseSparkMoE on the normed x: router (fp32), latent projection and norm, routed experts with the
    /// per-expert norm, back to the model width into part (fp32).
    fn moe(m: *Model, s: Stream, L: load_.Layer, n: usize) !void {
        const cfg = m.w.cfg;
        const E = cfg.experts;
        const K = cfg.top_k;
        {
            var a: cuda.Args = .{};
            a.add(m.b.x);
            a.add(L.router);
            a.add(@as(c_int, @intFromBool(L.router_bf16)));
            a.add(m.b.logits);
            a.add(ci(cfg.hidden));
            a.add(ci(E));
            try cuda.launch.launch(m.f_router, .{ .grid = .{ .x = u((E + 7) / 8), .y = u(n), .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
        }
        {
            var a: cuda.Args = .{};
            a.add(m.b.logits);
            a.add(L.bias);
            a.add(ci(E));
            a.add(ci(K));
            a.add(@as(f32, @floatCast(cfg.out_mult)));
            a.add(@as(f32, @floatCast(cfg.route_scale)));
            a.add(m.b.pick);
            a.add(m.b.wts);
            const t = std.math.ceilPowerOfTwo(usize, @max(E, 32)) catch unreachable;
            try cuda.launch.launch(m.f_topk, .{ .grid = .{ .x = u(n), .y = 1, .z = 1 }, .block = .{ .x = u(t), .y = 1, .z = 1 } }, s, &a);
        }
        try m.proj(s, L.pre, m.b.x, m.b.lat, false, n);
        if (m.comm) |c| {
            const half = L.pre.n; // this rank's latent outputs
            try c.allGather(m.b.lat, m.b.latg, n * half, .f16, s.handle);
            try m.normParts(s, m.b.latg, true, L.pre_norm, m.b.latn, n, cfg.latent, cfg.eps, half);
        } else try m.norm(s, m.b.lat, true, L.pre_norm, m.b.latn, n, cfg.latent, cfg.eps);
        // routed experts: group, rotate, gate / up, SwiGLU (fp32), down; per-slot outputs in sc.y [n * K, latent]
        const k3 = m.k3;
        const e = L.experts;
        const sc = m.b.sc;
        const p = n * K;
        const w = sc.window(E, n);
        try k3.group(s, m.b.pick, sc.ids, sc.count, sc.members, n, K, E, w[1]);
        try k3.expertsRotIn(s, m.b.latn, false, cfg.latent, m.b.pick, e.suh_g, e.suh_u, sc.xg, sc.xu, n, e.dims, K, E);
        const gu = sc.cfg_gu;
        try k3.grouped(s, .{ .x0 = sc.xg, .x1 = sc.xu, .tp0 = e.gate_ptr, .tp1 = e.up_ptr, .k2_0 = e.gate_k2, .k2_1 = e.up_k2, .uids = sc.ids, .ucount = sc.count, .members = sc.members, .z = sc.z, .mats = 2, .k = e.dims, .n = e.width, .p = p, .sk = gu[2], .slots = K, .maxm = w[1], .nexp_max = w[0], .nt = gu[0], .warps = gu[1], .pf = gu[3], .lo = e.k2_gu[0], .hi = e.k2_gu[1] });
        try k3.gateupEpilogue(s, sc.z, m.b.pick, e.svh_g, e.svh_u, e.suh_d, sc.xd, n, p, e.width, gu[2], K, E, std.math.inf(f32), 1);
        const dn = sc.cfg_d;
        try k3.grouped(s, .{ .x0 = sc.xd, .x1 = sc.xd, .tp0 = e.down_ptr, .tp1 = e.down_ptr, .k2_0 = e.down_k2, .k2_1 = e.down_k2, .uids = sc.ids, .ucount = sc.count, .members = sc.members, .z = sc.z, .mats = 1, .k = e.width, .n = e.dims, .p = p, .sk = dn[2], .slots = K, .maxm = w[1], .nexp_max = w[0], .nt = dn[0], .warps = dn[1], .pf = dn[3], .lo = e.k2_d[0], .hi = e.k2_d[1] });
        try k3.downEpilogue(s, sc.z, m.b.pick, e.svh_d, sc.y, n, p, e.dims, dn[2], K, E);
        {
            const y = try m.gather(s, sc.y, p * e.dims); // each rank's partial over its half of every expert's width
            var a: cuda.Args = .{};
            a.add(y[0]);
            a.add(y[1]);
            a.add(m.b.wts);
            a.add(L.expert_norm);
            a.add(@as(f32, @floatCast(cfg.post_eps)));
            a.add(m.b.acc);
            a.add(ci(cfg.latent));
            a.add(ci(K));
            a.add(ci(m.w.rank * L.post.k)); // this rank's post_expert_proj inputs
            a.add(ci(L.post.k));
            try cuda.launch.launch(m.f_esum, .{ .grid = .{ .x = u(n), .y = 1, .z = 1 }, .block = .{ .x = NORM_THREADS, .y = 1, .z = 1 } }, s, &a);
        }
        try m.proj(s, L.post, m.b.acc, m.b.part, true, n);
    }

    /// The window's kernels on staged inputs, no host work; full layers read `chunks` key chunks of CK.
    pub fn run(m: *Model, s: Stream, n: usize, chunks: usize, cap: ?Capture) !void {
        const cfg = m.w.cfg;
        const D = cfg.hidden;
        if (chunks > m.nch_max) return error.TooManyChunks;
        {
            var a: cuda.Args = .{};
            a.add(m.w.embed);
            a.add(m.b.ids);
            a.add(m.b.res);
            a.add(ci(D));
            a.add(@as(f32, @floatCast(cfg.eps)));
            try cuda.launch.launch(m.f_embed, .{ .grid = .{ .x = u(n), .y = 1, .z = 1 }, .block = .{ .x = NORM_THREADS, .y = 1, .z = 1 } }, s, &a);
        }
        try m.norm(s, m.b.res, false, m.w.layers[0].in_norm, m.b.x, n, D, cfg.eps);
        for (m.w.layers, 0..) |L, i| {
            if (cap) |c| if (c.teacher) {
                try c.hook(c.ctx, .input, i, m.b.res, n);
                try m.norm(s, m.b.res, false, L.in_norm, m.b.x, n, D, cfg.eps);
            };
            try m.proj(s, L.q, m.b.x, m.b.q, false, n);
            try m.proj(s, L.k, m.b.x, m.b.k, false, n);
            try m.proj(s, L.v, m.b.x, m.b.v, false, n);
            try m.proj(s, L.g, m.b.x, m.b.g, false, n);
            try m.attention(s, i, n, chunks);
            try m.proj(s, L.o, m.b.att, m.b.part, true, n);
            try m.gateNorm(s, m.b.part, 0, L.attn_ab, L.ffn_norm, n);
            if (cap) |c| try c.hook(c.ctx, .attn, i, m.b.res, n);
            try m.moe(s, L, n);
            const after = if (i + 1 < m.w.layers.len) m.w.layers[i + 1].in_norm else m.w.norm;
            try m.gateNorm(s, m.b.part, L.post_ffn, L.mlp_ab, after, n);
            if (cap) |c| try c.hook(c.ctx, .mlp, i, m.b.res, n);
        }
        try m.proj(s, m.w.head, m.b.x, m.b.head, true, n);
        const cfgf = m.w.full;
        var a: cuda.Args = .{};
        a.add(m.b.head);
        a.add(@as(i64, @intCast(n * m.w.head.n)));
        a.add(@as(f32, @floatCast(cfgf.out_mult)));
        a.add(@as(f32, @floatCast(cfgf.softcap)));
        try cuda.launch.launch(m.f_softcap, .{ .grid = .{ .x = u(@min((n * m.w.head.n + 255) / 256, 4096)), .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
        // each rank's [n, its rows] block, gathered as n * stride elements a rank (rank 1's slice is a block shorter)
        if (m.comm) |c| try c.allGather(m.b.head, m.b.recv, n * m.w.head_stride, .f32, s.handle);
    }

    /// Where the rows' head slices are: (rank 0's [n, w0], rank 1's [n, w1] offset in elements, w0, w1).
    fn heads(m: *const Model, n: usize) struct { base: u64, part: usize, w0: usize, w1: usize } {
        if (m.comm == null) return .{ .base = m.b.head, .part = 0, .w0 = m.w.head.n, .w1 = 0 };
        const st = m.w.head_stride;
        return .{ .base = m.b.recv, .part = n * st, .w0 = st, .w1 = m.w.head_rows - st }; // rank 1: the padded head's rest
    }

    /// Each row's argmax over the real vocabulary on the device; only the ids reach the host.
    pub fn argmaxTo(m: *Model, s: Stream, n: usize, out: []i32) !void {
        const h = m.heads(n);
        var a: cuda.Args = .{};
        a.add(h.base);
        a.add(m.b.am);
        a.add(ci(h.w0));
        a.add(ci(h.w1));
        a.add(@as(i64, @intCast(h.part)));
        a.add(ci(m.w.full.vocab));
        try cuda.launch.launch(m.f_argmax, .{ .grid = .{ .x = u(n), .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
        try s.synchronize();
        try m.d.check(m.d.api.cuMemcpyDtoH_v2(@ptrCast(out.ptr), m.b.am, n * 4), "argmax");
    }

    /// The window's logits on the host: [n, vocab] (the head's padding rows dropped).
    pub fn logitsTo(m: *Model, s: Stream, n: usize, out: []f32) !void {
        const V = m.w.full.vocab;
        const h = m.heads(n);
        try s.synchronize();
        for (0..n) |r| {
            const c0 = @min(V, h.w0);
            try m.d.check(m.d.api.cuMemcpyDtoH_v2(@ptrCast(out[r * V ..].ptr), h.base + r * h.w0 * 4, c0 * 4), "logits");
            if (V > c0) try m.d.check(m.d.api.cuMemcpyDtoH_v2(@ptrCast(out[r * V + c0 ..].ptr), h.base + (h.part + r * h.w1) * 4, (V - c0) * 4), "logits");
        }
    }
};

fn upload(d: *const cuda.Driver, at: u64, bytes: []const u8) !void {
    try d.check(d.api.cuMemcpyHtoD_v2(at, bytes.ptr, bytes.len), "upload");
}

test "ring" {
    try std.testing.expectEqual(@as(usize, 2176), ringSize(2047, 128));
    try std.testing.expectEqual(@as(usize, 2112), ringSize(2047, 1));
}
