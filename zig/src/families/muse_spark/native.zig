//! Muse Spark behind the native server: the lane backend over engine.zig, on one GPU or two TP ranks (rank 1 runs
//! `tf-muse follow`). After MiMo-V2's native.zig, without its drafter and prompt path: prompts run as decode windows.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const tp = @import("tp");
const fwd = @import("forward.zig");
const engine = @import("engine.zig");
const be = lanes.backend;
const Engine = engine.Engine;

pub const model_type = "muse_spark";
pub const formats: []const []const u8 = &.{"exl3"};
pub const default_context: i64 = 131072;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = fwd.DECODE_ROWS;

/// The server's settings: `lanes` (--parallel) is the streams' cache slots.
pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false, lanes: u32 = 4, policy: []const u8 = "" };

pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool, resumed: bool) anyerror!bool;

pub const Loaded = struct {
    backend: be.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32 = 1,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?LoneRun = null,
    target: *anyopaque, // no prompt cache (no snap_* decls): never read
    learns: ?*anyopaque = null, // no --slide
    learn_bytes: usize = 0,
    policy: []const u8 = "",
};

const Lane = struct { slot: u32 };
const ring = 1024;

pub const Lanes = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    lanes: std.AutoHashMapUnmanaged(*const lanes.Stream, Lane) = .empty,
    free: u32, // slot bits
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    costs: [1]lanes.config.Cost = undefined,
    shared: [32]lanes.config.Cost = undefined,
    nshared: usize = 0,

    fn of(p: *anyopaque) *Lanes {
        return @ptrCast(@alignCast(p));
    }

    pub fn backend(l: *Lanes) be.Backend {
        return .{ .ptr = l, .vtable = &.{ .prefill = prefillFn, .first = firstFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .release = releaseFn } };
    }

    pub fn facts(l: *const Lanes) lanes.Model {
        return .{
            .exact_width = 8, // windows are row-invariant (tf-muse m1: 8 x 2 and 32 x 4 windows give one-row bits)
            .gpu_tokens = false,
            .mtp = false,
            .speculate = false,
            .speculate_early = false,
            .drafts = 0,
            .window_costs = l.costs[0..1],
            .hidden_rows = true,
            .batch_rows = fwd.DECODE_ROWS,
            .max_streams = @intCast(l.e.model.slots),
            .shared_costs = l.shared[0..l.nshared],
        };
    }

    /// Graphed forward costs at load on distinct tokens (one row; 2 .. 32 rows over the slots).
    pub fn measure(l: *Lanes, io: std.Io) !void {
        const e = l.e;
        var rows: [32]fwd.Row = undefined;
        var rng = std.Random.DefaultPrng.init(0x3A5E);
        const V = e.w.full.vocab;
        rows[0] = .{ .slot = 0, .pos = 64, .token = 0 };
        l.costs[0] = .{ .width = 1, .ms = try l.time(io, rows[0..1], 16, &rng, V) };
        l.nshared = 0;
        const S = e.model.slots;
        for (2..33) |n| {
            for (0..n) |j| rows[j] = .{ .slot = @intCast(j % S), .pos = @intCast(64 + j / S), .token = 0 };
            l.shared[l.nshared] = .{ .width = @intCast(n), .ms = try l.time(io, rows[0..n], 8, &rng, V) };
            l.nshared += 1;
        }
        std.log.info("muse costs (distinct tokens): one row {d:.2} ms, shared 8 / 16 / 32 rows {d:.2} / {d:.2} / {d:.2} ms", .{ l.costs[0].ms, l.shared[6].ms, l.shared[14].ms, l.shared[l.nshared - 1].ms });
    }

    /// Median ms of `reps` graphed forwards of `rows`, each on fresh pseudo-random tokens, after warm replays.
    fn time(l: *Lanes, io: std.Io, rows: []fwd.Row, reps: usize, rng: *std.Random.DefaultPrng, V: usize) !f64 {
        const fill = struct {
            fn f(rs: []fwd.Row, r: *std.Random.DefaultPrng, v: usize) void {
                for (rs) |*x| x.token = @intCast(1000 + r.random().uintLessThan(usize, v - 2000));
            }
        }.f;
        for (0..3) |_| {
            fill(rows, rng, V);
            try l.e.forward(rows, .ids);
        }
        try l.e.s.synchronize();
        var ms: [32]f64 = undefined;
        const n = @min(reps, ms.len);
        for (0..n) |i| {
            fill(rows, rng, V);
            const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
            try l.e.forward(rows, .ids);
            try l.e.s.synchronize();
            ms[i] = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) / 1e6;
        }
        std.mem.sort(f64, ms[0..n], {}, std.sort.asc(f64));
        return ms[n / 2];
    }

    fn take(l: *Lanes, token: u32) u64 {
        const h = l.next;
        l.drawn[h % ring] = token;
        l.next += 1;
        return h;
    }

    fn rowOf(l: *Lanes, r: usize) []const f32 {
        const V = l.e.w.full.vocab;
        return l.e.logits[r * V ..][0..V];
    }

    /// The prompt as decode windows of up to 128 rows; the last row's token drawn.
    fn prefillFn(p: *anyopaque, s: *lanes.Stream) anyerror!void {
        const l = of(p);
        const e = l.e;
        const ids = s.prompt();
        if (ids.len == 0 or ids.len + s.max_new + 1 > e.model.context) return error.PromptTooLong;
        const gop = try l.lanes.getOrPut(l.gpa, s);
        if (!gop.found_existing) {
            if (l.free == 0) {
                l.lanes.removeByPtr(gop.key_ptr);
                return error.NoFreeSlot;
            }
            const slot: u32 = @ctz(l.free);
            l.free &= ~(@as(u32, 1) << @intCast(slot));
            gop.value_ptr.* = .{ .slot = slot };
        }
        const slot = gop.value_ptr.slot;
        s.cached = 0;
        const greedy = s.sampling == null or s.sampling.?.temperature <= 0;
        var rows: [fwd.DECODE_ROWS]fwd.Row = undefined;
        var at: usize = 0;
        while (at < ids.len) {
            if (s.isCancelled()) return error.Cancelled;
            const n = @min(fwd.DECODE_ROWS, ids.len - at);
            for (0..n) |j| rows[j] = .{ .slot = slot, .pos = @intCast(at + j), .token = ids[at + j] };
            const last = at + n == ids.len;
            try e.forward(rows[0..n], if (!last) .none else if (greedy) .ids else .logits);
            if (last) _ = l.take(if (greedy) @intCast(e.ids[n - 1]) else try draw(l.gpa, l.rowOf(n - 1), s.sampling, ids.len));
            at += n;
        }
    }

    fn firstFn(p: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const l = of(p);
        if (position != s.prompt_len) return error.PositionMismatch;
        return l.next - 1;
    }

    fn queueFn(_: *anyopaque, _: *lanes.Stream, _: be.Feed, _: u64) anyerror!u64 {
        return error.NotPipelined;
    }

    fn readFn(p: *anyopaque, handle: u64) anyerror!u32 {
        const l = of(p);
        if (handle >= l.next or l.next - handle > ring) return error.NoSuchToken;
        return l.drawn[handle % ring];
    }

    /// Every window's rows in one forward; each row drawn with its stream's sampling at its keyed position.
    fn verifyFn(p: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const l = of(p);
        var rows: [fwd.DECODE_ROWS]fwd.Row = undefined;
        var n: usize = 0;
        for (windows) |w| {
            if (w.parents != null) return error.TreesNotBuilt;
            const lane = l.lanes.getPtr(w.stream) orelse return error.UnknownStream;
            if (n + w.rows() > rows.len) return error.WindowTooWide;
            for (0..w.rows()) |r| {
                const tok: u32 = if (r == 0) w.pending else w.tokens[r - 1];
                rows[n] = .{ .slot = lane.slot, .pos = @intCast(w.positions[r] - 1), .token = tok };
                n += 1;
            }
        }
        var all_greedy = true;
        for (windows) |w| if (w.stream.sampling) |sm| if (sm.temperature > 0) {
            all_greedy = false;
        };
        try l.e.forward(rows[0..n], if (all_greedy) .ids else .logits);
        var r0: usize = 0;
        for (windows, out) |w, *o| {
            for (0..w.rows()) |r| o.sampled[r] = if (all_greedy) @intCast(l.e.ids[r0 + r]) else try draw(l.gpa, l.rowOf(r0 + r), w.stream.sampling, w.positions[r]);
            @memcpy(o.drafts, w.tokens);
            r0 += w.rows();
        }
    }

    fn keepFn(_: *anyopaque, _: []const be.Window, paths: []const []const u32) anyerror!void {
        for (paths) |path| {
            if (path.len == 0) return error.EmptyPath;
            for (path, 0..) |r, i| if (r != i) return error.TreesNotBuilt;
        }
    }

    fn draftFn(_: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        for (requests) |r| if (r.depth > 0) return error.NoDraftHead;
    }

    fn releaseFn(p: *anyopaque, s: *lanes.Stream) void {
        const l = of(p);
        const kv = l.lanes.fetchRemove(s) orelse return;
        l.free |= @as(u32, 1) << @intCast(kv.value.slot);
    }
};

/// The token for a row: the first maximum when greedy (or temperature 0), else the keyed sampler over the top-k.
pub fn draw(gpa: std.mem.Allocator, row: []const f32, settings: ?lanes.Sampling, position: u64) !u32 {
    var best: usize = 0;
    for (row, 0..) |x, i| if (x > row[best]) {
        best = i;
    };
    if (!std.math.isFinite(row[best])) return error.NonfiniteLogits;
    const sampling = settings orelse return @intCast(best);
    if (sampling.temperature <= 0) return @intCast(best);
    const k = if (sampling.top_k == 0) row.len else @min(row.len, sampling.top_k);
    const values = try gpa.alloc(f64, k);
    defer gpa.free(values);
    const ids = try gpa.alloc(u64, k);
    defer gpa.free(ids);
    var filled: usize = 0;
    for (row, 0..) |x32, id| {
        const x: f64 = x32;
        if (k == row.len) {
            values[id] = x;
            ids[id] = id;
            continue;
        }
        if (filled == k and x <= values[k - 1]) continue;
        var at = @min(filled, k - 1);
        while (at > 0 and x > values[at - 1]) : (at -= 1) {
            values[at] = values[at - 1];
            ids[at] = ids[at - 1];
        }
        values[at] = x;
        ids[at] = id;
        filled = @min(filled + 1, k);
    }
    return @intCast(try lanes.sampling.choose(gpa, values, ids, position, sampling));
}

const Owned = struct { gpa: std.mem.Allocator, e: *Engine, l: Lanes, session: ?*tp.Session };

/// The native family entry: TF_TP_* for two ranks (rank 0 serves; `tf-muse follow` on rank 1); own kernels only.
pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    _ = kernels; // no captured Triton set: Muse runs its own CUDA kernels
    if (o.lanes == 0 or o.lanes > 32) return error.Slots;
    if (o.drafts) return error.NoDrafter;
    if (o.policy.len > 0) return error.Policy;
    const su: engine.Setup = .{ .context = o.context, .slots = o.lanes };
    var session: ?*tp.Session = null;
    if (std.c.getenv("TF_TP_WORLD") != null) {
        const c = try tp.Config.fromEnv();
        if (c.world > 1) {
            if (c.rank != 0) return error.FollowerServes; // rank 1 runs `tf-muse follow`
            const ss = try gpa.create(tp.Session);
            errdefer gpa.destroy(ss);
            try ss.open(ctx.d, c, @import("tp_kernel").mailbox);
            session = ss;
            try ss.plan.send(&engine.hello(su));
        }
    }
    errdefer if (session) |ss| {
        ss.announceStop();
        ss.close();
        gpa.destroy(ss);
    };
    const e = try Engine.init(gpa, io, ctx.d, dir, su, session);
    errdefer e.deinit();
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .e = e, .l = .{ .gpa = gpa, .e = e, .free = if (su.slots == 32) ~@as(u32, 0) else (@as(u32, 1) << @intCast(su.slots)) - 1 }, .session = session };
    try own.l.measure(io);
    return .{
        .backend = own.l.backend(),
        .facts = own.l.facts(),
        .rows = 1,
        .stream_bytes = 0,
        .free_streams = @intCast(su.slots),
        .ctx = own,
        .deinit = release,
        .target = own,
    };
}

/// The weights this rank loads: the checkpoint over TF_TP_WORLD ranks.
pub fn rankWeights(total: u64) u64 {
    const text = std.mem.span(std.c.getenv("TF_TP_WORLD") orelse return total);
    const world = std.fmt.parseInt(u64, text, 10) catch 1;
    return total / @max(world, 1);
}

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PromptTooLong => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.NoFreeSlot => "every stream slot is busy: retry once another request ends",
        else => null,
    };
}

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    if (own.session) |ss| {
        ss.plan.send(&.{ @backingInt(engine.Op.stop), 0 }) catch {};
    }
    own.l.lanes.deinit(own.gpa);
    own.e.deinit();
    if (own.session) |ss| {
        ss.announceStop();
        ss.close();
        own.gpa.destroy(ss);
    }
    own.gpa.destroy(own);
}
