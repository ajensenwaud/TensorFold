//! Muse Spark's engine on one rank: the model, its DFlash2 drafter (opt-in) and the steps rank 0 sends rank 1 over
//! the plan link (MiMo-V2's shape).

const std = @import("std");
const cuda = @import("cuda");
const tp = @import("tp");
const load_ = @import("load.zig");
const fwd = @import("forward.zig");
const dflash2 = @import("dflash2.zig");

/// The plan link's words: one GPU step each, executed by every rank in order.
pub const Op = enum(i64) { hello = 0x3A5E, forward = 1, draft = 2, stop = 3 };

/// What rank 0 announces first so every rank builds the same model.
pub const Setup = struct { context: usize, slots: usize, drafts: bool = false };

/// The hello words: [hello, 3, context, slots, drafts].
pub fn hello(su: Setup) [5]i64 {
    return .{ @backingInt(Op.hello), 3, @intCast(su.context), @intCast(su.slots), @intFromBool(su.drafts) };
}

/// The setup a follower reads from rank 0's hello.
pub fn setupOf(words: []const i64) !Setup {
    if (words.len < 4 or words[0] != @backingInt(Op.hello)) return error.NoHello;
    return .{ .context = @intCast(words[2]), .slots = @intCast(words[3]), .drafts = words.len >= 5 and words[4] != 0 };
}

/// The drafter pack: TF_MUSE_DRAFTER (the same path on every rank), else `<pack>/dflash2`.
pub fn drafterDir(gpa: std.mem.Allocator, pack: []const u8) ![]u8 {
    if (std.c.getenv("TF_MUSE_DRAFTER")) |v| return gpa.dupe(u8, std.mem.span(v));
    return std.fmt.allocPrint(gpa, "{s}/dflash2", .{pack});
}

/// TF_MUSE_DRAFT_TAPS=a,b,..: tap these target layers instead of the drafter's (plumbing checks on k-layer packs).
pub fn tapsOverride(buf: []usize) !?[]const usize {
    const v = std.mem.span(std.c.getenv("TF_MUSE_DRAFT_TAPS") orelse return null);
    var n: usize = 0;
    var it = std.mem.tokenizeScalar(u8, v, ',');
    while (it.next()) |t| : (n += 1) {
        if (n == buf.len) return error.TooManyTaps;
        buf[n] = try std.fmt.parseInt(usize, t, 10);
    }
    return buf[0..n];
}

/// The model, its kernels and the TP session: what rank 0 serves and rank 1 follows.
pub const Engine = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    w: load_.Weights,
    k3: cuda.exl3.Kernels,
    model: fwd.Model,
    dr: ?*dflash2.Drafter = null,
    session: ?*tp.Session = null,
    s: cuda.Stream,
    logits: []f32, // [DECODE_ROWS, V] host
    ids: [fwd.DECODE_ROWS]i32 = undefined, // device argmax of each row (Want.ids)
    words: std.ArrayList(i64) = .empty,

    /// Load everything on this rank. `session`: the open TP session (world > 1), null on one GPU.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, su: Setup, session: ?*tp.Session) !*Engine {
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.* = .{ .gpa = gpa, .d = d, .w = undefined, .k3 = undefined, .model = undefined, .s = undefined, .logits = &.{}, .session = session };
        const rank: usize = if (session) |ss| ss.cfg.rank else 0;
        const world: usize = if (session) |ss| ss.cfg.world else 1;
        e.s = try cuda.Stream.init(d, false);
        errdefer e.s.deinit();
        e.w = try load_.load(gpa, io, d, dir, rank, world, .{ .log = rank == 0 });
        errdefer e.w.deinit();
        e.k3 = try cuda.exl3.Kernels.load(d);
        errdefer e.k3.deinit();
        const comm = if (session) |ss| ss.comm() else null;
        e.model = try fwd.Model.init(gpa, d, &e.w, &e.k3, comm, su.context, fwd.DECODE_ROWS, su.slots, e.s);
        errdefer e.model.deinit();
        if (su.drafts) {
            const ddir = try drafterDir(gpa, dir);
            defer gpa.free(ddir);
            var tb: [8]usize = undefined;
            e.dr = try dflash2.Drafter.init(gpa, io, &e.model, ddir, e.s, true, try tapsOverride(&tb));
            e.model.dr = e.dr;
            if (rank == 0) std.log.info("muse drafter: {s}, block {d}, {d:.2} GiB on this rank (split over {d})", .{ ddir, e.dr.?.block, @as(f64, @floatFromInt(e.dr.?.bytes())) / (1 << 30), world });
        }
        errdefer if (e.dr) |dr| dr.deinit();
        e.logits = try gpa.alloc(f32, fwd.DECODE_ROWS * e.w.full.vocab);
        return e;
    }

    pub fn deinit(e: *Engine) void {
        const gpa = e.gpa;
        e.words.deinit(gpa);
        gpa.free(e.logits);
        if (e.dr) |dr| dr.deinit();
        e.model.deinit();
        e.k3.deinit();
        e.w.deinit();
        e.s.deinit();
        gpa.destroy(e);
    }

    fn send(e: *Engine, words: []const i64) !void {
        if (e.session) |ss| try ss.plan.send(words);
    }

    pub const Want = enum { none, ids, logits };

    /// A graphed forward window on every rank; on rank 0 `ids` leaves each row's argmax in `ids`, `logits` [n, V].
    pub fn forward(e: *Engine, rows: []const fwd.Row, want: Want) !void {
        e.words.clearRetainingCapacity();
        try e.words.appendSlice(e.gpa, &.{ @backingInt(Op.forward), @intCast(rows.len) });
        for (rows) |r| try e.words.appendSlice(e.gpa, &.{ r.slot, r.pos, r.token });
        try e.send(e.words.items);
        try e.model.graphed(e.s, rows, if (want == .logits) e.logits[0 .. rows.len * e.w.full.vocab] else null);
        if (want == .ids) try e.model.argmaxTo(e.s, rows.len, e.ids[0..rows.len]);
    }

    /// A draft pass on every rank: drafts [S, block - 1] into `out` (each stream's anchor token, position, slot).
    pub fn draft(e: *Engine, tokens: []const i64, pos0: []const u32, slots: []const u32, out: []i32) !void {
        e.words.clearRetainingCapacity();
        try e.words.appendSlice(e.gpa, &.{ @backingInt(Op.draft), @intCast(tokens.len) });
        for (tokens, pos0, slots) |t, p, s| try e.words.appendSlice(e.gpa, &.{ t, p, s });
        try e.send(e.words.items);
        try e.dr.?.draft(e.s, tokens, pos0, slots, out);
    }

    /// One step from the plan link, on a follower; false at stop.
    pub fn follow(e: *Engine, words: []const i64) !bool {
        if (words.len < 2) return error.ShortPlan;
        const op: Op = @fromBackingInt(@intCast(words[0]));
        const n: usize = @intCast(words[1]);
        switch (op) {
            .forward => {
                if (words.len != 2 + 3 * n or n > fwd.DECODE_ROWS) return error.BadPlan;
                var rows: [fwd.DECODE_ROWS]fwd.Row = undefined;
                for (0..n) |i| rows[i] = .{ .slot = @intCast(words[2 + 3 * i]), .pos = @intCast(words[3 + 3 * i]), .token = words[4 + 3 * i] };
                try e.model.graphed(e.s, rows[0..n], null);
            },
            .draft => {
                if (words.len != 2 + 3 * n or n > dflash2.MAX_STREAMS) return error.BadPlan;
                const dr = e.dr orelse return error.NoDrafter;
                var t: [dflash2.MAX_STREAMS]i64 = undefined;
                var p: [dflash2.MAX_STREAMS]u32 = undefined;
                var sl: [dflash2.MAX_STREAMS]u32 = undefined;
                var out: [dflash2.MAX_STREAMS * 16]i32 = undefined;
                for (0..n) |i| {
                    t[i] = words[2 + 3 * i];
                    p[i] = @intCast(words[3 + 3 * i]);
                    sl[i] = @intCast(words[4 + 3 * i]);
                }
                try dr.draft(e.s, t[0..n], p[0..n], sl[0..n], &out);
            },
            .stop => return false,
            .hello => return error.LateHello,
        }
        return true;
    }
};

test "the hello carries the setup" {
    const su: Setup = .{ .context = 131072, .slots = 4, .drafts = true };
    try std.testing.expectEqual(su, try setupOf(&hello(su)));
    const plain: Setup = .{ .context = 65536, .slots = 2 };
    try std.testing.expectEqual(plain, try setupOf(&hello(plain)));
    try std.testing.expectEqual(plain, try setupOf(&.{ @backingInt(Op.hello), 2, 65536, 2 })); // an older rank 0
    try std.testing.expectError(error.NoHello, setupOf(&.{ 3, 0 }));
}
