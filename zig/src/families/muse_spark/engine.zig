//! Muse Spark's engine on one rank: the model and the steps rank 0 sends rank 1 over the plan link (MiMo-V2's shape).

const std = @import("std");
const cuda = @import("cuda");
const tp = @import("tp");
const load_ = @import("load.zig");
const fwd = @import("forward.zig");

/// The plan link's words: one GPU step each, executed by every rank in order.
pub const Op = enum(i64) { hello = 0x3A5E, forward = 1, stop = 3 };

/// What rank 0 announces first so every rank builds the same model.
pub const Setup = struct { context: usize, slots: usize };

/// The hello words: [hello, 2, context, slots].
pub fn hello(su: Setup) [4]i64 {
    return .{ @backingInt(Op.hello), 2, @intCast(su.context), @intCast(su.slots) };
}

/// The setup a follower reads from rank 0's hello.
pub fn setupOf(words: []const i64) !Setup {
    if (words.len < 4 or words[0] != @backingInt(Op.hello)) return error.NoHello;
    return .{ .context = @intCast(words[2]), .slots = @intCast(words[3]) };
}

/// The model, its kernels and the TP session: what rank 0 serves and rank 1 follows.
pub const Engine = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    w: load_.Weights,
    k3: cuda.exl3.Kernels,
    model: fwd.Model,
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
        e.logits = try gpa.alloc(f32, fwd.DECODE_ROWS * e.w.full.vocab);
        return e;
    }

    pub fn deinit(e: *Engine) void {
        const gpa = e.gpa;
        e.words.deinit(gpa);
        gpa.free(e.logits);
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
            .stop => return false,
            .hello => return error.LateHello,
        }
        return true;
    }
};

test "the hello carries the setup" {
    const su: Setup = .{ .context = 131072, .slots = 4 };
    try std.testing.expectEqual(su, try setupOf(&hello(su)));
    try std.testing.expectError(error.NoHello, setupOf(&.{ 3, 0 }));
}
