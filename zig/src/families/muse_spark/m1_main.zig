//! tf-muse: Muse Spark's decode forward on one GPU against exllamav3's answer key (tools/g5 in the recipe).
//!   tf-muse m1 MODEL IN_DIR OUT_DIR
//! IN_DIR: names.txt (a text a line), <name>.ids (int32), and with MUSE_TEACHER=1 <name>.states.f32 (fp32 [layers + 1,
//! T, D]: the embedding norm's output, then each block's output, from the reference). Every prompt token is fed as a
//! decode row (teacher-forced), then MUSE_GREEDY tokens greedily. OUT_DIR: <name>.logits.f32 [T, vocab],
//! <name>.greedy.i32, and with MUSE_CAPTURE=1 <name>.res.f32 / <name>.attn.f32 (fp32 [layers, T, D]).
//! Knobs: MUSE_ROWS (prompt rows a text a window, 1), MUSE_SLOTS (texts sharing a window, 1), MUSE_LAYERS (first
//! layers only), MUSE_CONTEXT (4096), MUSE_GREEDY (32), MUSE_CAPTURE, MUSE_TEACHER, MUSE_GRAPHS.
//! TP=2: TF_TP_WORLD=2 / TF_TP_RANK / TF_TP_MASTER / TF_TP_PORT on both nodes (same IN_DIR files on each); both
//! ranks run the same windows, rank 0 writes OUT_DIR.

const std = @import("std");
const cuda = @import("cuda");
const tp = @import("tp");
const muse = @import("muse_spark");
const Model = muse.forward.Model;

fn env(name: [*:0]const u8, default: usize) usize {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch default;
}

const Text = struct {
    name: []const u8,
    ids: []i32,
    states: ?[]f32 = null, // teacher: [L + 1, T, D]
    logits: []f32, // [T, V]
    res: ?[]f32 = null, // [L, T, D]
    attn: ?[]f32 = null,
    greedy: std.ArrayList(i32) = .empty,
    fed: usize = 0,
    last: usize = 0, // row of its last logits in the window buffer
};

const Win = struct { text: usize, pos: usize };

const Ctx = struct {
    d: *const cuda.Driver,
    s: cuda.Stream,
    texts: []Text,
    win: []Win,
    layers: usize,
    D: usize,
    host: []f32, // [R, D]
};

fn hook(ctx: *anyopaque, what: Model.Tap, layer: usize, dev: u64, n: usize) anyerror!void {
    const c: *Ctx = @ptrCast(@alignCast(ctx));
    const D = c.D;
    switch (what) {
        .input => {
            for (c.win[0..n], 0..) |w, r| {
                const t = c.texts[w.text];
                const st = t.states orelse return error.NoTeacher;
                const T = t.ids.len;
                if (w.pos >= T) { // greedy rows: keep ours
                    try c.s.synchronize();
                    try c.d.check(c.d.api.cuMemcpyDtoH_v2(@ptrCast(c.host[r * D ..].ptr), dev + r * D * 4, D * 4), "teacher keep");
                    continue;
                }
                @memcpy(c.host[r * D ..][0..D], st[(layer * T + w.pos) * D ..][0..D]);
            }
            try c.s.synchronize();
            try c.d.check(c.d.api.cuMemcpyHtoD_v2(dev, @ptrCast(c.host.ptr), n * D * 4), "teacher");
        },
        .attn, .mlp => {
            try c.s.synchronize();
            try c.d.check(c.d.api.cuMemcpyDtoH_v2(@ptrCast(c.host.ptr), dev, n * D * 4), "capture");
            for (c.win[0..n], 0..) |w, r| {
                const t = c.texts[w.text];
                const T = t.ids.len;
                if (w.pos >= T) continue;
                const dst = (if (what == .mlp) t.res else t.attn) orelse continue;
                @memcpy(dst[(layer * T + w.pos) * D ..][0..D], c.host[r * D ..][0..D]);
            }
        },
    }
}

fn readAll(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 34));
}

fn writeAll(io: std.Io, path: []const u8, bytes: []const u8) !void {
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes });
}

fn argmax(xs: []const f32) i32 {
    var best: usize = 0;
    for (xs, 0..) |x, i| if (x > xs[best]) {
        best = i;
    };
    return @intCast(best);
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 5 or !std.mem.eql(u8, args[1], "m1")) {
        std.debug.print("usage: tf-muse m1 MODEL IN_DIR OUT_DIR   (knobs: MUSE_ROWS, MUSE_SLOTS, MUSE_LAYERS, MUSE_CONTEXT, MUSE_GREEDY, MUSE_CAPTURE, MUSE_TEACHER)\n", .{});
        return 2;
    }
    const gpa = init.gpa;
    const ar = init.arena.allocator();
    const io = init.io;
    const rows_per = env("MUSE_ROWS", 1);
    const slots = env("MUSE_SLOTS", 1);
    const context = env("MUSE_CONTEXT", 4096);
    const n_greedy = env("MUSE_GREEDY", 32);
    const capture = env("MUSE_CAPTURE", 0) != 0;
    const teacher = env("MUSE_TEACHER", 0) != 0;
    const graphs = env("MUSE_GRAPHS", 0) != 0; // graphed windows (no captures): logits must equal the eager run's bits
    if (graphs and (capture or teacher)) return error.GraphsTakeNoCaptures;
    const max_layers: ?usize = if (env("MUSE_LAYERS", 0) > 0) env("MUSE_LAYERS", 0) else null;
    if (rows_per * slots > muse.forward.DECODE_ROWS) return error.WindowTooWide;

    const tpc: ?tp.Config = if (std.c.getenv("TF_TP_WORLD") != null) try tp.Config.fromEnv() else null;
    const world: usize = if (tpc) |c| c.world else 1;
    const rank: usize = if (tpc) |c| c.rank else 0;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, if (tpc) |c| @intCast(c.device) else 0);
    defer ctx.deinit();
    var session: tp.Session = undefined;
    if (world > 1) try session.open(&driver, tpc.?, @import("tp_kernel").mailbox);
    defer if (world > 1) {
        session.announceStop(); // a clean stop: the peer exits 0 instead of failing fast when this rank ends first
        session.close();
    };
    var s = try cuda.Stream.init(&driver, false);
    defer s.deinit();
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    var w = try muse.load.load(gpa, io, &driver, args[2], rank, world, .{ .max_layers = max_layers, .log = rank == 0 });
    defer w.deinit();
    var k3 = try cuda.exl3.Kernels.load(&driver);
    defer k3.deinit();
    var model = try Model.init(gpa, &driver, &w, &k3, if (world > 1) session.comm() else null, context, rows_per * slots, slots, s);
    defer model.deinit();
    const L = w.layers.len;
    const D = w.cfg.hidden;
    const V = w.full.vocab;
    std.debug.print("[m1] rank {d}/{d}: loaded {d} layers in {d:.0}s ({d:.1} GiB weights, {d:.1} GiB model buffers); rows {d} x slots {d}, context {d}\n", .{ rank, world, L, @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) / 1e9, @as(f64, @floatFromInt(w.arena.total)) / (1 << 30), @as(f64, @floatFromInt(model.arena.total)) / (1 << 30), rows_per, slots, context });

    // texts
    const names_path = try std.fs.path.join(ar, &.{ args[3], "names.txt" });
    const names_text = try readAll(ar, io, names_path);
    var texts: std.ArrayList(Text) = .empty;
    var it = std.mem.tokenizeAny(u8, names_text, " \r\n\t");
    while (it.next()) |name| {
        const raw = try readAll(ar, io, try std.fmt.allocPrint(ar, "{s}/{s}.ids", .{ args[3], name }));
        const ids = try ar.alloc(i32, raw.len / 4);
        @memcpy(std.mem.sliceAsBytes(ids), raw[0 .. ids.len * 4]);
        if (ids.len + n_greedy > context) return error.TextTooLong;
        var t: Text = .{ .name = name, .ids = ids, .logits = try gpa.alloc(f32, ids.len * V) };
        if (teacher) {
            const st = try readAll(gpa, io, try std.fmt.allocPrint(ar, "{s}/{s}.states.f32", .{ args[3], name }));
            if (st.len < L * ids.len * D * 4) return error.TeacherStatesShort;
            t.states = @alignCast(std.mem.bytesAsSlice(f32, st));
        }
        if (capture) {
            t.res = try gpa.alloc(f32, L * ids.len * D);
            t.attn = try gpa.alloc(f32, L * ids.len * D);
        }
        try texts.append(ar, t);
    }
    const R = rows_per * slots;
    const win = try ar.alloc(Win, R);
    const host = try ar.alloc(f32, R * D);
    const out = try gpa.alloc(f32, R * V);
    defer gpa.free(out);
    var cc: Ctx = .{ .d = &driver, .s = s, .texts = texts.items, .win = win, .layers = L, .D = D, .host = host };
    const cap: ?Model.Capture = if (capture or teacher) .{ .hook = &hook, .ctx = &cc, .teacher = teacher } else null;

    const t1 = std.Io.Clock.awake.now(io).toNanoseconds();
    var windows: usize = 0;
    var total_rows: usize = 0;
    var b: usize = 0;
    while (b < texts.items.len) : (b += slots) {
        const batch = texts.items[b..@min(b + slots, texts.items.len)];
        while (true) {
            var rows: std.ArrayList(muse.forward.Row) = .empty;
            defer rows.deinit(gpa);
            for (batch, 0..) |*t, j| {
                const T = t.ids.len;
                if (t.fed < T) {
                    const k = @min(rows_per, T - t.fed);
                    for (0..k) |q| {
                        win[rows.items.len] = .{ .text = b + j, .pos = t.fed + q };
                        try rows.append(gpa, .{ .slot = @intCast(j), .pos = @intCast(t.fed + q), .token = t.ids[t.fed + q] });
                    }
                    t.fed += k;
                    t.last = rows.items.len - 1;
                } else if (t.greedy.items.len < n_greedy) {
                    const nxt = argmax(out[t.last * V ..][0..V]);
                    try t.greedy.append(gpa, nxt);
                    if (t.greedy.items.len == n_greedy or w.full.isEos(@intCast(nxt))) {
                        if (w.full.isEos(@intCast(nxt))) while (t.greedy.items.len < n_greedy) try t.greedy.append(gpa, -1);
                        continue;
                    }
                    win[rows.items.len] = .{ .text = b + j, .pos = t.fed };
                    try rows.append(gpa, .{ .slot = @intCast(j), .pos = @intCast(t.fed), .token = nxt });
                    t.fed += 1;
                    t.last = rows.items.len - 1;
                }
            }
            if (rows.items.len == 0) break;
            if (graphs) try model.graphed(s, rows.items, out[0 .. rows.items.len * V]) else try model.forward(s, rows.items, out[0 .. rows.items.len * V], cap);
            windows += 1;
            total_rows += rows.items.len;
            for (rows.items, win[0..rows.items.len], 0..) |r, wv, i| {
                const t = &texts.items[wv.text];
                if (r.pos < t.ids.len) @memcpy(t.logits[r.pos * V ..][0..V], out[i * V ..][0..V]);
            }
        }
    }
    const secs = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t1)) / 1e9;
    std.debug.print("[m1] {d} windows, {d} rows in {d:.1}s ({s}; not a speed number); graphs captured {d}, replayed {d}\n", .{ windows, total_rows, secs, if (graphs) "graphed" else "eager", model.captures, model.replays });
    for (texts.items) |*t| if (rank == 0) {
        try writeAll(io, try std.fmt.allocPrint(ar, "{s}/{s}.logits.f32", .{ args[4], t.name }), std.mem.sliceAsBytes(t.logits));
        try writeAll(io, try std.fmt.allocPrint(ar, "{s}/{s}.greedy.i32", .{ args[4], t.name }), std.mem.sliceAsBytes(t.greedy.items));
        if (t.res) |x| try writeAll(io, try std.fmt.allocPrint(ar, "{s}/{s}.res.f32", .{ args[4], t.name }), std.mem.sliceAsBytes(x));
        if (t.attn) |x| try writeAll(io, try std.fmt.allocPrint(ar, "{s}/{s}.attn.f32", .{ args[4], t.name }), std.mem.sliceAsBytes(x));
        std.debug.print("[m1] {s}: {d} tokens, greedy {any}\n", .{ t.name, t.ids.len, t.greedy.items[0..@min(8, t.greedy.items.len)] });
    };
    for (texts.items) |*t| {
        gpa.free(t.logits);
        if (t.res) |x| gpa.free(x);
        if (t.attn) |x| gpa.free(x);
        if (t.states) |x| gpa.free(std.mem.sliceAsBytes(x));
        t.greedy.deinit(gpa);
    }
    std.debug.print("DONE m1 rank {d}\n", .{rank});
    return 0;
}
