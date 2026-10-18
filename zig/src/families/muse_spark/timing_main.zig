//! tf-muse timing MODEL _ _: graphed decode windows and prompt chunks per lever arm, one load (TP=2 as m1: both ranks
//! run the same loop, rank 0 prints). Arms (MUSE_TIMING_ARMS: TF_MUSE_FAST masks, o = router on a side stream)
//! switched at run time, graphs dropped between them. Per arm: windows of MUSE_TIMING_WIDTHS rows (default 1,2,4,8,16,32; distinct pseudo-random tokens
//! over 4 slots) at MUSE_TIMING_POS (64) and MUSE_TIMING_POS2 (8192): median ms of 16 after 3 warm replays; then
//! prompt chunks (prompt path) of 512 / 1024 rows and a 4096-row prompt from position 0 (eager): ms and tok/s.

const std = @import("std");
const cuda = @import("cuda");
const tp = @import("tp");
const muse = @import("muse_spark");
const Model = muse.forward.Model;
const Row = muse.forward.Row;

fn env(name: [*:0]const u8, default: usize) usize {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch default;
}

fn now(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

fn ms(io: std.Io, t0: i96) f64 {
    return @as(f64, @floatFromInt(now(io) - t0)) / 1e6;
}

pub fn run(gpa: std.mem.Allocator, io: std.Io, driver: *const cuda.Driver, args: []const [:0]const u8, rank: usize, world: usize, comm: ?tp.Collective) !void {
    var s = try cuda.Stream.init(driver, false);
    defer s.deinit();
    const context = env("MUSE_CONTEXT", 16384);
    const S: usize = 4;
    var w = try muse.load.load(gpa, io, driver, args[2], rank, world, .{ .log = rank == 0, .embed_host = muse.load.embedHostEnv() });
    defer w.deinit();
    var k3 = try cuda.exl3.Kernels.load(driver);
    defer k3.deinit();
    var model = try Model.init(gpa, driver, &w, &k3, comm, context, muse.forward.DECODE_ROWS, S, s);
    defer model.deinit();
    var side = try cuda.Stream.init(driver, false);
    defer side.deinit();
    var e1 = try cuda.Event.init(driver, false);
    defer e1.deinit();
    var e2 = try cuda.Event.init(driver, false);
    defer e2.deinit();
    const pr = try muse.prompt.Prompt.init(&model, gpa, env("TF_MUSE_PROMPT_ROWS", muse.prompt.CAP), env("TF_MUSE_PROMPT_EWIN", muse.prompt.EWIN), env("TF_MUSE_PROMPT_EXACT", 1) != 0, s);
    defer pr.deinit();
    const V = w.full.vocab;
    var rng = std.Random.DefaultPrng.init(0x3A5E);
    const widths_text = if (std.c.getenv("MUSE_TIMING_WIDTHS")) |v| std.mem.span(v) else "1,2,4,8,16,32";
    const positions = [_]usize{ env("MUSE_TIMING_POS", 64), env("MUSE_TIMING_POS2", 8192) };
    // arms: MUSE_TIMING_ARMS = "mask[o],..." (TF_MUSE_FAST bits, o = router side stream), default base, every lever,
    // every lever + overlap, then each lever alone
    const arms_text = if (std.c.getenv("MUSE_TIMING_ARMS")) |v| std.mem.span(v) else "0,31,31o,1,2,4,8";
    const Arm = struct { name: []const u8, fast: u32, overlap: bool };
    var arms_buf: [16]Arm = undefined;
    var na: usize = 0;
    var ait = std.mem.tokenizeScalar(u8, arms_text, ',');
    while (ait.next()) |t| if (na < arms_buf.len) {
        const o = std.mem.endsWith(u8, t, "o");
        arms_buf[na] = .{ .name = t, .fast = try std.fmt.parseInt(u32, if (o) t[0 .. t.len - 1] else t, 10), .overlap = o };
        na += 1;
    };
    const arms = arms_buf[0..na];
    var rows: [32]Row = undefined;
    const toks = try gpa.alloc(i64, 4096);
    defer gpa.free(toks);
    for (toks) |*t| t.* = @intCast(1000 + rng.random().uintLessThan(usize, V - 2000));
    for (arms) |arm| {
        model.fast = arm.fast;
        model.side = if (arm.overlap) side else null;
        model.ev_fork = if (arm.overlap) e1 else null;
        model.ev_join = if (arm.overlap) e2 else null;
        model.dropGraphs();
        for (positions) |P| {
            if (P + 64 > context or env("MUSE_TIMING_NO_DECODE", 0) != 0) continue;
            var it = std.mem.tokenizeScalar(u8, widths_text, ',');
            while (it.next()) |wt| {
                const n = try std.fmt.parseInt(usize, wt, 10);
                if (n == 0 or n > rows.len) continue;
                var t: [16]f64 = undefined;
                for (0..3 + t.len) |rep| {
                    for (0..n) |j| rows[j] = .{ .slot = @intCast(j % S), .pos = @intCast(P + j / S), .token = @intCast(1000 + rng.random().uintLessThan(usize, V - 2000)) };
                    const t0 = now(io);
                    try model.graphed(s, rows[0..n], null);
                    if (rep >= 3) t[rep - 3] = ms(io, t0);
                }
                std.mem.sort(f64, &t, {}, std.sort.asc(f64));
                if (rank == 0) std.debug.print("TIMING {s} graphed rows {d} pos {d}: {d:.2} ms (min {d:.2})\n", .{ arm.name, n, P, (t[7] + t[8]) / 2, t[0] });
            }
        }
        const ewin0 = pr.ewin;
        const ew_text = if (std.c.getenv("MUSE_TIMING_EWINS")) |v| std.mem.span(v) else "";
        var ews: [8]usize = undefined;
        var new_: usize = 0;
        var eit = std.mem.tokenizeScalar(u8, ew_text, ',');
        while (eit.next()) |e| if (new_ < ews.len) {
            const v = try std.fmt.parseInt(usize, e, 10);
            if (v <= ewin0) {
                ews[new_] = v;
                new_ += 1;
            }
        };
        if (new_ == 0) {
            ews[0] = ewin0;
            new_ = 1;
        }
        if (env("MUSE_TIMING_NO_PROMPT", 0) != 0) new_ = 0;
        for (ews[0..new_]) |ew| {
            pr.ewin = ew;
            var cn: usize = 512;
            while (cn <= pr.cap) : (cn *= 2) {
                var best: f64 = 1e30;
                for (0..2) |_| {
                    const t0 = now(io);
                    try pr.chunk(s, 0, 0, toks[0..cn]);
                    try s.synchronize();
                    best = @min(best, ms(io, t0));
                }
                if (rank == 0) std.debug.print("TIMING {s} prompt chunk {d} rows at 0 (expert window {d}): {d:.0} ms ({d:.0} tok/s)\n", .{ arm.name, cn, ew, best, @as(f64, @floatFromInt(cn)) / best * 1000 });
            }
        }
        pr.ewin = ewin0;
        if (env("MUSE_TIMING_PROFILE", 0) != 0 and env("MUSE_TIMING_NO_PROMPT", 0) == 0) {
            var prof: [9]f64 = @splat(0);
            pr.prof = &prof;
            pr.io = io;
            try pr.chunk(s, 0, 0, toks[0..pr.cap]);
            pr.prof = null;
            var tot: f64 = 0;
            for (prof) |x| tot += x;
            if (rank == 0) for (muse.prompt.Prompt.phases, prof) |name, x| std.debug.print("PROFILE {s} prompt chunk {d}: {s} {d:.0} ms ({d:.1}%)\n", .{ arm.name, pr.cap, name, x, 100 * x / tot });
        }
        if (context >= 4096 + 64 and env("MUSE_TIMING_NO_PROMPT", 0) == 0) {
            const t0 = now(io);
            var at: usize = 0;
            while (at < 4096) : (at += pr.cap) try pr.chunk(s, 1, at, toks[at..@min(4096, at + pr.cap)]);
            try s.synchronize();
            if (rank == 0) std.debug.print("TIMING {s} prompt 4096 tokens in chunks of {d}\n", .{ arm.name, pr.cap });
            try s.synchronize();
            const t = ms(io, t0);
            if (rank == 0) std.debug.print("TIMING {s} prompt 4096 tokens: {d:.0} ms ({d:.0} tok/s)\n", .{ arm.name, t, 4096.0 / t * 1000 });
        }
    }
    model.side = null;
    model.ev_fork = null;
    model.ev_join = null;
    if (rank == 0) std.debug.print("TIMING done\n", .{});
}
