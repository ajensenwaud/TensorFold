//! choose (no sort when nothing is cut by rank) against chooseSorted, the sorted reference, token for token.
const std = @import("std");
const sampling = @import("sampling.zig");

extern "c" fn log(x: f64) f64;

const Shape = enum { normal, ties, wide, flat, inf };

fn fill(r: std.Random, values: []f64, ids: []u64, shape: Shape, permute: bool) void {
    for (values, ids, 0..) |*v, *id, i| {
        const x = r.floatNorm(f64);
        v.* = switch (shape) {
            .normal => @as(f32, @floatCast(x * 3.0)),
            .ties => @round(x * 4.0) / 4.0,
            .wide => @as(f32, @floatCast(x * 25.0)),
            .flat => 0.5,
            .inf => if (i % 3 == 0) -std.math.inf(f64) else @as(f32, @floatCast(x)),
        };
        id.* = i;
    }
    if (shape == .ties and values.len > 2) values[1] = -0.0;
    if (permute) r.shuffle(u64, ids);
}

test "choose without the sort draws the sorted reference's token" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(20261024);
    const r = prng.random();
    const temps = [_]f64{ 0.6, 1.0, 1.5, 1e-7 };
    const min_ps = [_]f64{ 0.0, 0.05, 1.0, 1.5 };
    const top_ps = [_]f64{ 1.0, 0.0, 1.5, 0.95 };
    var draws: usize = 0;
    var differ: usize = 0;
    for ([_]usize{ 1, 2, 7, 64, 1000, 5000 }) |width| {
        const values = try a.alloc(f64, width);
        defer a.free(values);
        const ids = try a.alloc(u64, width);
        defer a.free(ids);
        for (std.enums.values(Shape)) |shape| for ([_]bool{ false, true }) |permute| {
            fill(r, values, ids, shape, permute);
            for (temps) |t| for (min_ps) |mp| for (top_ps) |tp| for ([_]u32{ 0, @intCast(width) }) |k| {
                for (0..4) |pos| {
                    const s = sampling.Sampling{ .seed = r.int(u64) >> 1, .temperature = t, .top_k = k, .top_p = tp, .min_p = mp };
                    const want = try sampling.chooseSorted(a, values, ids, pos, s);
                    const got = try sampling.choose(a, values, ids, pos, s);
                    draws += 1;
                    if (got != want) differ += 1;
                }
            };
        };
    }
    try std.testing.expectEqual(@as(usize, 0), differ);
    try std.testing.expect(draws > 10000);
}

test "chooseRow over f32 rows draws the sorted reference's token" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    const r = prng.random();
    for ([_]usize{ 1, 3, 100, 3000 }) |width| {
        const row = try a.alloc(f32, width);
        defer a.free(row);
        const values = try a.alloc(f64, width);
        defer a.free(values);
        const ids = try a.alloc(u64, width);
        defer a.free(ids);
        for (std.enums.values(Shape)) |shape| {
            fill(r, values, ids, shape, false);
            for (row, values) |*x, v| x.* = @floatCast(v);
            if (shape == .inf and width > 2) row[width - 1] = std.math.nan(f32);
            for (row, values) |x, *v| v.* = x;
            for ([_]f64{ 0.6, 1.0, 1.5 }) |t| for ([_]f64{ 0.0, 0.05 }) |mp| for ([_]u32{ 0, @intCast(width), 1 }) |k| for (0..3) |pos| {
                const s = sampling.Sampling{ .seed = r.int(u64) >> 1, .temperature = t, .top_k = k, .top_p = 1.0, .min_p = mp };
                try std.testing.expectEqual(try sampling.chooseSorted(a, values, ids, pos, s), try sampling.chooseRow(a, row, pos, s));
            };
        }
    }
}

test "a NaN value falls back to the sorted reference" {
    const a = std.testing.allocator;
    const values = [_]f64{ 1.0, std.math.nan(f64), 3.0, 2.0 };
    const ids = [_]u64{ 4, 5, 6, 7 };
    for (0..64) |pos| {
        const s = sampling.Sampling{ .seed = 99, .temperature = 1.0, .top_k = 0, .top_p = 1.0 };
        try std.testing.expectEqual(try sampling.chooseSorted(a, &values, &ids, pos, s), try sampling.choose(a, &values, &ids, pos, s));
    }
}

test "every skipped uniform's Gumbel noise stays under its gap" {
    // the largest uniform a bound skips is the double below it; its noise must sit more than 0.4 under the gap
    for (sampling.gaps, sampling.skipBounds()) |gap, b| {
        try std.testing.expect(b > 0.0 and b <= 1.0);
        try std.testing.expect(-log(-log(std.math.nextAfter(f64, b, 0.0))) < gap - 0.4);
    }
}
