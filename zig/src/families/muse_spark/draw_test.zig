//! native.draw (the unsorted full-vocabulary race) against the draw it replaced, token for token.
const std = @import("std");
const lanes = @import("lanes");
const native = @import("native.zig");

/// native.draw before the change: the same top-k preselection, then the sorted reference.
fn oldDraw(gpa: std.mem.Allocator, row: []const f32, settings: ?lanes.Sampling, position: u64) !u32 {
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
    return @intCast(try lanes.sampling.chooseSorted(gpa, values, ids, position, sampling));
}

test "draw over Muse's vocabulary gives the old draw's token at every top_k, top_p and temperature" {
    const a = std.testing.allocator;
    const vocab = 202048;
    const row = try a.alloc(f32, vocab);
    defer a.free(row);
    var prng = std.Random.DefaultPrng.init(816);
    const r = prng.random();
    var draws: usize = 0;
    for (0..2) |trial| {
        // a peaked row (one token far ahead) and a flat one (bf16-like ties)
        for (row) |*x| x.* = @floatCast(if (trial == 0) r.floatNorm(f64) * 2.5 else @round(r.floatNorm(f64) * 8.0) / 8.0);
        row[r.uintLessThan(usize, vocab)] = if (trial == 0) 18.0 else 3.0;
        for ([_]u32{ 0, 1, 64, 1024, vocab }) |k| for ([_]f64{ 1.0, 0.95, 0.5 }) |p| for ([_]f64{ 0.6, 1.0, 1.5 }) |t| {
            for (0..2) |pos| {
                const s = lanes.Sampling{ .seed = r.int(u64) >> 1, .temperature = t, .top_k = k, .top_p = p };
                const position = 1000 + pos * 7919;
                try std.testing.expectEqual(try oldDraw(a, row, s, position), try native.draw(a, row, s, position));
                draws += 1;
            }
        };
    }
    try std.testing.expectEqual(@as(usize, 180), draws);
}
