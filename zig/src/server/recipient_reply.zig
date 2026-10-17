//! Replies written as messages to recipients (Muse Spark): `<|start|>assistant to=R<|message|>body<|eom|>`.
//! The prompt ends at `<|start|>assistant`, so the first header is ` to=R`. Recipient "self" is reasoning; "user",
//! "commentary" (and none) are the answer; any other recipient is a tool call, its `<atem:function_calls>` block
//! rewritten as Qwen3-Coder's `<tool_call><function=..>` markup, which tool_parse reads against the tools' schemas.
const std = @import("std");
const reply_text = @import("reply_text.zig");
const Allocator = std.mem.Allocator;
const Split = reply_text.Split;
const findTag = reply_text.findTag;

const start = "<|start|>";
const message = "<|message|>";
const ends = [_][]const u8{ "<|eom|>", "<|eot|>" };

/// Whether the tokenizer and template write replies this way (every marker a single token, the template's "to=self").
pub fn detect(text: anytype) bool {
    for ([_][]const u8{ start, message, ends[0], ends[1] }) |m| if (text.tokenId(m) == null) return false;
    return std.mem.indexOf(u8, text.templateSource(), "to=self") != null;
}

const Kind = enum { reasoning, answer, call };

fn kindOf(header: []const u8) struct { kind: Kind, recipient: []const u8 } {
    const at = std.mem.indexOf(u8, header, "to=") orelse return .{ .kind = .answer, .recipient = "" };
    const rest = header[at + 3 ..];
    const end = std.mem.indexOfAny(u8, rest, " \t\r\n") orelse rest.len;
    const r = rest[0..end];
    if (std.mem.eql(u8, r, "self")) return .{ .kind = .reasoning, .recipient = r };
    if (r.len == 0 or std.mem.eql(u8, r, "user") or std.mem.eql(u8, r, "commentary")) return .{ .kind = .answer, .recipient = r };
    return .{ .kind = .call, .recipient = r };
}

fn add(a: Allocator, out: *std.ArrayList(u8), piece: []const u8) Allocator.Error!void {
    if (piece.len == 0) return;
    if (out.items.len > 0) try out.appendSlice(a, "\n\n");
    try out.appendSlice(a, piece);
}

/// The template's attribute escapes undone (`esc_attr`).
fn unescape(a: Allocator, s: []const u8) Allocator.Error![]const u8 {
    if (std.mem.indexOfScalar(u8, s, '&') == null) return s;
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    outer: while (i < s.len) {
        for ([_][2][]const u8{ .{ "&amp;", "&" }, .{ "&quot;", "\"" }, .{ "&#39;", "'" }, .{ "&lt;", "<" }, .{ "&gt;", ">" } }) |e| {
            if (std.mem.startsWith(u8, s[i..], e[0])) {
                try out.appendSlice(a, e[1]);
                i += e[0].len;
                continue :outer;
            }
        }
        try out.append(a, s[i]);
        i += 1;
    }
    return out.items;
}

/// Each `<atem:invoke>` of a body as a `<tool_call>` block (parameter values kept byte for byte); null: none in it.
pub fn calls(a: Allocator, body: []const u8) Allocator.Error!?[]const u8 {
    const inv = "<atem:invoke name=\"";
    const par = "<atem:parameter name=\"";
    var out: std.ArrayList(u8) = .empty;
    var pos: usize = 0;
    while (findTag(body, pos, inv)) |s| {
        const q = std.mem.indexOfPos(u8, body, s + inv.len, "\">") orelse break;
        const close = std.mem.indexOfPos(u8, body, q, "</atem:invoke>") orelse body.len;
        if (out.items.len > 0) try out.append(a, '\n');
        try out.print(a, "<tool_call>\n<function={s}>\n", .{try unescape(a, body[s + inv.len .. q])});
        var p = q + 2;
        while (findTag(body[0..close], p, par)) |ps| {
            const pq = std.mem.indexOfPos(u8, body, ps + par.len, "\">") orelse break;
            const pe = std.mem.indexOfPos(u8, body[0..close], pq + 2, "</atem:parameter>") orelse close;
            try out.print(a, "<parameter={s}>\n{s}\n</parameter>\n", .{ try unescape(a, body[ps + par.len .. pq]), body[pq + 2 .. pe] });
            p = pe;
        }
        try out.appendSlice(a, "</function>\n</tool_call>");
        pos = close;
    }
    return if (out.items.len > 0) out.items else null;
}

/// (reasoning, answer) of a reply so far. While it streams, an open call message and an unfinished header are held
/// back, and bodies are kept as written (so each split extends the last); a finished reply's bodies are trimmed.
pub fn split(a: Allocator, full: []const u8, finished: bool) Allocator.Error!Split {
    var reasoning: std.ArrayList(u8) = .empty;
    var answer: std.ArrayList(u8) = .empty;
    var pos: usize = 0;
    var any = false;
    while (pos < full.len) {
        const m = findTag(full, pos, message) orelse break;
        var header = full[pos..m];
        if (std.mem.startsWith(u8, header, start)) header = header[start.len..];
        const k = kindOf(header);
        const b0 = m + message.len;
        var b1 = full.len;
        var next = full.len;
        var closed = false;
        for (ends ++ [_][]const u8{start}) |e| if (findTag(full[0..b1], b0, e)) |at| {
            b1 = at;
            next = if (std.mem.eql(u8, e, start)) at else at + e.len;
            closed = true;
        };
        any = true;
        if (!closed and !finished) { // a tail that could begin a marker waits for the next token
            var held: usize = 0;
            for (ends ++ [_][]const u8{start}) |e| held = @max(held, reply_text.partialTag(full[b0..b1], e));
            b1 -= held;
        }
        const body = if (finished) reply_text.pyStrip(full[b0..b1]) else full[b0..b1];
        switch (k.kind) {
            .reasoning => try add(a, &reasoning, body),
            .answer => try add(a, &answer, body),
            .call => if (closed or finished) try add(a, &answer, (try calls(a, body)) orelse body),
        }
        if (!closed) break;
        pos = next;
    }
    if (!any and finished) return .{ .reasoning = "", .answer = reply_text.pyStrip(full) }; // no message markup at all
    return .{ .reasoning = reasoning.items, .answer = answer.items };
}

test "recipient replies: reasoning, commentary and a call, streamed prefixes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const reply = " to=self<|message|>We need the weather.<|eom|><|start|>assistant to=commentary<|message|>Checking." ++
        "<|eom|><|start|>assistant to=get_weather<|message|><atem:function_calls>\n<atem:invoke name=\"get_weather\">\n" ++
        "<atem:parameter name=\"city\">Sydney</atem:parameter>\n<atem:parameter name=\"days\">3</atem:parameter>\n" ++
        "</atem:invoke>\n</atem:function_calls>";
    const done = try split(a, reply, true);
    try std.testing.expectEqualStrings("We need the weather.", done.reasoning);
    try std.testing.expectEqualStrings("Checking.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nSydney\n" ++
        "</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>", done.answer);
    var last: Split = .{ .reasoning = "", .answer = "" };
    for (1..reply.len) |n| { // streamed: every split extends the previous one
        const s = try split(a, reply[0..n], false);
        try std.testing.expect(std.mem.startsWith(u8, s.reasoning, last.reasoning));
        try std.testing.expect(std.mem.startsWith(u8, s.answer, last.answer));
        try std.testing.expect(std.mem.indexOf(u8, s.answer, "atem") == null);
        last = s;
    }
    const plain = try split(a, " to=self<|message|>Think.<|eom|><|start|>assistant to=user<|message|>Blue sky.", true);
    try std.testing.expectEqualStrings("Think.", plain.reasoning);
    try std.testing.expectEqualStrings("Blue sky.", plain.answer);
    try std.testing.expectEqualStrings("no markup", (try split(a, " no markup ", true)).answer);
}
