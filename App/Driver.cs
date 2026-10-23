/// <summary>Gamepad ustidagi yuqori daraja buyruqlar. Skript: buyruqlar ';' bilan ajratiladi, ketma-ket bajariladi.
/// Tugma xaritasi (drift/nitro/...) o'yin ichida tasdiqlanmagan: `map drift X` bilan qayta bog'lanadi.</summary>
public sealed class Driver
{
    readonly Pad pad;
    readonly Dictionary<string, string> map = new()
    {
        ["drift"] = "LB", ["nitro"] = "RB", ["respawn"] = "B", ["menu"] = "START", ["emote"] = "RS",
    };
    int tapMs = 60, gapMs = 70;
    CancellationTokenSource cts = new();
    Task tail = Task.CompletedTask;
    readonly object gate = new();

    public Driver(Pad pad) { this.pad = pad; }

    /// <summary>Skriptni navbatga qo'yadi (oldingi tugagach bajariladi). `stop` navbatni bekor qilib padni nolga tushiradi.</summary>
    public string Run(string script)
    {
        if (script.Trim() == "stop") { Stop(); return "ok"; }
        lock (gate)
        {
            var tok = cts.Token;
            tail = tail.ContinueWith(_ => Exec(script, tok), TaskScheduler.Default).Unwrap();
        }
        return "ok";
    }

    public void Stop()
    {
        lock (gate)
        {
            cts.Cancel(); cts = new CancellationTokenSource();
            tail = Task.CompletedTask;
        }
        pad.Command("reset");
    }

    async Task Exec(string script, CancellationToken ct)
    {
        foreach (var raw in script.Split(';'))
        {
            if (ct.IsCancellationRequested) return;
            var c = raw.Trim();
            if (c.Length == 0) continue;
            try { await One(c, ct); }
            catch (OperationCanceledException) { return; }
            catch (Exception e) { Android.Util.Log.Warn("AsphaltAI", $"buyruq '{c}': {e.Message}"); }
        }
    }

    static double D(string s) => double.Parse(s, System.Globalization.CultureInfo.InvariantCulture);

    void Press(string name, bool down)
    {
        if (name == "LT" || name == "RT") pad.Axis(name, down ? 1 : 0); else pad.Button(name, down);
    }

    async Task Tap(string action, int ms, CancellationToken ct)
    {
        var b = map.TryGetValue(action, out var m) ? m : action.ToUpperInvariant();
        Press(b, true);
        try { await Task.Delay(ms, ct); } finally { Press(b, false); }
    }

    async Task Twice(string action, CancellationToken ct)
    {
        await Tap(action, tapMs, ct);
        await Task.Delay(gapMs, ct);
        await Tap(action, tapMs, ct);
    }

    async Task One(string c, CancellationToken ct)
    {
        var p = c.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        string a = p[0].ToLowerInvariant();
        int Ms(int i, int def) => p.Length > i ? int.Parse(p[i]) : def;
        switch (a)
        {
            case "steer": pad.Axis("X", D(p[1])); break;                 // -1 chap .. 1 o'ng
            case "gas": pad.Axis("RT", D(p[1])); break;                  // 0..1
            case "wait": await Task.Delay(int.Parse(p[1]), ct); break;
            case "drift":
            case "nitro":
                if (p[1] == "tap") await Tap(a, Ms(2, tapMs), ct);
                else Press(map[a], p[1] != "0");
                break;
            case "shockwave": await Twice("nitro", ct); break;           // nitro ikki marta (to'liq zaxirada)
            case "d360": await Twice("drift", ct); break;                // drift ikki marta tez
            case "punch": await Tap("drift", tapMs, ct); await Task.Delay(Ms(1, gapMs), ct); await Tap("nitro", tapMs, ct); break;
            case "respawn": case "menu": case "emote": await Tap(a, tapMs, ct); break;
            case "touchdrive": pad.Axis("HY", -1); await Task.Delay(250, ct); pad.Axis("HY", 0); break;   // 60 ms o'yin sezmaydi
            case "camera": pad.Axis("HY", 1); await Task.Delay(250, ct); pad.Axis("HY", 0); break;
            case "map": map[p[1].ToLowerInvariant()] = p[2].ToUpperInvariant(); break;   // map drift X
            case "set":
                if (p[1] == "tap") tapMs = int.Parse(p[2]); else if (p[1] == "gap") gapMs = int.Parse(p[2]);
                break;
            case "btn": case "axis": case "tap": case "reset": pad.Command(c); break;    // xom buyruqlar
            default: Android.Util.Log.Warn("AsphaltAI", "noma'lum buyruq: " + c); break;
        }
    }
}
