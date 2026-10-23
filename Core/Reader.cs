using System.Globalization;

/// <summary>Tezlik raqamini shablon NCC bilan tanish (0..605). Tarmoq ham, OCR ham yo'q.</summary>
public sealed class Reader
{
    public const int Max = 605;
    readonly List<(int d, float[] t)> tpl = new();

    public static Reader Load(string path)
    {
        var r = new Reader();
        foreach (var line in File.ReadLines(path))
        {
            var p = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (p.Length != Glyphs.CW * Glyphs.CH + 1) continue;
            var v = new float[p.Length - 1];
            for (int i = 0; i < v.Length; i++) v[i] = float.Parse(p[i + 1], CultureInfo.InvariantCulture);
            r.tpl.Add((int.Parse(p[0]), Center(v)));
        }
        return r;
    }

    public static float[] Center(float[] v)
    {
        float m = v.Average(), n = 0;
        var o = new float[v.Length];
        for (int i = 0; i < v.Length; i++) { o[i] = v[i] - m; n += o[i] * o[i]; }
        n = MathF.Sqrt(n);
        if (n > 1e-6f) for (int i = 0; i < o.Length; i++) o[i] /= n;
        return o;
    }

    public (int digit, float score) Classify(float[] g)
    {
        var c = Center(g);
        int bd = -1; float bs = -2;
        foreach (var (d, t) in tpl)
        {
            float s = 0;
            for (int i = 0; i < c.Length; i++) s += c[i] * t[i];
            if (s > bs) { bs = s; bd = d; }
        }
        return (bd, bs);
    }

    public struct Result { public int? Value; public string Digits; public float MinScore; }

    public static readonly int[] Ladder = { 225, 205, 185 };

    public Result Read(Frame f)
    {
        Result best = default; best.Digits = "";
        foreach (var t in Ladder)
        {
            Glyphs.Thresh = t;
            var r = ReadAt(f);
            if (r.Value == null || r.MinScore < 0.8f) { if (best.Digits == "") best = r; continue; }
            if (best.Value == null || r.Digits.Length > best.Digits.Length || (r.Digits.Length == best.Digits.Length && r.MinScore > best.MinScore)) best = r;
        }
        if (best.Value != null && best.MinScore < 0.8f) best.Value = null;
        return best;
    }

    Result ReadAt(Frame f)
    {
        var row = Glyphs.DigitRow(f);
        var r = new Result { Digits = "", MinScore = 1 };
        if (row.Count == 0 || row.Count > 3) return r;
        foreach (var b in row)
        {
            var (d, s) = Classify(Glyphs.Normalize(f, b));
            if (s < 0.55f) { r.MinScore = Math.Min(r.MinScore, s); r.Digits += "?"; continue; }
            r.MinScore = Math.Min(r.MinScore, s);
            r.Digits += d;
        }
        if (r.Digits.Contains('?')) return r;
        if (r.Digits.Length > 1 && r.Digits[0] == '0') return r;
        int v = int.Parse(r.Digits);
        if (v <= Max) r.Value = v;
        return r;
    }

    /// <summary>DIST% raqamlari (0..100). Kadr '%' belgisidan OLDINGI raqamlarni qamrashi kerak ('%' yopishib ketib o'qishni buzadi): 1-3 raqam.</summary>
    public int? ReadPercent(Frame f)
    {
        int? best = null; float bestScore = 0;
        foreach (var t in Ladder)
        {
            Glyphs.Thresh = t;
            var row = Glyphs.DigitRow(f);
            if (row.Count < 1 || row.Count > 3) continue;
            int v = 0; float min = 1; bool bad = false;
            foreach (var b in row)
            {
                var (d, s) = Classify(Glyphs.Normalize(f, b));
                min = Math.Min(min, s);
                if (s < 0.6f || d > 9) { bad = true; break; }
                v = v * 10 + d;
            }
            if (bad || v > 100) continue;
            if (min > bestScore) { bestScore = min; best = v; }
        }
        return best;
    }
}
