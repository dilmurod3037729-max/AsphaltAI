/// <summary>Kadrdagi katta oq raqam bo'laklarini topadi va 20x24 ga me'yorlaydi. OCR ishlatilmaydi.</summary>
public static class Glyphs
{
    public const int CW = 20, CH = 24;
    public static int Thresh = int.TryParse(Environment.GetEnvironmentVariable("TH"), out var t0) ? t0 : 190;

    public struct Box { public int Area; public int X0, Y0, X1, Y1; public int W => X1 - X0 + 1; public int H => Y1 - Y0 + 1; }

    public static List<Box> Components(Frame f)
    {
        int w = f.W, h = f.H;
        var seen = new bool[w * h];
        var res = new List<Box>();
        var st = new Stack<int>();
        for (int s = 0; s < seen.Length; s++)
        {
            if (seen[s] || f.White[s] < Thresh) continue;
            var b = new Box { X0 = s % w, X1 = s % w, Y0 = s / w, Y1 = s / w };
            int area = 0;
            seen[s] = true; st.Push(s);
            while (st.Count > 0)
            {
                int p = st.Pop(); int x = p % w, y = p / w; area++;
                if (x < b.X0) b.X0 = x; if (x > b.X1) b.X1 = x; if (y < b.Y0) b.Y0 = y; if (y > b.Y1) b.Y1 = y;
                if (x > 0) Push(p - 1); if (x < w - 1) Push(p + 1); if (y > 0) Push(p - w); if (y < h - 1) Push(p + w);
            }
            b.Area = area; if (area >= 12) res.Add(b);
            void Push(int q) { if (!seen[q] && f.White[q] >= Thresh) { seen[q] = true; st.Push(q); } }
        }
        return res;
    }

    /// <summary>Bir raqamning blur/chegara sabab uzilgan bo'laklarini (x oralig'i ustma-ust, tik bo'shliq kichik) birlashtiradi.</summary>
    static List<Box> MergeBroken(List<Box> cs)
    {
        var l = new List<Box>(cs);
        bool ch = true;
        while (ch)
        {
            ch = false;
            for (int i = 0; i < l.Count && !ch; i++)
                for (int j = i + 1; j < l.Count && !ch; j++)
                {
                    var a = l[i]; var b = l[j];
                    int ov = Math.Min(a.X1, b.X1) - Math.Max(a.X0, b.X0) + 1;
                    int gap = Math.Max(a.Y0, b.Y0) - Math.Min(a.Y1, b.Y1) - 1;
                    int uh = Math.Max(a.Y1, b.Y1) - Math.Min(a.Y0, b.Y0) + 1;
                    if (ov >= 0.5 * Math.Min(a.W, b.W) && gap <= 0.15 * uh && gap <= 4)
                    {
                        l[i] = new Box { Area = a.Area + b.Area, X0 = Math.Min(a.X0, b.X0), X1 = Math.Max(a.X1, b.X1), Y0 = Math.Min(a.Y0, b.Y0), Y1 = Math.Max(a.Y1, b.Y1) };
                        l.RemoveAt(j); ch = true;
                    }
                }
        }
        return l;
    }

    /// <summary>Raqam qatorini (o'ngdagi eng uzun zanjir) ajratadi, yopishgan raqamlarni ustun minimumi bo'yicha kesadi.</summary>
    public static List<Box> DigitRow(Frame f)
    {
        var cs = Components(f).Where(c => c.H < 0.9 * f.H && c.Area < 0.72 * c.W * c.H).ToList();
        cs = MergeBroken(cs);
        if (cs.Count == 0) return new();
        int hmax = cs.Max(c => c.H);
        if (hmax < 10) return new();
        var big = cs.Where(c => c.H >= 0.78 * hmax && c.W <= 2.4 * c.H).OrderBy(c => c.X0).ToList();
        // bir qatorda: markaz y yaqin
        if (big.Count == 0) return new();
        var t = big.OrderByDescending(c => c.H).First();
        double cy = (t.Y0 + t.Y1) / 2.0;
        big = big.Where(c => Math.Abs((c.Y0 + c.Y1) / 2.0 - cy) < 0.3 * hmax).ToList();
        // o'ngdan zanjir: bo'shliq <= 0.6*h
        var chain = new List<Box>();
        for (int i = big.Count - 1; i >= 0; i--)
        {
            if (chain.Count > 0 && chain[0].X0 - big[i].X1 > 0.6 * hmax) break;
            chain.Insert(0, big[i]);
        }
        var outl = new List<Box>();
        foreach (var c in chain)
        {
            int n = Math.Max(1, (int)Math.Round(c.W / (0.72 * c.H)));
            if (c.W < 0.85 * c.H) n = 1;
            if (n == 1) { outl.Add(c); continue; }
            int prev = c.X0;
            for (int k = 1; k < n; k++)
            {
                int ex = c.X0 + (int)(c.W * (double)k / n), r = Math.Max(2, (int)(0.15 * c.H));
                int best = ex, bv = int.MaxValue;
                for (int x = ex - r; x <= ex + r; x++)
                {
                    if (x <= prev + 2 || x >= c.X1 - 2) continue;
                    int v = 0;
                    for (int y = c.Y0; y <= c.Y1; y++) v += f.White[y * f.W + x] >= Thresh ? 1 : 0;
                    if (v < bv) { bv = v; best = x; }
                }
                outl.Add(new Box { X0 = prev, X1 = best - 1, Y0 = c.Y0, Y1 = c.Y1 });
                prev = best;
            }
            outl.Add(new Box { X0 = prev, X1 = c.X1, Y0 = c.Y0, Y1 = c.Y1 });
        }
        return outl;
    }

    /// <summary>Qutini balandligi bo'yicha 24 ga keltirib, 20x24 kanvas markaziga qo'yadi; qiymat 0..1.</summary>
    public static float[] Normalize(Frame f, Box b)
    {
        var o = new float[CW * CH];
        double sc = (double)CH / b.H;
        double ox = (CW - b.W * sc) / 2;
        for (int y = 0; y < CH; y++)
            for (int x = 0; x < CW; x++)
            {
                double sx = b.X0 + (x - ox + 0.5) / sc - 0.5, sy = b.Y0 + (y + 0.5) / sc - 0.5;
                if (sx < b.X0 - 0.5 || sx > b.X1 + 0.5) continue;
                o[y * CW + x] = Sample(f, sx, sy);
            }
        return o;
    }

    static float Sample(Frame f, double x, double y)
    {
        int x0 = (int)Math.Floor(x), y0 = (int)Math.Floor(y);
        double fx = x - x0, fy = y - y0, v = 0;
        for (int j = 0; j < 2; j++)
            for (int i = 0; i < 2; i++)
            {
                int xx = Math.Clamp(x0 + i, 0, f.W - 1), yy = Math.Clamp(y0 + j, 0, f.H - 1);
                double wgt = (i == 0 ? 1 - fx : fx) * (j == 0 ? 1 - fy : fy);
                v += wgt * Math.Clamp((f.White[yy * f.W + xx] - 120) / 135.0, 0, 1);
            }
        return (float)v;
    }
}
