using System.Diagnostics;
using System.Drawing;
using System.Globalization;

/*  Speed learn <crops_dir> <out_dir>          bo'laklarni klasterlab markaz rasmini chiqaradi (centers.png, centers.txt)
    Speed name  <centers.txt> <labels> <out>   labels = "0,1,2,..." klaster tartibida raqam nomlari -> templates.txt
    Speed read  <templates.txt> <img|dir>      o'qish
    Speed bench <templates.txt> <dir>          tezlik o'lchovi                                                    */
static class P
{
    static int Main(string[] a)
    {
        switch (a[0])
        {
            case "learn": return Learn(a[1], a[2], a.Length > 3 ? int.Parse(a[3]) : 16);
            case "name": return Name(a[1], a[2], a[3]);
            case "read": return Rd(a[1], a[2]);
            case "box": Dbg.Boxes(a[1], a[2]); return 0;
            case "dist": return DistRd(a[1], a[2]);
            case "bench": return Bench(a[1], a[2]);
        }
        return 1;
    }

    static int MaxGlyphs => int.TryParse(Environment.GetEnvironmentVariable("MAXG"), out var m) ? m : 3;   // DIST uchun MAXG=4 ('100%')

    static IEnumerable<string> Files(string p) => Directory.Exists(p) ? Directory.GetFiles(p, "*.png").OrderBy(x => x) : new[] { p };

    static int Learn(string dir, string outDir, int k)
    {
        Directory.CreateDirectory(outDir);
        var samples = new List<float[]>();
        foreach (var fn in Files(dir))
        {
            var f = FrameIO.Load(fn);
            var row = Glyphs.DigitRow(f);
            if (row.Count == 0 || row.Count > MaxGlyphs) continue;
            foreach (var b in row) samples.Add(Reader.Center(Glyphs.Normalize(f, b)));
        }
        Console.WriteLine($"{samples.Count} ta bo'lak");
        var rnd = new Random(7);
        var cen = Enumerable.Range(0, k).Select(_ => samples[rnd.Next(samples.Count)]).ToList();
        var asg = new int[samples.Count];
        for (int it = 0; it < 12; it++)
        {
            for (int i = 0; i < samples.Count; i++)
            {
                int b = 0; float bs = -2;
                for (int c = 0; c < k; c++) { float s = Dot(cen[c], samples[i]); if (s > bs) { bs = s; b = c; } }
                asg[i] = b;
            }
            for (int c = 0; c < k; c++)
            {
                var m = new float[Glyphs.CW * Glyphs.CH]; int n = 0;
                for (int i = 0; i < samples.Count; i++) if (asg[i] == c) { n++; for (int j = 0; j < m.Length; j++) m[j] += samples[i][j]; }
                if (n > 0) cen[c] = Reader.Center(m);
            }
        }
        var ord = Enumerable.Range(0, k).OrderByDescending(c => asg.Count(z => z == c)).Take(24).ToList();
        cen = ord.Select(c => cen[c]).ToList();
        asg = asg.Select(z => ord.IndexOf(z)).ToArray();
        k = ord.Count;
        var sheet = new Bitmap(k * (Glyphs.CW * 4 + 6), Glyphs.CH * 4 + 18);
        using (var g = Graphics.FromImage(sheet))
        {
            g.Clear(Color.White);
            for (int c = 0; c < k; c++)
            {
                float mn = cen[c].Min(), mx = cen[c].Max();
                for (int y = 0; y < Glyphs.CH; y++)
                    for (int x = 0; x < Glyphs.CW; x++)
                    {
                        int v = (int)(255 * (1 - (cen[c][y * Glyphs.CW + x] - mn) / (mx - mn)));
                        using var br = new SolidBrush(Color.FromArgb(v, v, v));
                        g.FillRectangle(br, c * (Glyphs.CW * 4 + 6) + x * 4, 18 + y * 4, 4, 4);
                    }
                g.DrawString($"{c} n={asg.Count(z => z == c)}", SystemFonts.DefaultFont, Brushes.Red, c * (Glyphs.CW * 4 + 6), 2);
            }
        }
        sheet.Save(Path.Combine(outDir, "centers.png"));
        File.WriteAllLines(Path.Combine(outDir, "centers.txt"), cen.Select(c => string.Join(' ', c.Select(v => v.ToString("R", CultureInfo.InvariantCulture)))));
        return 0;
    }

    static float Dot(float[] a, float[] b) { float s = 0; for (int i = 0; i < a.Length; i++) s += a[i] * b[i]; return s; }

    static int Name(string centers, string labels, string outp)
    {
        var lines = File.ReadAllLines(centers);
        var lab = labels.Split(',');
        var o = new List<string>();
        for (int i = 0; i < lines.Length; i++)
            if (lab[i] != "-") o.Add(lab[i] + " " + lines[i]);
        File.WriteAllLines(outp, o);
        Console.WriteLine(o.Count + " shablon");
        return 0;
    }

    static int Rd(string tpl, string path)
    {
        var r = Reader.Load(tpl);
        foreach (var fn in Files(path))
        {
            var res = r.Read(FrameIO.Load(fn));
            Console.WriteLine($"{Path.GetFileName(fn)}\t{(res.Value?.ToString() ?? "-")}\t{res.Digits}\t{res.MinScore:F2}");
        }
        return 0;
    }

    static int DistRd(string tpl, string path)
    {
        var r = Reader.Load(tpl);
        foreach (var fn in Files(path))
        {
            var v = r.ReadPercent(FrameIO.LoadCrop(fn, 20, 10, 63, 38));   // DIST qirqimi (230,70) ga nisbatan: raqamlar, '%' siz
            Console.WriteLine($"{Path.GetFileName(fn)}	{(v?.ToString() ?? "-")}");
        }
        return 0;
    }

    static int Bench(string tpl, string dir)
    {
        var r = Reader.Load(tpl);
        var fr = Files(dir).Take(300).Select(FrameIO.Load).ToList();
        foreach (var f in fr) r.Read(f);
        var sw = Stopwatch.StartNew(); int n = 0;
        for (int rep = 0; rep < 20; rep++) foreach (var f in fr) { r.Read(f); n++; }
        sw.Stop();
        Console.WriteLine($"{n} kadr, kadr boshiga {sw.Elapsed.TotalMilliseconds * 1000 / n:F0} mks ({fr[0].W}x{fr[0].H})");
        return 0;
    }
}
static class Dbg
{
    public static void Boxes(string tpl, string fn)
    {
        var f = FrameIO.Load(fn);
        foreach (var c in Glyphs.Components(f)) Console.WriteLine($"comp x{c.X0}-{c.X1} y{c.Y0}-{c.Y1} w{c.W} h{c.H}");
        Console.WriteLine("row:");
        foreach (var c in Glyphs.DigitRow(f)) Console.WriteLine($"  x{c.X0}-{c.X1} y{c.Y0}-{c.Y1}");
    }
}
