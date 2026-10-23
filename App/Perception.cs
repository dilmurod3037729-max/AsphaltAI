using Android.Content;
using Android.Graphics;
using Android.Util;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace AsphaltApp;

/// <summary>Yo'lni "ko'rish": SegFormer-B0 (yo'l/devor/mashina) + Depth Anything V2 Small (nisbiy chuqurlik), ikkalasi ONNX, qurilmada.
/// Chiqish: 15 ustun x 5 xususiyat (yo'l bo'sh masofasi, devor, to'siq-yo'l chetiga, mashina, chuqurlik yaqinligi), hammasi 0..1.</summary>
public sealed class Perception : IDisposable
{
    public const int Bins = 15, Feats = 5, Extra = 2, Len = Bins * Feats + Extra;
    const int SW = 352, SH = 160, SOW = 88, SOH = 40, DW = 224, DH = 112;
    static readonly float[] Mean = { 0.485f, 0.456f, 0.406f }, Std = { 0.229f, 0.224f, 0.225f };
    // HUD (kadr ulushi x0,y0,x1,y1): ustun bo'yicha pastdagi piksellardan to'ldiriladi
    static readonly float[][] Hud =
    {
        new[] { 0f, 0f, .28f, .32f }, new[] { .30f, 0f, .68f, .08f }, new[] { .80f, 0f, 1f, .48f },
        new[] { 0f, .30f, .20f, .56f }, new[] { .70f, .30f, 1f, .50f },
    };
    const float EgoX0 = .36f, EgoX1 = .66f, EgoY0 = .62f, Horizon = .38f;

    readonly InferenceSession seg, depth;
    int busy;
    public volatile float[] Last;
    public long LastAt; public int LastMs, SegMs, DepthMs;
    public string DebugPath, LogPath, EventDir;
    readonly Queue<int[]> ring = new();
    int ringW, ringH;

    public Perception(Context ctx)
    {
        seg = Load(ctx, "seg.onnx"); depth = Load(ctx, "depth.onnx");
    }

    static InferenceSession Load(Context ctx, string name)
    {
        var p = System.IO.Path.Combine(ctx.FilesDir.AbsolutePath, name);
        using (var s = ctx.Assets.Open(name)) using (var f = System.IO.File.Create(p)) s.CopyTo(f);   // har safar: model almashganda eski nusxa qolmasin
        var o = new SessionOptions { IntraOpNumThreads = 4 };
        return new InferenceSession(p, o);
    }

    /// <summary>Kadrni qabul qiladi (band bo'lsa o'tkazib yuboradi). RGBA_8888, qator qadami bilan.</summary>
    public void Submit(ReadOnlySpan<byte> rgba, int stride, int W, int H)
    {
        if (Interlocked.Exchange(ref busy, 1) == 1) return;
        float[] sx = Prep(rgba, stride, W, H, SW, SH), dx = Prep(rgba, stride, W, H, DW, DH);
        Task.Run(() =>
        {
            try { Infer(sx, dx); }
            catch (Exception e) { Log.Warn("AsphaltAI", "perception: " + e.Message); }
            finally { Volatile.Write(ref busy, 0); }
        });
    }

    static float[] Prep(ReadOnlySpan<byte> b, int stride, int W, int H, int ow, int oh)
    {
        var o = new float[3 * ow * oh];
        for (int y = 0; y < oh; y++)
        {
            float fv = (y + .5f) / oh;
            for (int x = 0; x < ow; x++)
            {
                float fu = (x + .5f) / ow, v = fv;
                foreach (var h in Hud)
                    if (fu >= h[0] && fu < h[2] && v >= h[1] && v < h[3]) v = Math.Min(.995f, h[3] + .01f);
                float sx = Math.Clamp(fu * W - .5f, 0, W - 1.001f), sy = Math.Clamp(v * H - .5f, 0, H - 1.001f);
                int x0 = (int)sx, y0 = (int)sy; float ax = sx - x0, ay = sy - y0;
                int p00 = y0 * stride + x0 * 4, p01 = p00 + 4, p10 = p00 + stride, p11 = p10 + 4;
                for (int c = 0; c < 3; c++)
                {
                    float val = (b[p00 + c] * (1 - ax) + b[p01 + c] * ax) * (1 - ay) + (b[p10 + c] * (1 - ax) + b[p11 + c] * ax) * ay;
                    o[c * ow * oh + y * ow + x] = (val / 255f - Mean[c]) / Std[c];
                }
            }
        }
        return o;
    }

    void Infer(float[] sx, float[] dx)
    {
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        float[] lg, dp; long tA, tB;
        using (var r = seg.Run(new[] { NamedOnnxValue.CreateFromTensor("x", new DenseTensor<float>(sx, new[] { 1, 3, SH, SW })) }))
            lg = r.First().AsTensor<float>().ToArray();
        tA = System.Diagnostics.Stopwatch.GetTimestamp();
        using (var r = depth.Run(new[] { NamedOnnxValue.CreateFromTensor("x", new DenseTensor<float>(dx, new[] { 1, 3, DH, DW })) }))
            dp = r.First().AsTensor<float>().ToArray();
        tB = System.Diagnostics.Stopwatch.GetTimestamp();

        var cls = new byte[SOW * SOH];
        for (int i = 0; i < cls.Length; i++)
        {
            int best = 0; float bv = float.MinValue;
            for (int c = 0; c < 19; c++) { float v = lg[c * cls.Length + i]; if (v > bv) { bv = v; best = c; } }
            cls[i] = (byte)best;
        }
        float dmax = dp.Max(), dmin = dp.Min();
        var f = Features(cls, dp, dmin, dmax);
        SegMs = (int)((tA - t0) * 1000 / System.Diagnostics.Stopwatch.Frequency); DepthMs = (int)((tB - tA) * 1000 / System.Diagnostics.Stopwatch.Frequency);
        Last = f; LastAt = Android.OS.SystemClock.ElapsedRealtime();
        LastMs = (int)((System.Diagnostics.Stopwatch.GetTimestamp() - t0) * 1000 / System.Diagnostics.Stopwatch.Frequency);
        if (LogPath != null)
        {
            System.IO.File.AppendAllText(LogPath, $"{LastAt},{AiService.Speed},{LastMs},{SegMs},{DepthMs}," + string.Join(",", f.Select(v => v.ToString("F2", System.Globalization.CultureInfo.InvariantCulture))) + "\n");
        }
        if (EventDir != null) Push(Compose(sx, cls, dp, dmin, dmax, f));
        if (DebugPath != null && LastAt - dbgAt > 2000) { dbgAt = LastAt; Debug(cls, dp, dmin, dmax, f); }
    }
    long dbgAt;

    static bool IsCar(byte c) => c == 11 || c == 12 || c == 13 || c == 14 || c == 15 || c == 17 || c == 18;
    static bool IsWall(byte c) => c == 2 || c == 3 || c == 4;

    static float[] Features(byte[] cls, float[] dp, float dmin, float dmax)
    {
        var f = new float[Len];
        int hr = (int)(Horizon * SOH), egoRow = (int)(EgoY0 * SOH);
        int ex0 = (int)(EgoX0 * SOW), ex1 = (int)(EgoX1 * SOW);
        for (int k = 0; k < Bins; k++)
        {
            int c0 = k * SOW / Bins, c1 = (k + 1) * SOW / Bins;
            bool egoBin = c1 > ex0 && c0 < ex1;
            int start = egoBin ? egoRow - 1 : SOH - 2;
            float span = Math.Max(1, start - hr);
            int roadEnd = hr, bad = 0; bool broke = false;
            int wallRow = hr, curbRow = hr, carRow = hr;
            for (int y = start; y > hr; y--)
            {
                int road = 0, wall = 0, curb = 0, car = 0, n = c1 - c0;
                for (int x = c0; x < c1; x++)
                {
                    byte c = cls[y * SOW + x];
                    if (c == 0) road++; else if (IsWall(c)) wall++; else if (c == 1) curb++; else if (IsCar(c) && !(y >= egoRow && x >= ex0 && x < ex1)) car++;
                }
                if (wall * 2 > n && wallRow == hr) wallRow = y;
                if (curb * 2 > n && curbRow == hr) curbRow = y;
                if (car * 3 > n && carRow == hr) carRow = y;
                if (!broke)
                {
                    bad = road * 2 > n ? 0 : bad + 1;
                    if (bad >= 3) { roadEnd = y + 2; broke = true; }
                }
            }
            if (!broke) roadEnd = hr;
            f[k * Feats + 0] = Math.Clamp((start - roadEnd) / span, 0, 1);          // 1 = ufqqacha yo'l ochiq
            f[k * Feats + 1] = wallRow == hr ? 0 : Math.Clamp((wallRow - hr) / span, 0, 1);   // 1 = devor pastda (yaqin)
            f[k * Feats + 2] = curbRow == hr ? 0 : Math.Clamp((curbRow - hr) / span, 0, 1);
            f[k * Feats + 3] = carRow == hr ? 0 : Math.Clamp((carRow - hr) / span, 0, 1);
            // chuqurlik yaqinligi: ufq ustidagi oraliq qatorlar, mashina ustidan
            int dy0 = (int)(.45f * DH), dy1 = (int)(.60f * DH), dx0 = k * DW / Bins, dx1 = (k + 1) * DW / Bins;
            double s = 0; int cnt = 0;
            for (int y = dy0; y < dy1; y++) for (int x = dx0; x < dx1; x++) { s += dp[y * DW + x]; cnt++; }
            f[k * Feats + 4] = (float)((s / cnt - dmin) / Math.Max(1e-6, dmax - dmin));
        }
        // yo'l oldinda (uzoq markaz) va yo'ldan tashqarida (yaqin polosa) ulushi
        int fr0 = (int)(.40f * SOH), fr1 = (int)(.55f * SOH), fc0 = (int)(.38f * SOW), fc1 = (int)(.62f * SOW);
        int nr0 = (int)(.50f * SOH), nr1 = (int)(.61f * SOH), nc0 = (int)(.15f * SOW), nc1 = (int)(.85f * SOW);
        int farRoad = 0, tot = 0, offN = 0, offTot = 0;
        for (int y = fr0; y < fr1; y++) for (int x = fc0; x < fc1; x++) { tot++; if (cls[y * SOW + x] == 0) farRoad++; }
        for (int y = nr0; y < nr1; y++)
            for (int x = nc0; x < nc1; x++)
            {
                byte c = cls[y * SOW + x];
                if (IsCar(c)) continue;
                offTot++; if (c != 0) offN++;
            }
        f[Bins * Feats] = tot == 0 ? 0 : farRoad / (float)tot;
        f[Bins * Feats + 1] = offTot == 0 ? 0 : offN / (float)offTot;
        return f;
    }

    static readonly int[] Pal =
    {
        unchecked((int)0xFF804080), unchecked((int)0xFFF423E8), unchecked((int)0xFF464646), unchecked((int)0xFF66669C), unchecked((int)0xFFBE9999),
        unchecked((int)0xFF999999), unchecked((int)0xFFFAAA1E), unchecked((int)0xFFDCDC00), unchecked((int)0xFF6B8E23), unchecked((int)0xFF98FB98),
        unchecked((int)0xFF4682B4), unchecked((int)0xFFDC143C), unchecked((int)0xFFFF0000), unchecked((int)0xFF00008E), unchecked((int)0xFF000046),
        unchecked((int)0xFF003C64), unchecked((int)0xFF005064), unchecked((int)0xFF0000E6), unchecked((int)0xFF770B20),
    };

    void Push(int[] img)
    {
        lock (ring) { ring.Enqueue(img); while (ring.Count > 4) ring.Dequeue(); }
    }

    /// <summary>[kadr | segmentatsiya+ustunlar | chuqurlik] — 352x160 har biri, bitta qatorda.</summary>
    int[] Compose(float[] sx, byte[] cls, float[] dp, float dmin, float dmax, float[] f)
    {
        const int W = SW, H = SH; ringW = 3 * W; ringH = H;
        var px = new int[3 * W * H];
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
            {
                int r = Math.Clamp((int)((sx[0 * W * H + y * W + x] * Std[0] + Mean[0]) * 255), 0, 255);
                int g = Math.Clamp((int)((sx[1 * W * H + y * W + x] * Std[1] + Mean[1]) * 255), 0, 255);
                int b = Math.Clamp((int)((sx[2 * W * H + y * W + x] * Std[2] + Mean[2]) * 255), 0, 255);
                px[y * 3 * W + x] = unchecked((int)0xFF000000) | r << 16 | g << 8 | b;
                px[y * 3 * W + W + x] = Pal[cls[(y * SOH / H) * SOW + x * SOW / W]];
                int gg = (int)(255 * (dp[(y * DH / H) * DW + x * DW / W] - dmin) / Math.Max(1e-6, dmax - dmin));
                px[y * 3 * W + 2 * W + x] = unchecked((int)0xFF000000) | gg << 16 | gg << 8 | gg;
            }
        for (int k = 0; k < Bins; k++)
        {
            int x0 = W + k * W / Bins, x1 = W + (k + 1) * W / Bins - 2;
            int h = (int)(f[k * Feats] * 40);
            for (int y = H - h; y < H; y++) for (int x = x0 + 2; x < x1; x++) px[y * 3 * W + x] = unchecked((int)0xFF00FF00);
            if (f[k * Feats + 3] > 0) { int cy = H - 1 - (int)(f[k * Feats + 3] * 40); for (int x = x0 + 2; x < x1; x++) px[cy * 3 * W + x] = unchecked((int)0xFFFFFFFF); }
            if (f[k * Feats + 1] > 0) { int cy = H - 1 - (int)(f[k * Feats + 1] * 40); for (int x = x0 + 2; x < x1; x++) px[cy * 3 * W + x] = unchecked((int)0xFFFF4040); }
        }
        return px;
    }

    /// <summary>Oxirgi 4 ta xulosani bitta PNG ga (yuqoridan pastga, eng eskisi birinchi) yozadi.</summary>
    public void SaveEvent(string tag)
    {
        int[][] frames;
        lock (ring) frames = ring.ToArray();
        if (frames.Length == 0 || EventDir == null) return;
        var all = new int[frames.Length * ringW * ringH];
        for (int i = 0; i < frames.Length; i++) Array.Copy(frames[i], 0, all, i * ringW * ringH, ringW * ringH);
        using var bmp = Bitmap.CreateBitmap(all, ringW, ringH * frames.Length, Bitmap.Config.Argb8888);
        using var fs = System.IO.File.Create(System.IO.Path.Combine(EventDir, $"evt_{tag}.png"));
        bmp.Compress(Bitmap.CompressFormat.Png, 90, fs);
    }

    void Debug(byte[] cls, float[] dp, float dmin, float dmax, float[] f)
    {
        const int S = 4, W = SOW * S, H = SOH * S;
        var px = new int[2 * W * H];
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
            {
                px[y * 2 * W + x] = Pal[cls[(y / S) * SOW + x / S]];
                int g = (int)(255 * (dp[(y * DH / H) * DW + x * DW / W] - dmin) / Math.Max(1e-6, dmax - dmin));
                px[y * 2 * W + W + x] = unchecked((int)0xFF000000) | g << 16 | g << 8 | g;
            }
        for (int k = 0; k < Bins; k++)
        {
            int x0 = k * W / Bins, x1 = (k + 1) * W / Bins - 2;
            int h = (int)(f[k * Feats] * 60);
            for (int y = H - h; y < H; y++) for (int x = x0 + 2; x < x1; x++) px[y * 2 * W + x] = unchecked((int)0xFF00FF00);
            int cy = H - 1 - (int)(f[k * Feats + 3] * 60);
            if (f[k * Feats + 3] > 0) for (int x = x0 + 2; x < x1; x++) px[cy * 2 * W + x] = unchecked((int)0xFFFFFFFF);
        }
        using var bmp = Bitmap.CreateBitmap(px, 2 * W, H, Bitmap.Config.Argb8888);
        using var fs = System.IO.File.Create(DebugPath);
        bmp.Compress(Bitmap.CompressFormat.Png, 100, fs);
    }

    public void Dispose() { seg.Dispose(); depth.Dispose(); }
}
