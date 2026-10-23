using Android.OS;
using Android.Util;

namespace AsphaltApp;

/// <summary>Tezlik + ko'rish (yo'l/devor/mashina/chuqurlik) kirishli o'rganuvchi agent. Yo'l haqida qoida YO'Q: faqat tezlikdan reward.
/// Tarmoq: shoxlangan DQN (rul 5, drift 3, nitro 3, gaz 2 — to'liq erkin boshqaruv), qurilmada onlayn o'rganadi, poygalar aro saqlanadi.</summary>
public sealed class Agent
{
    // Reward (har qadam, 250 ms). Manba g'oyalar: GT Sophy (progress + KE-ga mutanosib devor jarimasi + off-course*tezlik),
    // TORCS (oldinga tezlik, yon tezlik/markazdan chetlanish jazosi), Trackmania/GT (zich reward). Bizda trassa chizig'i yo'q:
    // "oldinga tezlik" = tezlik * yo'l oldinda borligi, devor jarimasi = yo'qotilgan kinetik energiya, off-course = ko'rishdan.
    const double UpW = 0.10, StableW = 0.5, StableBand = 1.5, GlitchMph = 80, RMin = -8, RMax = 5;
    const double WallW = 2.0, OffW = 0.1, SteerW = 0.05, ZigW = 0.10, ExtremeW = 0.10, ProgW = 2.0, WrongWayW = 3.0, NitroW = 3.0, WrongWayEvtW = 1.0;   // WrongWayEvtW: o'yinning WRONG WAY belgisi turgan har qadamga KICHIK jazo (agentni qo'rqoq qilmaslik uchun)   // NitroW: nitro chizig'i to'lishi (o'yinning o'z 'yaxshi haydash' signali)
    const bool OnlineTrain = false;   // onlayn DQN barqaror emas edi (G10 dan keyin yiqildi): siyosatni PC'da eng yaxshi epizodlardan o'rgatamiz (Vision/offline_bc.py)
    const int NStep = 5;
    // Marra va rekord: trassalar har xil, shuning uchun rekord TRASSA kaliti bo'yicha saqlanadi (records.json). Marra = HUD DIST >= 99.
    const double FinishBonus = 20, RecordBase = 10, RecordPerSec = 2, RecordCap = 20;
    public const int StepMs = 250, StaleMs = 1500;
    const double Gamma = 0.97;

    static readonly float[] SteerV = { -1f, -.5f, 0f, .5f, 1f };
    const int NS = 5, ND = 3, NN = 3, NOut = NS + ND + NN;   // gaz tarmog'i olib tashlandi: ma'lumotda gazni qo'yib yuborish/bosish farq qilmadi (o'yin o'zi tezlanadi)
    const int NF = Perception.Len;
    static readonly double[] Margin = { 0.30, 0.25, 0.10 };
    const float SafeT = 0.5f, SafeGap = 0.3f;
    int overrides;
    static readonly int[] Off = { 0, NS, NS + ND }, Cnt = { NS, ND, NN };

    readonly Driver drv; readonly Pad pad;
    Mlp net, target;
    readonly object netLock = new();
    readonly Random rnd = new();
    Replay rep;
    Vision vis; int nIn, vecLen; string kind = "h";
    readonly Dictionary<int, int> seen = new();
    Thread loop, trainer; volatile bool run, learn = true;
    public string Status = "to'xtagan";
    string dir; long steps;
    public string TrackKey = "default";
    const long RaceGapMs = 8000;
    long epStart, lastFresh; bool finished; Dictionary<string, double> records = new();
    float lastLoss; long lastEvt; int evts; System.IO.BinaryWriter recW;

    public Agent(Driver d, Pad p) { drv = d; pad = p; }
    public bool Running => run;

    volatile bool watch;

    public void Start(string d, string mode)
    {
        if (run) return;
        dir = d; learn = mode == "start"; watch = mode == "watch";
        bool eval = !learn;
        vis?.Dispose(); vis = Vision.Instance = Vision.TryLoad(AiService.AppCtx);   // har Start'da qayta yuklanadi (kodlovchi yangilangan bo'lishi mumkin)
        vecLen = vis != null ? Vision.Emb + Vision.Aux : NF;
        nIn = vecLen + 6;
        kind = vis != null ? "v2_" + vis.Version : "h2";   // siyosat kodlovchi versiyasiga bog'liq: yangi kodlovchi = toza siyosat (embedding siljishi DQN'ni buzadi)
        rep = new Replay(20000, nIn); net = null;
        AiService.SkipPerc = vis != null;      // SegFormer bo'sh-yo'l belgisi urilishni bashorat qilmaydi (AUC 0.50, 2026-10-21), ~2 yadro bekor ketardi
        Load();
        LoadRecords();
        run = true;
        loop = new Thread(Loop) { IsBackground = true, Name = "agent" }; loop.Start();
        if (learn) { trainer = new Thread(Train) { IsBackground = true, Name = "trainer" }; trainer.Start(); }
        Log.Info("AsphaltAI", $"agent start mode={mode} kind={kind} steps={steps} replay={rep.Count} nIn={nIn}");
    }

    public void Stop()
    {
        run = false;
        Thread.Sleep(StepMs + 100);
        AiService.SkipPerc = false;
        drv.Stop();
        Save();
        Status = "to'xtagan";
    }

    public void Reset()
    {
        if (run) return;
        if (dir != null)
            foreach (var f in System.IO.Directory.GetFiles(dir, "policy_*.bin").Concat(System.IO.Directory.GetFiles(dir, "replay_*.bin")))
                try { System.IO.File.Delete(f); } catch { }
        net = null; steps = 0; rep?.Clear();
    }

    // ---------- saqlash ----------
    void Load()
    {
        var rs = new Random(1);
        if (net == null)
        {
            net = new Mlp(new[] { nIn, 128, 128, NOut }, rs);
            net.SetOutBias(new float[] { 0, 0, .3f, 0, 0, /*drift*/ .3f, 0, 0, /*nitro*/ .3f, 0, 0 });
        }
        var p = System.IO.Path.Combine(dir, $"policy_{kind}.bin");
        if (System.IO.File.Exists(p))
            try
            {
                using var r = new System.IO.BinaryReader(System.IO.File.OpenRead(p));
                steps = r.ReadInt64(); net.Read(r);
            }
            catch (Exception e) { Log.Warn("AsphaltAI", "policy yuklanmadi: " + e.Message); }
        var rp = System.IO.Path.Combine(dir, $"replay_{kind}.bin");
        if (rep.Count == 0 && System.IO.File.Exists(rp))
            try { using var r = new System.IO.BinaryReader(System.IO.File.OpenRead(rp)); rep.Read(r); }
            catch (Exception e) { Log.Warn("AsphaltAI", "replay yuklanmadi: " + e.Message); }
        target = net.Clone();
    }

    void LoadRecords()
    {
        try
        {
            var p = System.IO.Path.Combine(dir, "records.json");
            records = System.IO.File.Exists(p) ? System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, double>>(System.IO.File.ReadAllText(p)) ?? new() : new();
        }
        catch (Exception e) { Log.Warn("AsphaltAI", "records: " + e.Message); records = new(); }
    }

    /// <summary>Marraga yetganda bonus: tugatish + (birinchi yoki yangi rekord bo'lsa) rekord bonusi, tejalgan soniyaga mutanosib.</summary>
    double FinishReward(long now)
    {
        double t = (now - epStart) / 1000.0, b = FinishBonus;
        records.TryGetValue(TrackKey, out double best);
        if (best <= 0) { b += RecordBase; records[TrackKey] = t; }
        else if (t < best) { b += RecordBase + RecordPerSec * Math.Min(RecordCap, best - t); records[TrackKey] = t; }
        try { System.IO.File.WriteAllText(System.IO.Path.Combine(dir, "records.json"), System.Text.Json.JsonSerializer.Serialize(records)); } catch { }
        Log.Info("AsphaltAI", $"FINISH track={TrackKey} time={t:F1}s rekord={records[TrackKey]:F1}s bonus={b:F1}");
        return b;
    }

    void Save()
    {
        if (net == null || !learn) return;
        try
        {
            lock (netLock)
                using (var w = new System.IO.BinaryWriter(System.IO.File.Create(System.IO.Path.Combine(dir, $"policy_{kind}.bin")))) { w.Write(steps); net.Write(w); }
            using var w2 = new System.IO.BinaryWriter(System.IO.File.Create(System.IO.Path.Combine(dir, $"replay_{kind}.bin")));
            rep.Write(w2);
        }
        catch (Exception e) { Log.Warn("AsphaltAI", "saqlash: " + e.Message); }
    }

    // ---------- holat ----------
    float[] State(int speed, int dS, float[] feat, long featAge, float[] last, float tFrac)
    {
        var s = new float[nIn]; int V = vecLen;
        if (feat != null) Array.Copy(feat, s, Math.Min(feat.Length, V));
        s[V] = speed / 200f; s[V + 1] = Math.Clamp(dS / 20f, -2, 2);
        s[V + 2] = last[0]; s[V + 3] = last[1]; s[V + 4] = last[2];
        s[V + 5] = vis != null ? Math.Clamp(tFrac, 0f, 1f) : Math.Min(featAge, 2000) / 2000f;   // ko'rishda: poyga soati (3:00 ga nisbatan)
        return s;
    }

    int Novelty(int speed, float[] feat, int steerIdx)
    {
        int cf = 0, car = 0, wall = 0;
        if (feat != null)
        {
            float m = 1; for (int k = 5; k <= 9; k++) m = Math.Min(m, feat[k * Perception.Feats]);
            cf = (int)(m * 3.99f);
            for (int k = 4; k <= 10; k++) { if (feat[k * Perception.Feats + 3] > .05f) car = 1; if (feat[k * Perception.Feats + 1] > .3f) wall = 1; }
        }
        int key = ((Math.Min(speed / 30, 6) * 4 + cf) * 2 + car) * 2 + wall;
        return key * 5 + steerIdx;
    }

    // ---------- asosiy sikl ----------
    readonly int[] hold = new int[4], cur = new int[4];
    readonly bool[] holdExpl = new bool[4];
    int explMask;   // shu qadamdagi tadqiqot (tasodifiy) shoxlari: BC ularni nusxalamasin

    // Tadqiqot: mantiqiy "bezovta qilish" — joriy harakat atrofida kichik burilishlar, gaz asosan yoqiq, makrolar kamdan-kam.
    // (Bir tekis tasodifiy tanlash mashinani yo'ldan chiqarib, o'rganish uchun natijani ko'rsatmaydi.)
    int Perturb(int b)
    {
        double u = rnd.NextDouble();
        switch (b)
        {
            case 0:
                hold[b] = rnd.Next(1, 4);
                if (u < .10) return rnd.Next(2) == 0 ? 0 : 4;                    // ba'zan keskin rul
                return Math.Clamp(cur[b] + (u < .55 ? -1 : u < .75 ? 0 : 1), 0, 4);
            case 1:
                if (u < .05) { hold[b] = rnd.Next(2, 6); return 1; }             // drift ushlash
                if (u < .06) return 2;                                           // 360
                return 0;
            case 2:
                if (u < .05) return 1;                                           // nitro
                if (u < .06) return 2;                                           // shockwave
                return 0;
            default:
                if (u < .03) { hold[b] = rnd.Next(1, 4); return 0; }             // gazni qisqa qo'yib yuborish
                return 1;
        }
    }

    int Pick(float[] q, int b, double eps)
    {
        if (hold[b] > 0) { hold[b]--; if (holdExpl[b]) explMask |= 1 << b; return cur[b]; }
        int n = Cnt[b], a;
        if (rnd.NextDouble() < eps) { a = Perturb(b); explMask |= 1 << b; }
        else
        {
            a = 0; for (int i = 1; i < n; i++) if (q[Off[b] + i] > q[Off[b] + a]) a = i;
            int def = b == 0 ? 2 : 0;      // Q farqi aniq ustun bo'lmasa: oddiy harakat (to'g'ri / drift yo'q / nitro yo'q). Q shovqini tasodifiy haydashga aylanmasin
            if (q[Off[b] + a] - q[Off[b] + def] < Margin[b]) a = def;
        }
        cur[b] = a; holdExpl[b] = (explMask & (1 << b)) != 0;
        return a;
    }

    void Apply(int[] a)
    {
        pad.Axis("X", SteerV[a[0]]);
        pad.Axis("RT", a[3] == 1 ? 1 : 0);
        // drift: 0 qo'yib yubor, 1 ushla, 2 ikki marta bos (360)
        if (a[1] == 1) pad.Button("LB", true); else pad.Button("LB", false);
        if (a[1] == 2) drv.Run("d360");
        if (a[2] == 1) drv.Run("nitro tap 80"); else if (a[2] == 2) drv.Run("shockwave");
    }

    /// <summary>Reward komponentlari (log va tekshiruv uchun alohida).</summary>
    struct Rw { public double Speed, Wall, Off, Steer, Prog, Nitro, WrongWay, Total; }

    static Rw Reward(int lastSteerIdx, int prev, int sp, double raPrev, double ra, double off, int dSteer, bool zig, int dDist, double dNitro, bool ww)
    {
        var w = new Rw();
        double sf = sp * (0.5 + 0.5 * ra), sfp = prev * (0.5 + 0.5 * raPrev), d = sf - sfp;   // oldinga tezlik ~ tezlik * yo'l oldinda
        if (d > 0) w.Speed += UpW * d;
        if (Math.Abs(sp - prev) <= StableBand) w.Speed += StableW * sf / 150.0;                 // barqaror ushlab turish
        double loss = (double)prev * prev - (double)sp * sp;                                   // yo'qotilgan kinetik energiya
        if (loss > 0) w.Wall = -WallW * loss / 10000.0;
        w.Off = -OffW * (sp / 100.0) * off;                                                     // yo'ldan tashqarida tezlik jazosi
        w.Steer = -SteerW * Math.Abs(dSteer) - (zig ? ZigW : 0) - (lastSteerIdx == 0 || lastSteerIdx == 4 ? ExtremeW : 0);                                 // silliq rul
        if (dDist > 0) w.Prog = ProgW * dDist; else if (dDist < 0) w.Prog = WrongWayW * dDist;   // HUD'dagi DIST% (bor bo'lsa)
        if (dNitro > 0) w.Nitro = NitroW * dNitro;                                              // nitro to'lishi
        if (ww) w.WrongWay = -WrongWayEvtW;
        w.Total = Math.Clamp(w.Speed + w.Wall + w.Off + w.Steer + w.Prog + w.Nitro + w.WrongWay, RMin, RMax);
        return w;
    }

    void Loop()
    {
        int prevSpeed = -1, prevDist = -1; int[] act = { 2, 0, 0, 1 }; float[] st = null; float[] lastIn = { 0, 0, 1 };
        double raPrev = .5, prevNitro = -1; int lastSteer = 2, lastDSteer = 0;
        var ring = new List<byte[]>();
        long ticks = 0;
        var win = new Queue<(float[] s, int[] a, double r)>();
        if (!watch) Apply(act);
        while (run)
        {
            Thread.Sleep(StepMs);
            long now = SystemClock.ElapsedRealtime();
            int sp = AiService.Speed;
            bool fresh = sp >= 0 && now - AiService.SpeedAt < StaleMs;
            if (!fresh)
            {
                Status = "HUD yo'q — kutilmoqda";
                if (prevSpeed >= 0) { pad.Command("reset"); prevSpeed = -1; prevDist = -1; st = null; win.Clear(); }
                continue;
            }
            if (prevSpeed < 0 && now - lastFresh > RaceGapMs) { epStart = now; finished = false; }   // HUD qisqa yo'qolishi (kamera/effekt) poygani qayta boshlamaydi
            lastFresh = now;
            var feat = AiService.Perc?.Last; long age = feat == null ? 2000 : now - AiService.Perc.LastAt;
            bool okF = feat != null && age <= 2500;
            double ra = okF ? feat[Perception.Bins * Perception.Feats] : .5, off = okF ? feat[Perception.Bins * Perception.Feats + 1] : 0;
            int dS = prevSpeed < 0 ? 0 : sp - prevSpeed;
            if (dS <= -25 && prevSpeed >= 0 && now - lastEvt > 4000 && evts < 40) { lastEvt = now; AiService.Perc?.SaveEvent($"{evts++}_{sp}"); }
            float[] vec = okF ? feat : null; float[] aux = null;
            var sn = AiService.Last;
            if (sn != null) { ring.Add(sn.Frame); if (ring.Count > 3) ring.RemoveAt(0); }
            if (vis != null && ring.Count > 0)
            {
                try { var (emb, ax) = vis.Run(ring[0], ring[Math.Min(1, ring.Count - 1)], ring[ring.Count - 1]); aux = ax; vec = new float[vecLen]; Array.Copy(emb, vec, Vision.Emb); Array.Copy(ax, 0, vec, Vision.Emb, Vision.Aux); }
                catch (Exception e) { Log.Warn("AsphaltAI", "vision: " + e.Message); vec = null; }
            }
            var s2 = State(sp, dS, vec, age, lastIn, (now - epStart) / 180000f);
            double eps = learn ? Math.Max(0.08, 0.6 * Math.Exp(-steps / 4000.0)) : 0.0;
            Rw rw = default; int dist = AiService.Dist, dDist = (dist >= 0 && prevDist >= 0) ? dist - prevDist : 0;
            double bonus = 0;
            if (dist >= 99 && ticks % 8 == 0) Log.Info("AsphaltAI", $"dist99 finished={finished} dt={(now - epStart) / 1000.0:F1}s");
            if (!finished && dist >= 99 && now - epStart > 20000) { finished = true; bonus = FinishReward(now); }
            if (st != null && Math.Abs(dS) < GlitchMph)
            {
                int dSteer = act[0] - lastSteer, ds = Math.Sign(dSteer);
                bool zig = ds != 0 && lastDSteer != 0 && ds != Math.Sign(lastDSteer);
                double nf = AiService.NitroFill, dN = prevNitro >= 0 ? nf - prevNitro : 0;
                rw = Reward(act[0], prevSpeed, sp, raPrev, ra, off, dSteer, zig, dDist, dN, AiService.WrongWay);
                if (ds != 0) lastDSteer = dSteer;
                int nk = Novelty(sp, okF ? feat : null, act[0]);
                seen.TryGetValue(nk, out int c); seen[nk] = c + 1;
                double rt = rw.Total + bonus + (learn ? 0.25 / Math.Sqrt(c + 1) : 0);
                if (learn)
                {
                    win.Enqueue((st, act, rt));
                    if (win.Count >= NStep)
                    {
                        double g = 1, sum = 0; foreach (var w in win) { sum += g * w.r; g *= Gamma; }
                        var first = win.Dequeue();
                        rep.Add(first.s, first.a, (float)sum, s2);          // n-qadamli qaytim: s_t, a_t, sum g^k r, s_{t+n}
                    }
                    if (bonus > 0)
                        while (win.Count > 0)                              // marra: qolgan o'tishlarni (bonusli) hammasini yozib qo'yamiz
                        {
                            double g2 = 1, sum2 = 0; foreach (var w2 in win) { sum2 += g2 * w2.r; g2 *= Gamma; }
                            var f2 = win.Dequeue();
                            rep.Add(f2.s, f2.a, (float)sum2, s2);
                        }
                }
            }
            lastSteer = act[0]; prevNitro = AiService.NitroFill;
            float[] q;
            lock (netLock) q = net.Forward(s2);
            var a = new int[4];
            explMask = 0;
            for (int b = 0; b < 3; b++) a[b] = Pick(q, b, eps);
            if (aux != null)
            {
                // "yo'l yo'q joy = to'siq": tanlangan rul tomonida urilish xavfi yuqori, boshqa tomon ancha bo'sh bo'lsa — bo'sh tomonga.
                // J13 (kodlovchi ko'rmagan): bunday holatlarda urilish 31% (o'rtacha 15%), eng xavfsiz rul olinganda 27% vs 41%.
                int best = 0; for (int k = 1; k < 5; k++) if (aux[12 + k] < aux[12 + best]) best = k;
                float tk = aux[12 + a[0]];
                if (tk >= SafeT && tk - aux[12 + best] >= SafeGap) { a[0] = best; explMask |= 8; overrides++; }
            }
            a[3] = 1;                                           // gaz doim bosiq
            if (!watch) Apply(a);
            lastIn = new[] { SteerV[a[0]], a[1] == 1 ? 1f : 0f, a[3] == 1 ? 1f : 0f };
            st = s2; act = a; prevSpeed = sp; raPrev = ra; if (dist >= 0) prevDist = dist; steps++; ticks++;
            Status = $"qadam {steps} tezlik {sp} dist {dist} eps {eps:F2} replay {rep.Count} loss {lastLoss:F3} bo'sh-tomon {overrides}";
            System.IO.File.AppendAllText(System.IO.Path.Combine(dir, "agent.csv"),
                $"{steps},{now},{sp},{a[0]},{a[1]},{a[2]},{a[3]},{rw.Total:F3},{eps:F2},{lastLoss:F4},{age},{rw.Speed:F3},{rw.Wall:F3},{rw.Off:F3},{rw.Steer:F3},{rw.Prog:F3},{dist},{ra:F2},{off:F2},{AiService.NitroFill:F2},{AiService.EvtActive:F3},{(aux != null ? aux[0] : 0):F3},{(AiService.WrongWay ? 1 : 0)},{bonus:F2},{explMask}\n");
            var snap = sn;
            if (snap != null)
            {
                recW ??= new System.IO.BinaryWriter(new System.IO.FileStream(System.IO.Path.Combine(dir, "rec.bin"), System.IO.FileMode.Append));
                recW.Write((int)steps); recW.Write(snap.T); recW.Write(snap.Frame); recW.Write(snap.Evt); recW.Write(snap.Nitro); recW.Flush();
            }
            if (ticks % 40 == 0) Log.Info("AsphaltAI", "agent " + Status);
            if (learn && ticks % 400 == 0) Save();
        }
        pad.Command("reset");
        recW?.Dispose(); recW = null;
    }

    // ---------- o'rganish ----------
    void Train()
    {
        var rs = new Random(5);
        int upd = 0;
        while (run)
        {
            if (!OnlineTrain || rep.Count < 400) { Thread.Sleep(200); continue; }
            for (int i = 0; i < 4; i++)
            {
                var b = rep.Sample(32, rs);
                lock (netLock) lastLoss = net.TrainBatch(b, target, Math.Pow(Gamma, NStep), NOut, Off, Cnt);
                if (++upd % 200 == 0) lock (netLock) target = net.Clone();
            }
            Thread.Sleep(30);
        }
    }
}

sealed class Replay
{
    readonly int cap, dim;
    readonly float[][] s, s2; readonly int[][] a; readonly float[] r;
    int n, head;
    public int Count => n;
    public Replay(int cap, int dim)
    {
        this.cap = cap; this.dim = dim; s = new float[cap][]; s2 = new float[cap][]; a = new int[cap][]; r = new float[cap];
    }
    public void Clear() { n = 0; head = 0; }
    public void Add(float[] st, int[] act, float rew, float[] nx)
    {
        lock (this)
        {
            s[head] = st; s2[head] = nx; a[head] = (int[])act.Clone(); r[head] = rew;
            head = (head + 1) % cap; n = Math.Min(n + 1, cap);
        }
    }
    public (float[][] s, int[][] a, float[] r, float[][] s2) Sample(int k, Random rs)
    {
        var bs = new float[k][]; var ba = new int[k][]; var br = new float[k]; var bn = new float[k][];
        lock (this)
            for (int i = 0; i < k; i++) { int j = rs.Next(n); bs[i] = s[j]; ba[i] = a[j]; br[i] = r[j]; bn[i] = s2[j]; }
        return (bs, ba, br, bn);
    }
    public void Write(System.IO.BinaryWriter w)
    {
        lock (this)
        {
            w.Write(n); w.Write(dim);
            for (int i = 0; i < n; i++)
            {
                int j = (head - n + i + cap) % cap;
                foreach (var v in s[j]) w.Write(v);
                foreach (var v in s2[j]) w.Write(v);
                foreach (var v in a[j]) w.Write((sbyte)v);
                w.Write(r[j]);
            }
        }
    }
    public void Read(System.IO.BinaryReader rd)
    {
        int cnt = rd.ReadInt32(), d = rd.ReadInt32();
        if (d != dim) return;
        for (int i = 0; i < cnt; i++)
        {
            var st = new float[dim]; var nx = new float[dim]; var ac = new int[4];
            for (int k = 0; k < dim; k++) st[k] = rd.ReadSingle();
            for (int k = 0; k < dim; k++) nx[k] = rd.ReadSingle();
            for (int k = 0; k < 4; k++) ac[k] = rd.ReadSByte();
            Add(st, ac, rd.ReadSingle(), nx);
        }
    }
}

sealed class Mlp
{
    readonly int[] L; readonly float[][] W, B, mW, vW, mB, vB; int t;

    public Mlp(int[] sizes, Random rs)
    {
        L = sizes; int n = sizes.Length - 1;
        W = new float[n][]; B = new float[n][]; mW = new float[n][]; vW = new float[n][]; mB = new float[n][]; vB = new float[n][];
        for (int l = 0; l < n; l++)
        {
            W[l] = new float[L[l] * L[l + 1]]; B[l] = new float[L[l + 1]];
            mW[l] = new float[W[l].Length]; vW[l] = new float[W[l].Length]; mB[l] = new float[B[l].Length]; vB[l] = new float[B[l].Length];
            double sd = Math.Sqrt(2.0 / L[l]);
            for (int i = 0; i < W[l].Length; i++) W[l][i] = (float)(sd * Gauss(rs));
        }
    }
    static double Gauss(Random r) => Math.Sqrt(-2 * Math.Log(1 - r.NextDouble())) * Math.Cos(2 * Math.PI * r.NextDouble());

    public void SetOutBias(float[] b) { Array.Copy(b, B[W.Length - 1], b.Length); }

    public Mlp Clone()
    {
        var m = new Mlp(L, new Random(0));
        for (int l = 0; l < W.Length; l++) { Array.Copy(W[l], m.W[l], W[l].Length); Array.Copy(B[l], m.B[l], B[l].Length); }
        return m;
    }

    public float[] Forward(float[] x) => Forward(x, null);

    float[] Forward(float[] x, float[][] acts)
    {
        var h = x;
        if (acts != null) acts[0] = x;
        for (int l = 0; l < W.Length; l++)
        {
            int ni = L[l], no = L[l + 1];
            var o = new float[no];
            for (int j = 0; j < no; j++)
            {
                float sum = B[l][j];
                int wo = j * ni;
                for (int i = 0; i < ni; i++) sum += W[l][wo + i] * h[i];
                o[j] = l < W.Length - 1 ? Math.Max(0, sum) : sum;
            }
            h = o;
            if (acts != null) acts[l + 1] = o;
        }
        return h;
    }

    /// <summary>Shoxlangan DQN: har shox uchun maqsad = r + gamma * o'rtacha_shox(max Q_target'). Qaytaradi: o'rtacha yo'qotish.</summary>
    public float TrainBatch((float[][] s, int[][] a, float[] r, float[][] s2) b, Mlp target, double gamma, int nOut, int[] off, int[] cnt)
    {
        int n = W.Length, k = b.s.Length;
        var gW = new float[n][]; var gB = new float[n][];
        for (int l = 0; l < n; l++) { gW[l] = new float[W[l].Length]; gB[l] = new float[B[l].Length]; }
        double loss = 0;
        for (int i = 0; i < k; i++)
        {
            var qn = target.Forward(b.s2[i]);
            double nextMean = 0;
            for (int h = 0; h < cnt.Length; h++)
            {
                float m = float.MinValue;
                for (int j = 0; j < cnt[h]; j++) m = Math.Max(m, qn[off[h] + j]);
                nextMean += m;
            }
            nextMean /= cnt.Length;
            double y = b.r[i] + gamma * nextMean;
            var acts = new float[n + 1][];
            var q = Forward(b.s[i], acts);
            var d = new float[nOut];
            for (int h = 0; h < cnt.Length; h++)
            {
                int idx = off[h] + b.a[i][h];
                float err = (float)(q[idx] - y);
                err = Math.Clamp(err, -5f, 5f);       // Huber
                d[idx] = err / cnt.Length; loss += err * err / cnt.Length;
            }
            var delta = d;
            for (int l = n - 1; l >= 0; l--)
            {
                int ni = L[l], no = L[l + 1];
                var prev = new float[ni];
                for (int j = 0; j < no; j++)
                {
                    float dj = delta[j];
                    if (dj == 0) continue;
                    gB[l][j] += dj;
                    int wo = j * ni;
                    for (int ii = 0; ii < ni; ii++) { gW[l][wo + ii] += dj * acts[l][ii]; prev[ii] += dj * W[l][wo + ii]; }
                }
                if (l > 0) for (int ii = 0; ii < ni; ii++) if (acts[l][ii] <= 0) prev[ii] = 0;
                delta = prev;
            }
        }
        t++;
        const float lr = 3e-4f, b1 = .9f, b2 = .999f, eps = 1e-8f;
        float c1 = 1 - MathF.Pow(b1, t), c2 = 1 - MathF.Pow(b2, t);
        for (int l = 0; l < n; l++)
        {
            for (int i = 0; i < W[l].Length; i++) Adam(ref W[l][i], gW[l][i] / k, ref mW[l][i], ref vW[l][i]);
            for (int i = 0; i < B[l].Length; i++) Adam(ref B[l][i], gB[l][i] / k, ref mB[l][i], ref vB[l][i]);
        }
        void Adam(ref float p, float g, ref float m, ref float v)
        {
            m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g;
            p -= lr * (m / c1) / (MathF.Sqrt(v / c2) + eps);
        }
        return (float)(loss / k);
    }

    public void Write(System.IO.BinaryWriter w)
    {
        w.Write(L.Length); foreach (var s in L) w.Write(s);
        for (int l = 0; l < W.Length; l++) { foreach (var v in W[l]) w.Write(v); foreach (var v in B[l]) w.Write(v); }
    }

    public void Read(System.IO.BinaryReader r)
    {
        int nl = r.ReadInt32(); var sz = new int[nl];
        for (int i = 0; i < nl; i++) sz[i] = r.ReadInt32();
        if (!sz.SequenceEqual(L)) throw new InvalidDataException("arxitektura mos emas");
        for (int l = 0; l < W.Length; l++) { for (int i = 0; i < W[l].Length; i++) W[l][i] = r.ReadSingle(); for (int i = 0; i < B[l].Length; i++) B[l][i] = r.ReadSingle(); }
    }
}
