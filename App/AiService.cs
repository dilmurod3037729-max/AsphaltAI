using Android.Runtime;
using Android.App;
using Android.Content;
using Android.Content.PM;
using Android.Graphics;
using Android.Hardware.Display;
using Android.Media;
using Android.Media.Projection;
using Android.OS;
using Android.Util;
using Android.Views;

namespace AsphaltApp;

[Service(Exported = false, ForegroundServiceType = ForegroundService.TypeMediaProjection)]
public class AiService : Service
{
    public const string Tag = "AsphaltAI", PadAction = "space.asphaltai.PAD";
    public static volatile int Speed = -1;
    public static volatile string Status = "to'xtagan";
    public static readonly Pad Gamepad = new();
    public static readonly Driver Drv = new(Gamepad);
    public static readonly Agent Ag = new(Drv, Gamepad);
    public static long SpeedAt;
    public static volatile int Dist = -1;
    public static volatile bool WrongWay;                 // o'yinning "WRONG WAY!" belgisi (qizil X) kadrda turibdimi
    int distStable = -1, distCand = -1, distCnt;
    /// <summary>Oxirgi kadrning kichik nusxalari (offlayn o'rgatish uchun yoziladi): butun kadr 96x48 RGB, hodisa yozuvlari ustuni 128x40 kul, nitro chizig'i 96x8 RGB.</summary>
    public sealed class Snap { public byte[] Frame = new byte[96 * 48 * 3], Evt = new byte[128 * 40], Nitro = new byte[96 * 8 * 3]; public long T; }
    public static volatile Snap Last;
    long lastSnap;
    public static Perception Perc;
    public static volatile bool SkipPerc;                 // o'rganilgan ko'rish bor va o'rgatilmayotgan bo'lsa tayyor modellar o'chiriladi
    public static Context AppCtx;
    public static volatile float NitroFill, EvtActive;    // HUD o'lchovlari: nitro chizig'i to'liqligi (0..1), hodisa yozuvlari faolligi (0..1)
    Reader distReader; long lastDist;
    // DIST% raqamlari (1560x720 koordinatalarida x,y,w,h; '%' belgisi QAMRAMAYDI): yakka poyga va jamoaviy poyga satrlari
    static readonly int[][] DistBoxes = { new[] { 250, 80, 63, 38 }, new[] { 250, 24, 63, 38 } };

    MediaProjection proj; VirtualDisplay vd; ImageReader reader;
    Reader speedReader; HandlerThread ht; Handler hnd;
    int w, ht2, dpi; long lastLog; int lastSpeed = -2;
    PadReceiver rcv;

    public override IBinder OnBind(Intent intent) => null;

    public override StartCommandResult OnStartCommand(Intent intent, StartCommandFlags flags, int startId)
    {
        var nm = (NotificationManager)GetSystemService(NotificationService);
        nm.CreateNotificationChannel(new NotificationChannel("ai", "AsphaltAI", NotificationImportance.Low));
        var n = new Notification.Builder(this, "ai").SetContentTitle("AsphaltAI").SetContentText("tezlik o'qilmoqda").SetSmallIcon(Android.Resource.Drawable.IcMediaPlay).Build();
        StartForeground(1, n, ForegroundService.TypeMediaProjection);

        if (rcv == null)
        {
            rcv = new PadReceiver();
            RegisterReceiver(rcv, new IntentFilter(PadAction), ReceiverFlags.Exported);
        }
        var code = intent.GetIntExtra("code", 0);
        var data = intent.GetParcelableExtra("data")?.JavaCast<Intent>();
        if (data != null && proj == null) StartCapture(code, data);
        return StartCommandResult.NotSticky;
    }

    void StartCapture(int code, Intent data)
    {
        var tmp = System.IO.Path.Combine(CacheDir.AbsolutePath, "templates.txt");
        using (var s = Assets.Open("templates.txt"))
        using (var f = System.IO.File.Create(tmp)) s.CopyTo(f);
        speedReader = Reader.Load(tmp);
        AppCtx = ApplicationContext;
        try
        {
            if (Assets.List("")?.Contains("dist_templates.txt") == true)
            {
                var dp = System.IO.Path.Combine(CacheDir.AbsolutePath, "dist_templates.txt");
                using (var s2 = Assets.Open("dist_templates.txt")) using (var f2 = System.IO.File.Create(dp)) s2.CopyTo(f2);
                distReader = Reader.Load(dp);
            }
        }
        catch (Exception e) { Log.Warn(Tag, "dist shablonlari: " + e.Message); }
        Perc = new Perception(this) { DebugPath = System.IO.Path.Combine(GetExternalFilesDir(null).AbsolutePath, "dbg.png"), LogPath = System.IO.Path.Combine(GetExternalFilesDir(null).AbsolutePath, "perc.csv"), EventDir = GetExternalFilesDir(null).AbsolutePath };
        ht = new HandlerThread("ai"); ht.Start(); hnd = new Handler(ht.Looper);
        var mgr = (MediaProjectionManager)GetSystemService(MediaProjectionService);
        proj = mgr.GetMediaProjection(code, data);
        proj.RegisterCallback(new ProjCb(), hnd);
        Build();
        Status = "yozmoqda " + w + "x" + ht2;
        hnd.PostDelayed(Watch, 1000);
    }

    class ProjCb : MediaProjection.Callback { public override void OnStop() { Status = "proyeksiya to'xtadi"; } }

    void Build()
    {
        var wm = GetSystemService(WindowService).JavaCast<IWindowManager>();
        var m = new DisplayMetrics();
        wm.DefaultDisplay.GetRealMetrics(m);
        w = m.WidthPixels; ht2 = m.HeightPixels; dpi = (int)m.DensityDpi;
        reader?.Close();
        reader = ImageReader.NewInstance(w, ht2, (Android.Graphics.ImageFormatType)1, 2);
        reader.SetOnImageAvailableListener(new Listener(this), hnd);
        if (vd == null) vd = proj.CreateVirtualDisplay("ai", w, ht2, dpi, (DisplayFlags)16, reader.Surface, null, null);
        else { vd.Resize(w, ht2, dpi); vd.Surface = reader.Surface; }
    }

    void Watch()
    {
        var wm = GetSystemService(WindowService).JavaCast<IWindowManager>();
        var m = new DisplayMetrics(); wm.DefaultDisplay.GetRealMetrics(m);
        if (m.WidthPixels != w || m.HeightPixels != ht2) { Build(); Status = "yozmoqda " + w + "x" + ht2; }
        hnd.PostDelayed(Watch, 1000);
    }

    class Listener : Java.Lang.Object, ImageReader.IOnImageAvailableListener
    {
        readonly AiService s; public Listener(AiService s) { this.s = s; }
        public void OnImageAvailable(ImageReader r) => s.OnFrame(r);
    }

    unsafe void OnFrame(ImageReader r)
    {
        var img = r.AcquireLatestImage();
        if (img == null) return;
        try { Process(img); } finally { img.Close(); }
    }

    unsafe void Process(Image img)
    {
        var pl = img.GetPlanes()[0];
        var buf = pl.Buffer;
        int stride = pl.RowStride, W = img.Width, H = img.Height;
        if (W < H) { Status = "portret — poyga kutilmoqda"; return; }
        double k = W / 1560.0;
        int x0 = (int)(1340 * k), y0 = (int)(20 * k), cw = Math.Min(W - x0, (int)(220 * k)), ch = Math.Min(H - y0, (int)(90 * k));
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        var span = new ReadOnlySpan<byte>((void*)buf.GetDirectBufferAddress(), buf.Capacity());
        var f = Frame.FromRgba(span, stride, x0, y0, cw, ch);
        var res = speedReader.Read(f);
        long us = (System.Diagnostics.Stopwatch.GetTimestamp() - t0) * 1000000 / System.Diagnostics.Stopwatch.Frequency;
        Speed = res.Value ?? -1;
        if (!SkipPerc) Perc?.Submit(span, stride, W, H);
        long nowT = SystemClock.ElapsedRealtime();
        if (distReader != null && nowT - lastDist >= 240)
        {
            lastDist = nowT;
            int dv = -1;
            foreach (var b in DistBoxes)
            {
                int bx = (int)(b[0] * k), by = (int)(b[1] * k), bw = (int)(b[2] * k), bh = (int)(b[3] * k);
                if (bx + bw > W || by + bh > H) continue;
                var p = distReader.ReadPercent(Frame.FromRgba(span, stride, bx, by, bw, bh));
                if (p != null) { dv = p.Value; break; }
            }
            if (dv >= 0)
            {
                // filtr: faqat [joriy, joriy+3] oralig'idagi o'qishlar qabul qilinadi; boshqasi 4 marta ketma-ket tasdiqlansagina (xato o'qish raqam tashlab ketadi)
                if (distStable < 0 || (dv >= distStable && dv <= distStable + 3)) { distStable = dv; distCnt = 0; }
                else
                {
                    if (distCnt > 0 && Math.Abs(dv - distCand) <= 1) distCnt++; else { distCand = dv; distCnt = 1; }
                    if (distCnt >= 4) { distStable = dv; distCnt = 0; }
                }
            }
            if (res.Value == null && nowT - SpeedAt > 1500) distStable = -1;       // HUD yo'q (menyu): holatni tashla
            Dist = distStable;
        }
        if (SystemClock.ElapsedRealtime() - lastSnap >= 100)
        {
            lastSnap = SystemClock.ElapsedRealtime();
            var sn = new Snap { T = lastSnap };
            for (int y = 0; y < 48; y++)
                for (int x = 0; x < 96; x++)
                {
                    int o = (y * H / 48) * stride + (x * W / 96) * 4, d = (y * 96 + x) * 3;
                    sn.Frame[d] = span[o]; sn.Frame[d + 1] = span[o + 1]; sn.Frame[d + 2] = span[o + 2];
                }
            int ex0 = (int)(.66 * W), ey0 = (int)(.33 * H), ew = (int)(.34 * W), eh = (int)(.22 * H);
            for (int y = 0; y < 40; y++)
                for (int x = 0; x < 128; x++)
                {
                    int o = (ey0 + y * eh / 40) * stride + (ex0 + x * ew / 128) * 4;
                    sn.Evt[y * 128 + x] = (byte)((span[o] * 77 + span[o + 1] * 150 + span[o + 2] * 29) >> 8);
                }
            int nx0 = (int)(.30 * W), ny0 = (int)(.02 * H), nw = (int)(.38 * W), nh = (int)(.07 * H);
            for (int y = 0; y < 8; y++)
                for (int x = 0; x < 96; x++)
                {
                    int o = (ny0 + y * nh / 8) * stride + (nx0 + x * nw / 96) * 4, d = (y * 96 + x) * 3;
                    sn.Nitro[d] = span[o]; sn.Nitro[d + 1] = span[o + 1]; sn.Nitro[d + 2] = span[o + 2];
                }
            Last = sn;
            int pinkN = 0;
            for (int y = 8; y < 34; y++)
                for (int x = 28; x < 68; x++)
                {
                    int d = (y * 96 + x) * 3, pr = sn.Frame[d], pg = sn.Frame[d + 1], pb = sn.Frame[d + 2];
                    if (pr > 200 && pg < 110 && pb > 40 && pb < 170 && pr - pg > 110) pinkN++;
                }
            WrongWay = pinkN >= 30;
            int filled = 0;
            for (int x = 0; x < 96; x++)
            {
                int sr = 0, sg = 0, sb = 0;
                for (int y = 2; y < 6; y++) { int d = (y * 96 + x) * 3; sr += sn.Nitro[d]; sg += sn.Nitro[d + 1]; sb += sn.Nitro[d + 2]; }
                int mx = Math.Max(sr, Math.Max(sg, sb)) / 4, mn = Math.Min(sr, Math.Min(sg, sb)) / 4;
                if (mx > 120 && mx - mn > 80) filled++;
            }
            NitroFill = filled / 96f;
            int bright = 0; foreach (var v in sn.Evt) if (v > 200) bright++;
            EvtActive = bright / (float)sn.Evt.Length;
        }
        if (res.Value != null) SpeedAt = SystemClock.ElapsedRealtime();
        long now = SystemClock.ElapsedRealtime();
        if ((Speed != lastSpeed && now - lastLog > 250) || now - lastLog > 2000)
        {
            lastLog = now; lastSpeed = Speed;
            Log.Info(Tag, $"speed={Speed} digits={res.Digits} score={res.MinScore:F2} read={us}us");
        }
    }

    class PadReceiver : BroadcastReceiver
    {
        public override void OnReceive(Context c, Intent i)
        {
            var cmd = i.GetStringExtra("cmd") ?? "";
            string r;
            if (cmd == "ai start" || cmd == "ai eval" || cmd == "ai watch") { Ag.Start(c.GetExternalFilesDir(null).AbsolutePath, cmd.Substring(3)); r = "ok"; }
            else if (cmd == "ai stop") { Ag.Stop(); r = "ok"; }
            else if (cmd == "ai reset") { Ag.Reset(); r = "ok"; }
            else if (cmd.StartsWith("ai track ")) { Ag.TrackKey = cmd.Substring(9).Trim(); r = "ok"; }
            else if (cmd == "create") r = Gamepad.Created ? "ok (bor)" : Gamepad.Create();
            else r = Drv.Run(cmd);
            Log.Info(Tag, $"pad '{cmd}' -> {r}");
        }
    }

    public override void OnDestroy()
    {
        if (rcv != null) UnregisterReceiver(rcv);
        vd?.Release(); reader?.Close(); proj?.Stop(); ht?.QuitSafely();
        base.OnDestroy();
    }
}
