using Android.App;
using Android.Content;
using Android.OS;
using Android.Widget;

namespace AsphaltApp;

[Activity(Label = "AsphaltAI", MainLauncher = true, Exported = true)]
public class MainActivity : Activity
{
    const int Req = 77;
    TextView info;
    readonly Handler ui = new Handler(Looper.MainLooper);

    protected override void OnCreate(Bundle b)
    {
        base.OnCreate(b);
        var l = new LinearLayout(this) { Orientation = Orientation.Vertical };
        l.SetPadding(32, 32, 32, 32);
        info = new TextView(this) { TextSize = 20 };
        var cap = new Button(this) { Text = "Ekranni yozishni boshlash" };
        cap.Click += (_, _) =>
        {
            RequestPermissions(new[] { "android.permission.POST_NOTIFICATIONS" }, 1);
            var mgr = (Android.Media.Projection.MediaProjectionManager)GetSystemService(MediaProjectionService);
            StartActivityForResult(mgr.CreateScreenCaptureIntent(), Req);
        };
        var pad = new Button(this) { Text = "Virtual gamepad yaratish" };
        pad.Click += (_, _) =>
        {
            var r = AiService.Gamepad.Created ? "ok (bor)" : AiService.Gamepad.Create();
            Android.Util.Log.Info("AsphaltAI", "pad create: " + r);
            Toast.MakeText(this, "pad: " + r, ToastLength.Long).Show();
        };
        var test = new Button(this) { Text = "Sinov kadrlar" };
        test.Click += (_, _) => StartActivity(new Intent(this, typeof(TestActivity)));
        l.AddView(info); l.AddView(cap); l.AddView(pad); l.AddView(test);
        SetContentView(l);
        Tick();
    }

    void Tick()
    {
        info.Text = $"Tezlik: {(AiService.Speed < 0 ? "-" : AiService.Speed.ToString())}\nHolat: {AiService.Status}\nPad: {(AiService.Gamepad.Created ? "bor" : "yo'q")}";
        ui.PostDelayed(Tick, 300);
    }

    protected override void OnActivityResult(int req, Result res, Intent data)
    {
        base.OnActivityResult(req, res, data);
        if (req != Req || res != Result.Ok) return;
        var i = new Intent(this, typeof(AiService));
        i.PutExtra("code", (int)res);
        i.PutExtra("data", data);
        StartForegroundService(i);
    }
}
