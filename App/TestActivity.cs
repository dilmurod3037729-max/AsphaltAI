using Android.App;
using Android.Content.PM;
using Android.Graphics;
using Android.OS;
using Android.Views;
using Android.Widget;

namespace AsphaltApp;

/// <summary>Poyga HUD'i bor statik kadrlarni to'liq ekranda aylantiradi: yozuv zanjirini o'yinsiz sinash uchun.</summary>
[Activity(Label = "Sinov", Theme = "@android:style/Theme.Material.NoActionBar.Fullscreen", Exported = false, ScreenOrientation = ScreenOrientation.Landscape)]
public class TestActivity : Activity
{
    static readonly string[] Files = { "t1.png", "t2.png", "t3.png" };
    ImageView iv; int i;
    readonly Handler ui = new Handler(Looper.MainLooper);

    protected override void OnCreate(Bundle b)
    {
        base.OnCreate(b);
        iv = new ImageView(this);
        iv.SetScaleType(ImageView.ScaleType.FitXy);
        SetContentView(iv);
        Window.DecorView.SystemUiFlags = (SystemUiFlags)(0x1000 | 0x4 | 0x2 | 0x200);
        Next();
    }

    void Next()
    {
        using (var s = Assets.Open(Files[i++ % Files.Length])) iv.SetImageBitmap(BitmapFactory.DecodeStream(s));
        ui.PostDelayed(Next, 2000);
    }
}
