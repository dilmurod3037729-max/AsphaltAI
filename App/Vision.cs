using Android.Content;
using Android.Util;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace AsphaltApp;

/// <summary>O'rganilgan ko'rish kodlovchisi (PC'da o'rgatilgan CNN, ONNX). Kirish: ketma-ket 3 ta 96x48 RGB kadr (9 kanal).
/// Chiqish: embedding (128) + yordamchi bashoratlar (to'qnashuv ehtimoli, tezlik yo'qotishi, tezlik o'zgarishi, nitro o'sishi, hodisa sinflari).
/// Fayllar bo'lmasa Available=false: agent qo'lda hisoblangan xususiyatlarga qaytadi.</summary>
public sealed class Vision : IDisposable
{
    public const int Emb = 128, Aux = 18, W = 96, H = 48;
    readonly InferenceSession sess;
    public string Version { get; }
    public static Vision Instance;

    Vision(InferenceSession s, string ver) { sess = s; Version = ver; }

    public static Vision TryLoad(Context ctx)
    {
        try
        {
            // 1) tashqi fayl (adb bilan yangilanadi, APK qayta qurilmaydi), 2) APK assets
            var ext = ctx.GetExternalFilesDir(null)?.AbsolutePath;
            string p, ver = "0";
            if (ext != null && System.IO.File.Exists(System.IO.Path.Combine(ext, "vision.onnx")))
            {
                p = System.IO.Path.Combine(ext, "vision.onnx");
                var vp = System.IO.Path.Combine(ext, "vision.ver");
                if (System.IO.File.Exists(vp)) ver = System.IO.File.ReadAllText(vp).Trim();
            }
            else
            {
                var list = ctx.Assets.List("") ?? Array.Empty<string>();
                if (!list.Contains("vision.onnx")) return null;
                p = System.IO.Path.Combine(ctx.CacheDir.AbsolutePath, "vision.onnx");
                using (var s = ctx.Assets.Open("vision.onnx")) using (var f = System.IO.File.Create(p)) s.CopyTo(f);
                if (list.Contains("vision.ver")) using (var r = new System.IO.StreamReader(ctx.Assets.Open("vision.ver"))) ver = r.ReadToEnd().Trim();
            }
            var o = new SessionOptions { IntraOpNumThreads = 2 };
            return new Vision(new InferenceSession(p, o), ver);
        }
        catch (Exception e)
        {
            Log.Warn("AsphaltAI", "vision yuklanmadi: " + e.Message);
            return null;
        }
    }

    /// <summary>f0 (eng eski), f1, f2 (hozirgi): har biri 96*48*3 bayt RGB. Qaytaradi (embedding, aux).</summary>
    public (float[] emb, float[] aux) Run(byte[] f0, byte[] f1, byte[] f2)
    {
        var x = new float[9 * H * W];
        var fr = new[] { f0, f1, f2 };
        for (int k = 0; k < 3; k++)
            for (int c = 0; c < 3; c++)
            {
                int o = (k * 3 + c) * H * W;
                for (int i = 0; i < H * W; i++) x[o + i] = fr[k][i * 3 + c] / 255f;
            }
        using var res = sess.Run(new[] { NamedOnnxValue.CreateFromTensor("x", new DenseTensor<float>(x, new[] { 1, 9, H, W })) });
        var outs = res.ToArray();
        return (outs[0].AsTensor<float>().ToArray(), outs[1].AsTensor<float>().ToArray());
    }

    public void Dispose() => sess.Dispose();
}
